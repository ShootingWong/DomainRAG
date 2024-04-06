import torch
from transformers import BertTokenizer, BertModel
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import torch.nn.functional as F
import torch.nn as nn
import collections
from collections import defaultdict
import re
# import openai
from tqdm import tqdm
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
from multiprocessing import Lock, Pool
import json
import sys

import sys
import os
import json
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

class OpenAIApiException(Exception):
    def __init__(self, msg, error_code):
        self.msg = msg
        self.error_code = error_code

class OpenAIApiProxy():
    def __init__(self, openai_api, api_key=None):
        self.openai_api = openai_api
        retry_strategy = Retry(
            total=3,  # 最大重试次数（包括首次请求）
            backoff_factor=5, # 重试之间的等待时间因子
            status_forcelist=[429, 500, 502, 503, 504], # 需要重试的状态码列表  
            allowed_methods=["POST"] # 只对POST请求进行重试
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.api_key = api_key
    def call(self, params_gpt, headers={}):
        headers['Content-Type'] = headers['Content-Type'] if 'Content-Type' in headers else 'application/json'
        if self.api_key:
            headers['Authorization'] = "Bearer " + self.api_key
        if self.openai_api != 'https://api.openai.com/v1/completions':
            url = self.openai_api + '/v1/chat/completions'
        else:
            url = self.openai_api
        # print('Call url = ', url)
        
        myflag = True
        while(myflag):
            try:
                response = self.session.post(url, headers=headers, data=json.dumps(params_gpt))
                # print("response", response.text)
                #while response.status_code != 200:
                #    err_msg = "access openai error, status code: {myStatusCode}，errmsg: {myResponse}, exception: {e}, sleeping 20s ..."
                #    response = self.session.post(url, headers=headers, data=json.dumps(params_gpt))
                myflag = False
            except Exception as e:
                print("access openai error, sleeping 20s ...")
                print(e)
                sys.stdout.flush()
                time.sleep(20)
        data = json.loads(response.text)
        return data

def get_answer(proxy, model, prompt):

    headers = {}
    prompt_dict = {
        "model": model,
        # "messages": [json.dumps({"role": "system", "content":prompt})]
        # "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content":prompt}],
        "temperature": 0.01
    }

    ans_json = proxy.call(prompt_dict)

    print(ans_json)
    try:
        resp = ans_json["choices"][0]["message"]["content"]
    except:
        if ans_json['error']['code'] == 'context_length_exceeded':
            resp = ''
        else:
            resp = ans_json["choices"][0]["message"]["content"]
    return resp

def gpt_generate_str(proxy, model, input_str):
    answer = get_answer(proxy, model, input_str)

    return answer

def runProcess_gpt(inputs):
    strs, proxy, model = inputs[:]
    res_list = []
    for s in strs:
        res = gpt_generate_str(proxy, model, s)
        res_list.append(res)

    return res_list

def gpt_generate(input_str, pool_size, proxy, model):
    
    input_lists = []
    bsz = len(input_str)
    if pool_size > bsz: pool_size = bsz
    per_cnt = bsz // pool_size 
    

    for i in range(pool_size):
        bg = i * per_cnt
        ed = (i+1) * per_cnt if i < pool_size-1 else bsz
        input_lists.append([input_str[bg:ed], proxy, model])

    pool = Pool(pool_size)
    res_list = pool.map(runProcess_gpt, input_lists)
    final_list = sum(res_list, [])

    return final_list
    


class MyLLM:
    def __init__(self, opt, model, tokenizer, evaluator, sample_params=None):
        self.opt = opt
        self.infer_batch = opt.infer_batch
        self.test_save_path = opt.test_save_path
        if os.path.exists(self.test_save_path):
            os.remove(self.test_save_path)
        self.html_root = opt.html_root
        self.docs_root = opt.docs_root
        self.model_name = opt.llm_name
        
        print(f'self.model_name = {self.model_name}')
        
        self.model = model 
        self.reader_tokenizer = tokenizer
        self.evaluator = evaluator
        
        if tokenizer is not None:
            if 'chatglm' not in self.model_name:
                self.reader_tokenizer.pad_token = self.reader_tokenizer.eos_token
            self.reader_tokenizer.padding_side = "left"
            
        
        self.sample_params = sample_params
        
        
        self.lock = Lock()
        if self.opt.gpt:
            self.proxy = OpenAIApiProxy(opt.openai_api, api_key=opt.apikey)
            self.model = opt.llm_name
            self.pool_size = opt.infer_batch
        if self.opt.baichuan:
            self.pool_size = opt.infer_batch
            
        if self.opt.fastchat:
            if 'chatglm' not in self.model_name: 
                self.max_input_length = model.config.max_position_embeddings
            else:
                self.max_input_length = model.config.seq_length

            print(f'for Model {self.model_name}, self.max_input_length = {self.max_input_length}')
        
        
    def load_docs(self, root):
        paths = [os.path.join(root, n) for n in os.listdir(root)]
        docs = []
        for path in tqdm(paths):
            if '.json' not in path: continue
            data = json.loads(open(path).read())
            docs.append(data)

    def parse_generate(self, batch_input_ids, batch_generate_ids):
        responses = []

        bsz, max_input = batch_input_ids.size()[:2]
        for i, generated_sequence in enumerate(batch_generate_ids):
            input_ids = batch_input_ids[i]
            text = self.reader_tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if input_ids is None:
                prompt_length = 0
            else:
                prompt_length = len(
                    self.reader_tokenizer.decode(
                        input_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )
            new_text = text[prompt_length:]
            responses.append(new_text.strip())

        return responses

    @torch.no_grad()
    def get_llm_output(self, reader_input_strs):
        bsz = len(reader_input_strs)
        if self.opt.fastchat:
            
            reader_inputs = self.reader_tokenizer(reader_input_strs, padding='longest', return_tensors='pt')
            out_txt_seqs = []
            
            for i in tqdm(range(0, bsz, self.infer_batch)):
                bg, ed = i, i+self.infer_batch
                max_input_tokens = self.max_input_length - self.opt.generation_max_length
                
                reader_input_ids = reader_inputs['input_ids'][bg:ed].to(self.model.device)[:, -max_input_tokens:]
                reader_attetion_mask = reader_inputs['attention_mask'][bg:ed].to(self.model.device)[:, -max_input_tokens:]

                reader_output = self.model.generate(
                    input_ids=reader_input_ids,
                    attention_mask=reader_attetion_mask,
                    max_new_tokens=self.opt.generation_max_length,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    do_sample=False, 
                    # output_scores=True,
                    use_cache=True,
                ) 
                
                reader_output = reader_output['sequences']
                per_out_txt_seqs = self.parse_generate(reader_input_ids, reader_output)
                out_txt_seqs += per_out_txt_seqs
        elif self.opt.vllm:
            out_txt_seqs = []
            for i in range(0, bsz, self.infer_batch):
                bg, ed = i, min(i+self.infer_batch, bsz)
               
                prompt_token_ids = self.reader_tokenizer(reader_input_strs[bg: ed], return_tensors='pt', padding=True, truncation=True)['input_ids']
                max_input_tokens = self.max_input_length - self.opt.generation_max_length
                prompt_token_ids = prompt_token_ids[:, -max_input_tokens:]
               
                reader_output = self.model.generate(prompt_token_ids=prompt_token_ids.tolist(), sampling_params=self.sample_params)

                per_out_txt_seqs = [output.outputs[0].text for output in reader_output]
    
                out_txt_seqs += per_out_txt_seqs
                
        elif self.opt.gpt:
            out_txt_seqs = gpt_generate(reader_input_strs, self.pool_size, self.proxy, self.model)

        return out_txt_seqs
    
    def test_batch(self, batch_data):
        
        if self.opt.task in ["basic", "time", "multidoc", "conversation", "noisy"]:
            return self.test_basic(batch_data)
        elif self.opt.task == 'structure':
            return self.test_structure(batch_data)
        elif self.opt.task == 'counterfactual':
            return self.test_counterfact(batch_data)
        
        elif self.opt.task == 'conversation':
            return self.test_conversation(batch_data)
        else:
            raise Exception(f"Task {self.opt.task } is inavaliable")

    
    def save_data(self, batch_data, pred):
        cnt = len(pred)
        for i in range(cnt):
            per_data = dict()
            for key in batch_data:
                per_data[key] = batch_data[key][i]
            per_data['prediction'] = pred[i]
        
            self.lock.acquire()
            with open(self.test_save_path, 'a+', encoding='utf-8') as wf:
                # print(per_data)
                wf.write(json.dumps(per_data, ensure_ascii=False) + '\n')
            self.lock.release()
            
    def save_data_structure(self, batch_data, html_pred, doc_pred):
        cnt = len(html_pred)
        for i in range(cnt):
            per_data = dict()
            for key in batch_data:
                per_data[key] = batch_data[key][i]
            per_data['html_prediction'] = html_pred[i]
            per_data['doc_prediction'] = doc_pred[i]
        
            self.lock.acquire()
            with open(self.test_save_path, 'a+', encoding='utf-8') as wf:
                wf.write(json.dumps(per_data, ensure_ascii=False) + '\n')
            self.lock.release()
            
    def save_data_counter(self, batch_data, pred, counter_pred):
        cnt = len(pred)
        for i in range(cnt):
            per_data = dict()
            for key in batch_data:
                per_data[key] = batch_data[key][i]
            per_data['prediction'] = pred[i]
            per_data['counter_prediction'] = counter_pred[i]
        
            self.lock.acquire()
            with open(self.test_save_path, 'a+', encoding='utf-8') as wf:
                wf.write(json.dumps(per_data, ensure_ascii=False) + '\n')
            self.lock.release()
            
        
    def test_basic(self, batch_data):
        input_str = batch_data['input_str']
        answers = batch_data['answers']
        
        pred = self.get_llm_output(input_str)
        self.save_data(batch_data, pred)
        
        metrics = defaultdict(lambda: 0.0)
        for i in range(len(input_str)):
            
            per_metric_map = self.evaluator.evaluation(pred[i], answers[i])
            for key in per_metric_map:
                metrics[key] += np.array(per_metric_map[key])
        
        return metrics, len(input_str)
    
    def test_structure(self, batch_data):
        answers = batch_data['answers']
        if self.opt.knowledge_type == 'CLOSE':
            input_str = batch_data['input_str']
            pred = self.get_llm_output(input_str)
            
            self.save_data(batch_data, pred)
            metrics = defaultdict(lambda: 0.0)
            for i in range(len(input_str)):
                per_metric_map = self.evaluator.evaluation(pred[i], answers[i])
                for key in per_metric_map:
                    metrics[key] += np.array(per_metric_map[key])

            return metrics, len(input_str)
    
        else:
            input_html_str = batch_data['input_html_str']
            input_doc_str = batch_data['input_doc_str']
            html_pred = self.get_llm_output(input_html_str)
            doc_pred = self.get_llm_output(input_doc_str)
            self.save_data_structure(batch_data, html_pred, doc_pred)
        
            metrics = defaultdict(lambda: 0.0)
            for i in range(len(input_html_str)):
                per_metric_map = self.evaluator.evaluation(html_pred[i], answers[i])
                for key in per_metric_map:
                    metrics['html_' + key] += np.array(per_metric_map[key])

            for i in range(len(input_doc_str)):
                per_metric_map = self.evaluator.evaluation(doc_pred[i], answers[i])
                for key in per_metric_map:
                    metrics['doc_' + key] += np.array(per_metric_map[key])
                    
            return metrics, len(input_html_str)
        
        
    def test_counterfact(self, batch_data):
        input_str = batch_data['input_str']
        counterfact_input_str = batch_data['counterfact_input_str']
        answers = batch_data['answers']
        counterfact_answers = batch_data['counterfact_answers']
        
        pred  = self.get_llm_output(input_str)
        counter_pred  = self.get_llm_output(counterfact_input_str)
        
        self.save_data_counter(batch_data, pred, counter_pred)
        
        metrics = defaultdict(lambda: 0.0)
        for i in range(len(input_str)):
            per_metric_map = self.evaluator.evaluation(pred[i], answers[i], counterfact_answers[i])
            for key in per_metric_map:
                metrics['true_bg_' + key] += np.array(per_metric_map[key])
        
        for i in range(len(counterfact_input_str)):
            per_metric_map = self.evaluator.evaluation(counter_pred[i], answers[i], counterfact_answers[i])
            for key in per_metric_map:
                metrics['counterfact_bg_' + key] += np.array(per_metric_map[key])
                
        return metrics, len(input_str)
        
