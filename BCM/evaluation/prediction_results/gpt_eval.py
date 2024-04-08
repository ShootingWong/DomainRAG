# from llm_models import gpt_generate, OpenAIApiProxy
import json
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
import collections
from collections import defaultdict
import re
from tqdm import tqdm
import numpy as np
import argparse

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

    # print(ans_json)
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
    print('multiprocess over')

    return final_list
    
    
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--eval_root", type=str, default="", help="Root of prediction results need to be evaluated. "
)
parser.add_argument(
    "--openai_api", type=str, default="", help=""
)
parser.add_argument(
    "--apikey", type=str, default="", help=""
)
parser.add_argument(
    "--model", type=str, default="gpt-4-1106-preview", help="Name of used GPT model"
)
parser.add_argument(
    "--pool_size", type=int, default=50, help="Pool size of multiprocessing"
)

opt = parser.parse_args()

pair_eval_prompt = '''
You is a good evaluator. I will give you a golden answer and a prediction, you should return a score to assess whether the prediction is accurately agrees with the golden answer. Make sure that: 
1. The score should be 5-scales, where 0 means totallly incorret and 5 means totally corret.
2. A good prediction is should satisfy the following aspects: a. Factually consistent with the golden answer. b. Should comprehensively cover the contents of the golden answer. c. Should contain no other information that not contained by the golden answer.
2. Only return a number without anyother words and punctuation.

Golden Answer: {}
Prediction: {}
Your score:
'''

def eval_batch(path):
    datas = [json.loads(line) for line in open(path).readlines()]
    results = []
    input_str_list = [pair_eval_prompt.format(data['answers'], data['prediction']) for data in datas]

        # ans = gpt_generate_str(proxy, model, input_str)

    ans = gpt_generate(input_str_list, pool_size, proxy, model)
    results = [int(a) for a in ans]
    return np.mean(results) / 5


if __name__ == '__main__':
    openai_api = opt.openai_api
    apikey = opt.apikey
    proxy = OpenAIApiProxy(openai_api, api_key=apikey)
    
    model = opt.model
    root = opt.eval_root


    pool_size = opt.pool_size
    
    files = os.listdir(root)
    paths = [os.path.join(root, f) for f in files if '.jsonl' in f]
    save_root = os.path.join(root, 'gpt_eval')

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    with open(os.path.join(save_root, 'ALL_RESULT.txt'), 'w') as wf:
        for i, path in tqdm(enumerate(paths)):
            res = eval_batch(path)

            wf.write('{}: {:.4}\n'.format(files[i], res))

