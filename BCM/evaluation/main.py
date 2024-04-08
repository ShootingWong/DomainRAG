# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoTokenizer, AutoModel, AutoTokenizer, AutoModelForCausalLM
from torch.distributions import MultivariateNormal
# import gym
from collections import defaultdict
import numpy as np
import math
import copy 
import time
from tqdm import tqdm
from copy import deepcopy
import os
from vllm import LLM, SamplingParams
from fastchat.model import load_model

from options import get_options
# from src.util import cast_to_precision
# from src.fid_ranker import FidRanker
# from safetensors import safe_open
import sys
sys.path.append('..')
from models.llm_models import MyLLM
import json

from tasks import get_dataset
from options import get_options
from Evaluator import ExtractiveEvaluator, FactualEvaluator, ConversationEvaluator

# "conversation", "counterfactual", "basic", "time", "structure", "multidoc"
evaluator_map = {
    "basic": ExtractiveEvaluator,
    "time": ExtractiveEvaluator,
    "structure": ExtractiveEvaluator,
    "multidoc": MultidocEvaluator,
    "conversation": ConversationEvaluator,
    "counterfactual": FactualEvaluator,
    "noisy": ExtractiveEvaluator,
}
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def test(opt):
    print(f'In test')
    num_gpus=torch.cuda.device_count()
    if 'chatglm' in opt.llm_name:
        num_gpus=1
    if opt.fastchat:
        llm_model,llm_tokenizer = load_model(opt.reader_model_path,
                                        device='cuda', 
                                        num_gpus=num_gpus,
                                        max_gpu_memory='120GiB',
                                        load_8bit=False,
                                        cpu_offloading=False,
                                        debug=False,
                                        # trust_remote_code=True
                                        )
        print(f'llm_model.device = ', llm_model.device)
        sampling_params=None
    elif opt.vllm:
        # llm_model = LLM(
        #     model=opt.reader_model_path,
        #     tensor_parallel_size=num_gpus,
        #     gpu_memory_utilization=0.7,
        #     trust_remote_code=True
        # )
        
        llm_model = LLM(
            model=opt.reader_model_path,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.5,
            trust_remote_code=True
        )
        
        sampling_params = SamplingParams(
            temperature=opt.gen_temperature, 
            # max_tokens=opt.generation_max_length
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(opt.reader_model_path, trust_remote_code=True)
    elif opt.gpt:
        print('We use GPT Models')
        llm_model = None #GPTModel(opt)
        llm_tokenizer = None
        sampling_params=None
    elif opt.baichuan:
        # print('We use BAICHUAN Models')
        sampling_params = {
            # "repetition_penalty":1.05,
            "temperature":0.01,
            # "top_k":5,
            # "top_p":0.85,
            "max_new_tokens":opt.generation_max_length,
            # "do_sample":True, 
            # "seed": 3
        }
        llm_model = None #GPTModel(opt)
        llm_tokenizer = None
        
    evaluator = evaluator_map[opt.task]()
    
        # opt, model, tokenizer, evaluator, sample_params
    llm = MyLLM(opt, llm_model, llm_tokenizer, evaluator, sample_params=sampling_params)
    
    if opt.is_test:
        test_batch_size = opt.per_gpu_eval_batch_size * num_gpus
        test_dataset = get_dataset(opt, opt.test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collator, shuffle=False, num_workers=2)

        metrics = defaultdict(lambda: 0.0)
        all_cnt = 0
        for step, batch in tqdm(enumerate(test_dataloader)):
            batch_metric, per_cnt = llm.test_batch(batch)
            for key in batch_metric:
                metrics[key] += batch_metric[key]
            all_cnt += per_cnt
        
        print(f'------------For Task {opt.task}------------')
        for key in metrics:
            print(f'AVG {key}: {metrics[key] / all_cnt}')
        
def load_opt(opt):
    configs = json.load(open('config/config.json'))
    task = opt.task
    knowledge_type = opt.knowledge_type
    topk = opt.retrieved_topk
    
    llm_name = opt.llm_name
    data_root = configs['data_root']
    save_root = configs['save_root']
    save_name = configs['save_name'][knowledge_type].format(llm_name, topk)
    
    
    task_config = configs["tasks"][task]
    task_data_root = task_config['data_root']
    task_save_root = task_config['save_root']
    task_data_name = task_config['data_path']
    if knowledge_type in ["BM25", "DENSE"]:
        if knowledge_type == 'BM25':
            kt = knowledge_type.lower()
        else:
            kt = 'dense_instruct'
        data_suffix = "_retrieved_{}".format(kt)
    else:
        data_suffix = ""
    task_data_name = task_data_name.format(data_suffix)
    
    if task == "noisy":
        save_name = task_config["save_name"].format(llm_name, opt.noisy_cnt, opt.noisy_gold_idx)
    
    data_path = os.path.join(data_root, task_data_root, task_data_name)
    
    save_root1 = os.path.join(save_root, task_save_root)
    if not os.path.exists(save_root1):
        os.mkdir(save_root1)
        
    save_path = os.path.join(save_root1, save_name)
    
    if opt.repeat == 1:
        save_path += '.repeat'
        
    opt.test_data = data_path
    opt.test_save_path = save_path
    
    print(f'opt.test_data = {opt.test_data}')
    print(f'opt.test_save_path = {opt.test_save_path}')
    
    opt.psg_pmt = configs["psg_pmt"]
    opt.llm_pmt = task_config["llm_pmt"]
    opt.close_pmt = task_config["close_pmt"]
    if task == "conversation":
        opt.conv_pmt = task_config["conv_pmt"]
    
    ["vllm", "fastchat", "gpt", "baichuan"]
    
    if opt.infer_type == "vllm":
        opt.vllm = True
    elif opt.infer_type == "fastchat":
        opt.fastchat = True
    elif opt.infer_type == "gpt":
        opt.gpt = True
    elif opt.infer_type == "baichuan":
        opt.baichuan = True
    else:
        print('Invalid infer type! exit')
        import sys
        sys.exit(0)
        
    if opt.task in ['structure', 'multidoc', 'conversation']:
        opt.per_gpu_eval_batch_size = min(opt.per_gpu_eval_batch_size, 16)
        opt.infer_batch = min(opt.infer_batch, 16)
        
        if 'chatglm' in opt.llm_name:
            opt.per_gpu_eval_batch_size = 1
            opt.infer_batch = 1
    
    return opt

    
if __name__ == "__main__":
    
    options = get_options()
    opt = options.parse()
    opt = load_opt(opt)
    print('OPT!!!!!!', opt)

 
    f = open(opt.log_path, 'w', encoding='utf8')
    original_stdout = sys.stdout
    sys.stdout = f

    test(opt)

    sys.stdout = original_stdout
    f.close()


