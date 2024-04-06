from llm_models import MyLLM
from Retriever import Retriever
import sys
sys.path.append('..')
from evaluation.options import get_options
import torch
import json
import os

def load_jsonl(path):
    print(f'-------------Begin to Load data from {path}-------------')
    datas = [json.loads(line) for line in open(path).readlines()]

    return datas

if __name__ == "__main__":
    
    options = get_options()
    opt = options.parse()
    print(opt)
    device = torch.device("cuda:0")
    # opt.print_options(opt)
    retriever = Retriever(opt, device)
    
    datas = load_jsonl(opt.retrieve_data_path)
    
    retriever.retrieve_and_save(datas, opt.retrieve_save_path)


