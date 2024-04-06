from torch.utils.data import Dataset
import pickle
import linecache
import json
import numpy as np
from tqdm import tqdm

import copy

class BasicDataset(Dataset):
    def __init__(self, opt, path, tokenizer):

        self.max_input_len = opt.text_maxlength, 
        self.max_new_len = opt.generation_max_length
        self.pmt = opt.llm_pmt
        self.psg_pmt = opt.psg_pmt
        self.is_eval = opt.is_eval
        self.istest = opt.is_test
        
        self.tokenizer = tokenizer
        self.is_encoder_decoder=opt.is_encoder_decoder
        self.path = path

        
        self.tokenizer.pad_token = tokenizer.eos_token
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = 'left'
        else:
            self.tokenizer.padding_side = 'right'
            
            
        self.is_test = opt.is_test
        self.is_eval = opt.is_eval

        if not self.is_eval:
            self.total_len = len(open(self.path).readlines())
        else:
            self.total_len = 100
        
        
    
    def __len__(self):
        return self.total_len
    
    def parse_psgs(self, psgs):
        reference_str = ''
        for i, psg in enumerate(psgs):
            need_psgs = {
                "id": i+1,
                "title": psg['title'],
                "contents": psg['contents']
            }
            reference_str += self.psg_pmt.format_map(need_psgs)
            
        return reference_str

    def parse_data(self, line):
        line.replace("'", '"')
        data = json.loads(line, strict=False)
        query = data['question']
        positive_reference = data['positive_reference']
        
        if not isinstance(positive_reference, list):
            positive_reference = [positive_reference]
            
        reference_str = self.parse_psgs(positive_reference)
        
        input_data = {
            "docs": reference_str,
            "query": query
        }
        
        assert 'answers' in data or 'answer' in data
        
        answers = data['answers']
        
        return self.pmt.format_map(input_data), answers
    
    def __getitem__(self, idx):
        try:
            input_str, answers = self.parse_data(linecache.getline(self.path, idx+1))
            
        except:
            print('erro idx = ', idx+1)
            input_str, answers = self.parse_data(linecache.getline(self.path, idx+1))
        
        # inputs = self.tokenizer(input_str, max_length=self.max_input_len+1, padding='max_length')
        
        data = {
            'input_str': input_str,
            'answers': answers
        }
        return data

class ConversationDataset(BasicDataset):
    def __init__(self, opt, path, tokenizer):
        super(FactualDataset, self).__init__(opt, path, tokenizer)
        
    def __len__(self):
        return self.total_len
    
    def parse_data(self, line):
        data = json.loads(line)
        
    
        return None
    
    def __getitem__(self, idx):
        None
        
        

class FactualDataset(BasicDataset):
    def __init__(self, opt, path, tokenizer):
        super(FactualDataset, self).__init__(opt, path, tokenizer)
        
    def __len__(self):
        return self.total_len
    
    def parse_data(self, line):
        data = json.loads(line)
        query = data['question']
        positive_reference = data['positive_reference']
        counterfact_reference = data['counterfact_reference']
        
        if not isinstance(positive_reference, list):
            positive_reference = [positive_reference]
        if not isinstance(counterfact_reference, list):
            counterfact_reference = [counterfact_reference]
            
        reference_str = self.parse_psgs(positive_reference)
        counterfact_reference_str = self.parse_psgs(counterfact_reference)
        
        input_data = {
            "docs": reference_str,
            "query": query
        }
        
        counterfact_input_data = {
            "docs": counterfact_reference_str,
            "query": query
        }
        
        answers = data['answers']
        counterfact_answers = data['counterfact_answers']
        
    
        return self.pmt.format_map(input_data), self.pmt.format_map(counterfact_input_data), answers, counterfact_answers
    
    def __getitem__(self, idx):
        input_str, counterfact_input_str, answers, counterfact_answers = self.parse_data(linecache.getline(self.path, idx+1))
        
        data = {
            'input_str': input_str,
            'counterfact_input_str': counterfact_input_str,
            'answers': answers,
            'counterfact_answers': counterfact_answers
        }
        
        return data
        
        
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
