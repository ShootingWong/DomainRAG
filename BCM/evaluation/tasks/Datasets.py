from torch.utils.data import Dataset
import pickle
import linecache
import json
import numpy as np
from tqdm import tqdm

import copy

def remove_space(s):
    return ''.join(s.split(' '))

class BasicDataset(Dataset):
    def __init__(self, opt, path):

        self.max_input_len = opt.text_maxlength, 
        self.max_new_len = opt.generation_max_length
        self.pmt = opt.llm_pmt
        self.close_pmt = opt.close_pmt
        self.psg_pmt = opt.psg_pmt
        self.is_eval = opt.is_eval
        self.istest = opt.is_test
        self.knowledge_type = opt.knowledge_type # == 'GOLDEN'
        self.topk = opt.retrieved_topk
        self.opt = opt
        
        # self.tokenizer = tokenizer
        self.is_encoder_decoder=opt.is_encoder_decoder
        self.path = path

        # if tokenizer is not None:
        #     self.tokenizer.pad_token = tokenizer.eos_token
        # if not self.is_encoder_decoder:
        #     self.tokenizer.padding_side = 'left'
        # else:
        #     self.tokenizer.padding_side = 'right'
            
            
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
                "title": remove_space(psg['title']),
                "contents": remove_space(psg['contents'])
            }
            reference_str += self.psg_pmt.format_map(need_psgs)
            
        return reference_str
    
    def get_reference(self, data):
        if self.knowledge_type == "GOLDEN":
            reference = data['positive_reference']
        else:
            reference = data['retrieved_psgs'][:self.topk]
        
        # print(f"positive_reference = {data['positive_reference']}\n")
        # print(f"retrieved_psgs = {data['retrieved_psgs'][:self.topk]}\n")
            
        return reference

    def parse_data(self, line):
        line.replace("'", '"')
        data = json.loads(line, strict=False)
        query = data['question']
        
        if self.opt.task == 'time':
            # print('Original query = ', query)
            date = data['date']
            query = date + '年 '+ query
            # print('Time query = ', query)
        
        if self.knowledge_type != "CLOSE":
            reference = self.get_reference(data)

            if not isinstance(reference, list):
                reference = [reference]

            reference_str = self.parse_psgs(reference)

            input_data = {
                "docs": reference_str,
                "query": query
            }

            prompt = self.pmt
        else:
            input_data = {
                "query": query
            }
            prompt = self.close_pmt
            
        assert 'answers' in data or 'answer' in data
        ans_key = 'answers' if 'answers' in data else 'answer'
        answers = data[ans_key]

        return prompt.format_map(input_data), answers
            
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

class NoisyDataset(BasicDataset):
    def __init__(self, opt, path):
        super(NoisyDataset, self).__init__(opt, path)
        self.noisy_cnt = opt.noisy_cnt
    
    def __len__(self):
        return self.total_len
    
    def get_reference(self, data):
        pos_reference = data['positive_reference']
        noi_reference = data['noisy_reference'][:self.noisy_cnt]
        # mid_idx = len(noi_reference) // 2
        idx = self.opt.noisy_gold_idx
        
        if not isinstance(pos_reference, list):
            pos_reference = [pos_reference]
        if self.opt.repeat == 1:
            reference = pos_reference * (self.noisy_cnt + 1)
        else:
            reference = noi_reference[:idx] + pos_reference + noi_reference[idx:]
        
        return reference
    
class MultidocDataset(BasicDataset):
    def __init__(self, opt, path):
        super(MultidocDataset, self).__init__(opt, path)
        # self.conv_pmt = opt.conv_pmt
        
    def __len__(self):
        return self.total_len
    
    def get_reference(self, data):
        # print(f'data keys = {data.keys()}')
        if self.knowledge_type == "GOLDEN":
            reference = data['positive_references'] #data['referred_docs']
        else:
            reference = data['retrieved_psgs'][:self.topk]
            
        return reference
    
    
class ConversationDataset(BasicDataset):
    def __init__(self, opt, path):
        super(ConversationDataset, self).__init__(opt, path)
        self.conv_pmt = opt.conv_pmt
        
    def __len__(self):
        return self.total_len
    
    def get_reference(self, data):
        if self.knowledge_type == "GOLDEN":
            reference = data['positive_reference']
        else:
            reference = data['retrieved_psgs'][:self.topk]
            
        return reference
    
    def parse_data(self, line):
        data = json.loads(line)
        history_qa = data['history_qa']
        query = data['question']
        # positive_reference = data['positive_reference']
        
        
        answers = data['answers']

        

        history = ""
        for qa in history_qa:
            his_reference_str = self.parse_psgs([qa['positive_reference']])
            qa_data = {
                "docs": his_reference_str,
                "query": qa["question"],
                "answer": qa["answers"][0],
            }
            history += self.conv_pmt.format_map(qa_data)

        if self.knowledge_type != "CLOSE":
            reference = self.get_reference(data)
            if not isinstance(reference, list):
                reference = [reference]
            reference_str = self.parse_psgs(reference)

            input_data = {
                "history": history,
                "docs": reference_str,
                "query": query
            }
            prompt = self.pmt
        else:
            input_data = {
                "history": history,
                "query": query
            }
            prompt = self.close_pmt
    
        return prompt.format_map(input_data), answers
    
    def __getitem__(self, idx):
        
        input_str, answers  = self.parse_data(linecache.getline(self.path, idx+1))
        
        data = {
            'input_str': input_str,
            'answers': answers,
        }
        
        return data
        

class FactualDataset(BasicDataset):
    def __init__(self, opt, path):
        super(FactualDataset, self).__init__(opt, path)
        
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
        
        if self.knowledge_type != "CLOSE":
            
            input_data = {
                "docs": reference_str,
                "query": query
            }

            counterfact_input_data = {
                "docs": counterfact_reference_str,
                "query": query
            }
            prompt = self.pmt
        else:
            input_data = {
                # "docs": reference_str,
                "query": query
            }

            counterfact_input_data = {
                # "docs": counterfact_reference_str,
                "query": query
            }
            prompt = self.close_pmt

        answers = data['answers']
        counterfact_answers = data['counterfact_answers']
        
        return prompt.format_map(input_data), prompt.format_map(counterfact_input_data), answers, counterfact_answers
    
    def __getitem__(self, idx):
        input_str, counterfact_input_str, answers, counterfact_answers = self.parse_data(linecache.getline(self.path, idx+1))
        
        data = {
            'input_str': input_str,
            'counterfact_input_str': counterfact_input_str,
            'answers': answers,
            'counterfact_answers': counterfact_answers
        }
        
        return data
        

class StructureDataset(BasicDataset):
    def __init__(self, opt, path):
        # print(f'Initialize StructureDataset, path = {path}')
        super(StructureDataset, self).__init__(opt, path)
        
    def __len__(self):
        return self.total_len
    
    def parse_html(self, psgs):
        reference_str = ''
        for i, psg in enumerate(psgs):
            reference_str += psg+'\n' #self.psg_pmt.format_map(need_psgs)
            
        return reference_str
    
    def parse_docs(self, psgs):
        reference_str = ''
        # print(f'PASSAGES = {psgs}')
        for i, psg in enumerate(psgs):
            # print(f'psg = {psg}')
            need_psgs = {
                "id": i+1,
                "title": psg['title'],
                "contents": psg['contents']
            }
            reference_str += self.psg_pmt.format_map(need_psgs)
            
        return reference_str
    
    def parse_data(self, line):
        data = json.loads(line)
        query = data['question']
        answers = data['answers']
        
        if self.knowledge_type != "CLOSE":
            positive_html = data['positive_html']
            positive_reference = data['positive_reference']
        
            if not isinstance(positive_html, list):
                positive_html = [positive_html]
            if not isinstance(positive_reference, list):
                positive_reference = [positive_reference]
            # print(f'positive_reference = {positive_reference}')
            reference_html_str = self.parse_html(positive_html)
            reference_str = self.parse_docs(positive_reference)
            # counterfact_reference_str = self.parse_psgs(counterfact_reference)
        
            input_data_html = {
                "docs": reference_html_str,
                "query": query
            }
            
            input_data_doc = {
                "docs": reference_str,
                "query": query
            }
            prompt = self.pmt
            
            return prompt.format_map(input_data_html), prompt.format_map(input_data_doc), answers
        
        else:
            input_data = {
                "query": query
            }
            prompt = self.close_pmt
            
            return prompt.format_map(input_data), answers
        
    
        
    
    def __getitem__(self, idx):
        if self.knowledge_type == "CLOSE":
            input_str, answers = self.parse_data(linecache.getline(self.path, idx+1))
        
            data = {
                'input_str': input_str,
                'answers': answers,
            }
        else:
            input_html_str, input_doc_str, answers = self.parse_data(linecache.getline(self.path, idx+1))
        
            data = {
                'input_html_str': input_html_str,
                'input_doc_str': input_doc_str,
                'answers': answers,
            }
        return data
        
        
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])
