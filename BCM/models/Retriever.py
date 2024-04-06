import torch
import torch.nn as nn
import os
from tqdm import tqdm
import json
import copy
from dense_retriever import DualRetriever, FaissIndex
from rank_bm25 import BM25Okapi
import jieba
import numpy as np

class Retriever:
    
    def __init__(self, opt, device):
        
        self.opt = opt
        self.html_roots = opt.html_roots
        self.text_roots = opt.text_roots
        self.model_type = opt.retriever_type
        
        self.doc_corpus = dict()
        self.load_text_corpus(self.text_roots)
        
        if self.model_type == 'dense':
            self.retriever = DualRetriever(opt.query_encoder_path, opt.context_encoder_path, opt, device)
            self.retriever.build_index(self.all_psgs, opt.gpu_embedder_batch_size)
            
        elif self.model_type == 'bm25':
            documents = [['{} {}'.format(item['title'], psg).split(' ') for i, psg in enumerate(item['passages'])] for item in list(self.doc_corpus.values())]
            documents = sum(documents, [])
            self.retriever = BM25Okapi(documents)
        
    def load_text_corpus(self, text_roots):
        print('---------------Begin to Load TEXT Corpus---------------')
        self.all_doc_psg_ids = []
        self.all_psgs = []
        for text_root in text_roots:
            names = os.listdir(text_root)

            for name in tqdm(names):
                if 'json' not in name:continue
                path = os.path.join(text_root, name)
                data = json.loads(open(path).read())
                did = data['id']
                url = data['url'].strip()
                did = '\0'.join([str(did), url])
                
                self.doc_corpus[did] = data
  
                for i, psg in enumerate(data['passages']):
                    self.all_doc_psg_ids.append('\t'.join([str(did), str(i)]))
                    self.all_psgs.append({
                        'title': data['title'],
                        'contents': ''.join(psg.split(' '))
                    })
                    
        print('---------------Load TEXT Corpus Over---------------')
        
        
    def from_dpid_to_psgdict(self, dpid):
        did, pid = dpid.split('\t')
        did = did
        pid = int(pid)
        
        psg_dict = copy.deepcopy(self.doc_corpus[did])
        
        psg_dict['contents'] = psg_dict['passages'][pid]
        
        del psg_dict['passages']
        
        return psg_dict
    
        
    def from_dpid_to_docdict(self, dpid):
        did, pid = dpid.split('\t')
        did = did
        # pid = int(pid)
        doc_dict = copy.deepcopy(self.doc_corpus[did])
        
        del doc_dict['passages']
        
        return doc_dict
    
    def save_retrieve_result(self, datas, all_sel_psgs, save_path):
        print('-------------Begin to save retrieved results-------------')
        with open(save_path, 'w') as wf:
            for i, data in enumerate(datas):
                new_data = copy.deepcopy(data)
                new_data['retrieved_psgs'] = all_sel_psgs[i]
                wf.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        print('-------------Save retrieved results over-------------')
        
    def retrieve_and_save(self, datas, save_path):
        if self.opt.task == 'time':
            all_querys = [data['date'] + '年 ' + data['question'] for data in datas]
        elif self.opt.task == 'conversation':
            all_querys = []
            for i, data in enumerate(datas):
                q = data['question']
                hisq = [qa['question'] for qa in data['history_qa']]
                cat_q = ' '.join(hisq + [q])
                all_querys.append(cat_q)
        else:
            all_querys = [data['question'] for data in datas]
        
            
        all_doc_psg_ids_np = np.array(self.all_doc_psg_ids)
        all_sel_psgs = []
        
        def get_sel_psgs(index_list):
            sel_dpids = all_doc_psg_ids_np[index_list]
            sel_psgs = [self.from_dpid_to_psgdict(dpid) for dpid in sel_dpids]
            
            return sel_psgs
        
        if self.model_type == 'dense':
            D, I = self.retriever.retrieve(self.opt.retrieved_topk, all_querys)
            print(f'D shape = {np.array(D).shape} I shape = {np.array(I).shape}')
            
            for j, query in enumerate(all_querys):
                index_list = np.array(list(I[j]), dtype=np.int32)
                dist_list = np.array(list(D[j]))
                
                sel_psgs = get_sel_psgs(index_list)
                all_sel_psgs.append(sel_psgs)
                
        elif self.model_type == 'bm25':
            
            for query in all_querys:
                tokenized_query = list(jieba.cut(query))
                scores = self.retriever.get_scores(tokenized_query)
                
                index_list = np.argsort(-scores)[:self.opt.retrieved_topk]
                dist_list = scores[index_list]
                
                sel_psgs = get_sel_psgs(index_list)
                all_sel_psgs.append(sel_psgs)
                
        
        self.save_retrieve_result(datas, all_sel_psgs, save_path)
                
                

            
            
        