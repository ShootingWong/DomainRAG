from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import math
from tqdm import tqdm
import faiss

def _to_cuda(tok_dict):
        return {k: v.cuda() for k, v in tok_dict.items()}
    
class BaseRetriever(torch.nn.Module):
    """A retriever needs to be able to embed queries and passages, and have a forward function"""

    def __init__(self, *args, **kwargs):
        super(BaseRetriever, self).__init__()

    def embed_queries(self, *args, **kwargs):
        raise NotImplementedError()

    def embed_passages(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, is_passages=False, **kwargs):
        if is_passages:
            return self.embed_passages(*args, **kwargs)
        else:
            return self.embed_queries(*args, **kwargs)

    def gradient_checkpointing_enable(self):
        for m in self.children():
            m.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        for m in self.children():
            m.gradient_checkpointing_disable()

            
class DualRetriever(BaseRetriever):
    
    def __init__(self, query_encoder_path, context_encoder_path, opt, device):
        super(DualRetriever, self).__init__()

        # self.query_encoder = query_encoder #AutoModel.from_pretrained(query_encoder_path)
        # self.context_encoder = context_encoder #AutoModel.from_pretrained(context_encoder_path)
        
        self.query_encoder = AutoModel.from_pretrained(query_encoder_path).to(device)
        self.context_encoder = AutoModel.from_pretrained(context_encoder_path).to(device)
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(context_encoder_path)

        self.opt = opt
        self.device = device
        
        self.query_encoder.eval()
        self.context_encoder.eval()
    
    @torch.no_grad()
    def embed_queries(self, input_ids, attention_mask, token_type_ids=None):
        embs = self.query_encoder(input_ids, attention_mask)[0][:, 0] 
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs
    @torch.no_grad()
    def embed_passages(self, input_ids, attention_mask, token_type_ids=None):
        embs = self.context_encoder(input_ids, attention_mask)[0][:, 0]
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        return embs
    
    def instruction(self, querys):
        if self.opt.query_instruct != '':
            new_querys = [self.opt.query_instruct.format(q) for q in querys]
            return new_querys
        else:
            return querys
    
    def retrieve(self, topk, querys):
        
        query_input = self.retriever_tokenizer(
            querys,
            padding="longest",
            return_tensors="pt",
            # max_length=self.opt.q_maxlength,
            truncation=True,
        )
        
        query_vec = self(**_to_cuda(query_input), is_passages=False).detach().cpu().numpy()

        D, I = self.index.search(query_vec, self.opt.retrieved_topk)

        return D, I#passages, scores, query_emb

    @torch.no_grad()
    def build_index(self, passages, gpu_embedder_batch_size, logger=None):
        print('------------Building Index Ing------------')
        
        index_embeddings = torch.zeros(len(passages), 768)

        n_batch = math.ceil(len(passages) / gpu_embedder_batch_size)
        pbar = tqdm(total=n_batch)
        # retrieverfp16 = self._get_fp16_retriever_copy()
        self._get_fp16_retriever_copy()

        # retrieverfp16
        total = 0
        for i in range(n_batch):
            batch = passages[i * gpu_embedder_batch_size : (i + 1) * gpu_embedder_batch_size]
            # print(f'batch[0] = {batch[0]} self.opt.retriever_format = {self.opt.retriever_format}')
            batch = [self.opt.retriever_format.format_map(example) for example in batch]
            batch_enc = self.retriever_tokenizer(
                batch,
                padding="longest",
                return_tensors="pt",
                # max_length=min(self.opt.d_maxlength, gpu_embedder_batch_size),
                truncation=True,
            )

            embeddings = self(**_to_cuda(batch_enc), is_passages=True)
            print(f'embeddings size = {embeddings.size()}')
            index_embeddings[total : total + len(embeddings), :] = embeddings
            total += len(embeddings)
            # if i % 500 == 0 and i > 0:
            #     logger.info(f"Number of passages encoded: {total}")
            pbar.update(1)
        # dist_utils.barrier()

        print(f"{total} passages encoded index_embeddings size = {index_embeddings.size()} ")

        self.index = FaissIndex(self.device)
        self.index.build(index_embeddings, "Flat", "ip")
        
        # return index

    def _get_fp16_retriever_copy(self):
        # if hasattr(self.retriever, "module"):
        #     retriever_to_copy = self.retriever.module
        #     self.query_encoder = query_encoder
        # else:
        #     retriever_to_copy = self.retriever
        self.query_encoder = self.query_encoder.half().eval()
        self.context_encoder = self.context_encoder.half().eval()

        
class FaissIndex:
    def __init__(self, device) -> None:
        if isinstance(device, torch.device):
            if device.index is None:
                device = "cpu"
            else:
                device = device.index
        self.device = device

    def build(self, encoded_corpus, index_factory, metric):
        if metric == "l2":
            metric = faiss.METRIC_L2
        elif metric in ["ip", "cos"]:
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise NotImplementedError(f"Metric {metric} not implemented!")
        
        index = faiss.index_factory(encoded_corpus.shape[1], index_factory, metric)
        
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            # logger.info("using fp16 on GPU...")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)

        print("training index...")
        index.train(encoded_corpus)
        print("adding embeddings...")
        index.add(encoded_corpus)
        self.index = index
        return index

    def load(self, index_path):
        print(f"loading index from {index_path}...")
        index = faiss.read_index(index_path)
        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), self.device, index, co)
        self.index = index
        return index
    
    def save(self, index_path):
        print(f"saving index at {index_path}...")
        if isinstance(self.index, faiss.GpuIndex):
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index
        faiss.write_index(index, index_path)

    def search(self, query, hits):
        return self.index.search(query, k=hits)
    
