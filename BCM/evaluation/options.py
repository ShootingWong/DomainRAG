# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument(
            "--name", type=str, default="experiment_name", help="name of the experiment - also used as directory name "
        )
        self.parser.add_argument(
            "--llm_name", type=str, default="llama", help="name of the llm "
        )
        self.parser.add_argument(
            "--log_path", type=str, default="", help="output path"
        )
        
        self.parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="./checkpoint/",
            help="models are saved here",
        )
        self.parser.add_argument(
            "--model_path",
            type=str,
            default="",
            help="Path to a pretrained model to initialize from (pass 'none' to init from t5 and contriever)",
        )
        self.parser.add_argument(
            "--per_gpu_batch_size",
            default=1,
            type=int,
            help="Batch size per GPU/CPU for training.",
        )
        self.parser.add_argument(
            "--per_gpu_eval_batch_size",
            default=1,
            type=int,
            help="Batch size per GPU/CPU for training.",
        )
        self.parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
        
        self.parser.add_argument(
            "--log_freq",
            type=int,
            default=100,
            help="log train stats <log_freq> steps during training",
        )
        self.parser.add_argument(
            "--eval_freq",
            type=int,
            default=500,
            help="evaluate model every <eval_freq> steps during training",
        )
        self.parser.add_argument(
            "--save_freq",
            type=int,
            default=5000,
            help="save model every <save_freq> steps during training",
        )
        self.parser.add_argument(
            "--html_root", type=str, default="", help="path of document corpus that store the orignal html"
        )
        self.parser.add_argument(
            "--docs_root", type=str, default="", help="path of document corpus that store the pure text"
        )
        self.parser.add_argument(
            "--train_data", type=str, default="", help="path of space-separated paths to jsonl-formatted train sets"
        )
        self.parser.add_argument(
            "--eval_data",
            type=str, 
            default="",
            help="path of space-separated paths to jsonl-formatted evaluation sets",
        )
        
        self.parser.add_argument(
            "--test_data",
            type=str, 
            default="",
            help="path of space-separated paths to jsonl-formatted test sets",
        )
        self.parser.add_argument(
            "--test_save_path",
            type=str, 
            default="",
            help="path of space-separated paths to save predictions of jsonl-formatted test sets",
        )
        self.parser.add_argument(
            "--openai_api",
            type=str, 
            default="",
            help="path of space-separated paths to save predictions of jsonl-formatted test sets",
        )
        self.parser.add_argument(
            "--apikey",
            type=str, 
            default="",
            help="path of space-separated paths to save predictions of jsonl-formatted test sets",
        )
        
        
        
#         self.parser.add_argument(
#             "--train_data", nargs="+", default=[], help="list of space-separated paths to jsonl-formatted train sets"
#         )
#         self.parser.add_argument(
#             "--eval_data",
#             nargs="+",
#             default=[],
#             help="list of space-separated paths to jsonl-formatted evaluation sets",
#         )
        
#         self.parser.add_argument(
#             "--test_data",
#             nargs="+",
#             default=[],
#             help="list of space-separated paths to jsonl-formatted evaluation sets",
#         )
        
        self.parser.add_argument(
            "--is_encoder_decoder", action="store_true", help="Whether is encoder_decoder"
        )
        self.parser.add_argument(
            "--is_eval", action="store_true", help="Whether is encoder_decoder"
        )
        self.parser.add_argument(
            "--is_test", action="store_true", help="Whether is encoder_decoder"
        )
        
        

    def add_optim_options(self):
        
        self.parser.add_argument("--warmup_steps", type=int, default=1000, help="number of learning rate warmup steps")
        self.parser.add_argument("--K_epochs", type=int, default=3, help="total number of EPOCH when training on sampled pool")
        self.parser.add_argument("--max_episodes", type=int, default=3, help="total number of EPOCH")
        self.parser.add_argument("--total_steps", type=int, default=1000, help="total number of training steps")
        self.parser.add_argument(
            "--scheduler_steps",
            type=int,
            default=None,
            help="total number of step for the scheduler, if None then scheduler_total_step = total_step",
        )
        self.parser.add_argument("--accumulation_steps", type=int, default=1, help="gradient accumulation")
        self.parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--lr_retriever", type=float, default=1e-5, help="learning rate for retriever")
        self.parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
        self.parser.add_argument(
            "--scheduler",
            type=str,
            default="cosine",
            choices=["linear", "cosine", "fixed"],
            help="learning rate schedule to use",
        )
        self.parser.add_argument(
            "--weight_decay", type=float, default=0.1, help="amount of weight decay to apply in training"
        )
        self.parser.add_argument(
            "--save_optimizer", action="store_true", help="Pass flag to save optimizer state in saved checkpoints"
        )
        self.parser.add_argument("--shuffle", action="store_true", help="shuffle data for training")

        # memory optimizations:
        self.parser.add_argument(
            "--precision",
            type=str,
            default="fp32",
            choices=["fp16", "fp32", "bf16"],
            help="numerical precision - recommend bf16 if available, fp16 likely to be unstable for training",
        )
        
    def add_modeling_options(self):
        self.parser.add_argument(
            "--reader_model_path",
            # required=True,
            type=str,
            help="Decoder-only Architecture for reader RADIT model",
        )
        self.parser.add_argument(
            "--text_maxlength",
            type=int,
            default=200,
            help="maximum number of tokens in input text segments (concatenated question+passage). Inputs longer than this will be truncated.",
        )
        self.parser.add_argument(
            "--q_maxlength",
            type=int,
            default=200,
            help="maximum number of tokens in input text segments (concatenated question+passage). Inputs longer than this will be truncated.",
        )
        self.parser.add_argument(
            "--d_maxlength",
            type=int,
            default=200,
            help="maximum number of tokens in input text segments (concatenated question+passage). Inputs longer than this will be truncated.",
        )
        # self.parser.add_argument("--topk", type=int, default=1, help="number of top k passages to pass to reader")
        self.parser.add_argument("--retriever_n_context", type=int, default=50, help="number of top k passages to pass to reader")

        # Retriever modelling options
        self.parser.add_argument(
            "--passages_root",
            type=str,
            default="",
            help="list of paths to jsonl files containing passages to index and retrieve from. Unused if loading a saved index using --load_index_path",
        )
        self.parser.add_argument(
            "--passages",
            nargs="+",
            help="list of paths to jsonl files containing passages to index and retrieve from. Unused if loading a saved index using --load_index_path",
        )
        self.parser.add_argument(
            "--max_passages",
            type=int,
            default=-1,
            help="maximum number of passages to index. -1 to read all passages in passage files",
        )
        self.parser.add_argument(
            "--sample_times",
            type=int,
            default=1,
            help="maximum number of tokens in input text segments (concatenated question+passage). Inputs longer than this will be truncated.",
        )
        self.parser.add_argument(
            "--infer_batch",
            type=int,
            default=2,
            help="maximum number of tokens in input text segments (concatenated question+passage). Inputs longer than this will be truncated.",
        )
        self.parser.add_argument(
            "--rank_model_path",
            type=str,
            default="facebook/contriever",
            help="path to rerank model to init from (overridden if passing a value to --model_path ",
        )
        self.parser.add_argument(
            "--vllm",
            action="store_true",
            help='uses vllm to load LLM',
        )
        self.parser.add_argument(
            "--fastchat",
            action="store_true",
            help='uses vllm to load LLM',
        )
        self.parser.add_argument(
            "--gpt",
            action="store_true",
            help='uses vllm to load LLM',
        )
        self.parser.add_argument(
            "--baichuan",
            action="store_true",
            help='uses baichuan api to run LLM',
        )
        self.parser.add_argument(
            "--infer_type",
            type=str,
            default="fastchat",
            choices=["vllm", "fastchat", "gpt", "baichuan"],
            help='uses baichuan api to run LLM',
        )
        self.parser.add_argument(  # TODO: decide whether to remove functionality
            "--psg_pmt",
            type=str,
            default="Title:{title}. Content:{text}",
            help='format for decoder prompting, for instance "what is the answer to {query}:"',
        )
        self.parser.add_argument(  # TODO: decide whether to remove functionality
            "--llm_pmt",
            type=str,
            default="Background:{docs}. Question:{query}. Answer:",
            help='format for decoder prompting, for instance "what is the answer to {query}:"',
        )
        self.parser.add_argument(  # TODO: decide whether to remove functionality
            "--close_pmt",
            type=str,
            default="Background:{docs}. Question:{query}. Answer:",
            help='format for decoder prompting, for instance "what is the answer to {query}:"',
        )
        self.parser.add_argument(  # TODO: decide whether to remove functionality
            "--conv_pmt",
            type=str,
            default="**Given Information:**\n{docs}\n**Question:**\n{query}\n**Answer:**\n{answer}\n",
            help='format for decoder prompting, for instance "what is the answer to {query}:"',
        )
        self.parser.add_argument(  # TODO: decide whether to remove functionality
            "--query_instruct",
            type=str,
            default="",
            help='format for query instruction:"',
        )
        
        # Generation options
        self.parser.add_argument("--generation_max_length", type=int, default=128)
        self.parser.add_argument("--generation_min_length", type=int, default=None)
        self.parser.add_argument("--generation_length_penalty", type=float, default=1.0)
        self.parser.add_argument("--generation_num_beams", type=int, default=1)
        
        # Task-specific options:
        self.parser.add_argument(
            "--task",
            type=str,
            default=None,
            choices=["conversation", "counterfactual", "basic", "time", "structure", "multidoc", "noisy"],
            help="Task performed by the model. Used to setup preprocessing, retrieval filtering, evaluations, etc.",
        )
        
        self.parser.add_argument(
            "--knowledge_type",
            type=str,
            default=None,
            choices=["CLOSE", "GOLDEN", "BM25", "DENSE"],
            help="Task performed by the model. Used to setup preprocessing, retrieval filtering, evaluations, etc.",
        )
        # generation parameters
        self.parser.add_argument(
            "--gen_temperature",
            type=float,
            default=0,
            help="the higher value, the higher diversity. 0 is greedy search",
        )
        self.parser.add_argument(
            "--top_p",
            type=float,
            default=0.95,
            help="",
        )
        self.parser.add_argument(
            "--stop_words", nargs="+", default=['。'], help="list of words to stop generation of LLM"
        )
        
        
        
    def add_retrieval_options(self):
        self.parser.add_argument(
            "--html_roots",
            # type=str,
            nargs="+", 
            default=[''],
            help="Root of html contents of all documents",
        )
        self.parser.add_argument(
            "--text_roots",
            # type=str,
            nargs="+", 
            default=[''],
            help="Root of text contents of all documents",
        )
        self.parser.add_argument(
            "--retriever_type",
            type=str,
            default='bm25',
            help="Model type of retriever, bm25 or dense",
        )
        self.parser.add_argument(
            "--retrieved_topk",
            type=int,
            default=10,
            help="Count of candidate documents to retrieve.",
        )
        self.parser.add_argument(
            "--retrieve_data_path",
            type=str,
            default='',
            help="Datas that need to retrieve topk documents",
        )
        self.parser.add_argument(
            "--retrieve_save_path",
            type=str,
            default='',
            help="Root to save the retrieved data",
        )
        self.parser.add_argument(
            "--query_encoder_path",
            type=str,
            default='',
            help="Path of query encoder for dense retriever",
        )
        self.parser.add_argument(
            "--context_encoder_path",
            type=str,
            default='',
            help="Path of document encoder for dense retriever",
        )
        # self.parser.add_argument(
        #     "--golden_ref", action="store_true", help="Whether use golden reference to RAG"
        # )
        self.parser.add_argument(
            "--gpu_embedder_batch_size",
            type=int,
            default=512,
            help="",
        )
        self.parser.add_argument(
            "--retriever_format",
            type=str,
            default='{title}。{contents}',
            help="",
        )
        self.parser.add_argument(
            "--noisy_cnt",
            type=int,
            default=1,
            help="",
        )
        
        self.parser.add_argument(
            "--noisy_gold_idx",
            type=int,
            default=0,
            help="",
        )
        
        self.parser.add_argument(
            "--repeat",
            type=int,
            default=0,
            help="",
        )
        
    def parse(self):
        opt = self.parser.parse_args()
        
        return opt


def get_options():
    options = Options()
    options.add_modeling_options()
    options.add_optim_options()
    options.add_retrieval_options()
    
    return options
