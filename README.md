# DomainRAG
DomainRAG: A Chinese Benchmark for Evaluating Domain-specifc Retrieval-Augmented Generation

## Preliminary
The document corpurs is provided here: https://drive.google.com/file/d/1NquEyPGwP0MpTGJwDUUYKU37snYN4Er4/view?usp=sharing

After download your should put the corpus.tar.gz in the #{your-root}/DomainRAG/ and decompressing as following,
```
mv corpus.tar.gz #{your-root}/DomainRAG/
cd #{your-root}/DomainRAG/
tar -zxvf corpus.tar.gz 
```
Then there should be a directory, corpus, which is in a parallel folder with BCM.

## Requirements
The requirements of DomainRAG is wrapped in the py39.yaml and py310.yaml file, install them by :

```
conda env create -f py39.yaml
conda env create -f py310.yaml
```
Note that the py39 is used for runing retrieval process. It involves faiss-gpu package, which is hard to download an adapted version... The py310 is used for evaluation. You can only build the environment of py310 if you do not need to do retrieve process.

## Run Retrieval
we provide python files related to retrieve documents from corpus via basic retrievers in #{your-root}/DomainRAG/BCM/models. You can do this process by the following commands:

```
cd #{your-root}/DomainRAG/BCM/models
sh retrieve.sh # it will direct retrieve documents for extractive/conversational/multi-doc/time-sensitive datasets.
```

## Run Evaluation

The run files for evaluating models is put in #{your-root}/DomainRAG/BCM/evaluation. In each eval_${xx}.sh file, you should provide values for parameters of "openai_api" and "api_key" if you want to evaluate GPT models. You should also privide the value for "llm_root", which is name of the directory where you store LLMs' checkpoints. The default is '${your_root}/plms'.

You can run the following commands for generation and evaluation:
```
cd #{your-root}/DomainRAG/BCM/evaluation/
sh scripts/eval_close.sh # evaluate all models in the close-book setting
sh scripts/eval_golden.sh # evaluate all models in the golden-references setting
sh scripts/eval_retrieve.sh # evaluate all models in the retrieved-references setting
sh scripts/eval_noisy.sh # evaluate all models in different noisy settings.
```

The gpt-4 evaluation is conducted after all predicted results are generated (implemented by the above commands). You can run the following commands:

```
cd #{your-root}/DomainRAG/BCM/evaluation/prediction_results
sh gpt_eval.sh #{eval_root} #{openai_api} #{apikey} #{model}
```

where #{eval_root} is the directory that save all the predicted results for a certain task, such as 'conversation_qa'. #{model} is the name of GPT model used to evaluate, where we use 'gpt-4-1106-preview'.




