CORPUS_ROOT='../../corpus/rdzs'
DATA_ROOT='../labeled_data'
MODEL_ROOT='../../plms' #your root
RETRIEVER_TYPE='bm25'

python -u main_retrieval.py \
    --text_roots ${CORPUS_ROOT}/json_output \
    --retriever_type ${RETRIEVER_TYPE} \
    --query_encoder_path ${MODEL_ROOT}/${rank_name} \
    --context_encoder_path ${MODEL_ROOT}/${rank_name} \
    --retrieved_topk 20 \
    --task time \
    --retrieve_data_path ${DATA_ROOT}/time-sensitive_qa/time_sensitive.jsonl \
    --retrieve_save_path ${DATA_ROOT}/time-sensitive_qa/time_sensitive_retrieved_${RETRIEVER_TYPE}.jsonl 
    
RETRIEVER_TYPE='dense'
rank_name="bge-base-zh-v1.5"

python -u main_retrieval.py \
    --text_roots ${CORPUS_ROOT}/json_output \
    --retriever_type ${RETRIEVER_TYPE} \
    --query_encoder_path ${MODEL_ROOT}/${rank_name} \
    --context_encoder_path ${MODEL_ROOT}/${rank_name} \
    --retrieved_topk 20 \
    --task time \
    --query_instruct '为这个句子生成表示以用于检索相关文章：{}' \
    --retrieve_data_path ${DATA_ROOT}/time-sensitive_qa/time_sensitive.jsonl \
    --retrieve_save_path ${DATA_ROOT}/time-sensitive_qa/time_sensitive_retrieved_${RETRIEVER_TYPE}.jsonl 