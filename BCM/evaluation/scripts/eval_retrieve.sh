openai_api=''
apikey=''
run() {
    LLM_NAME=$1 
    LLM_ROOT=$2 
    READER_PATH=${LLM_ROOT}/${LLM_NAME}
    TOPK=$3 
    INFER=$4
    TASK=$5 
    KNOWLEDGE=$6 

    MAX_NEW_TOKEN=$7 
    INFER_BATCH=$8 
    LOG_PATH=$9
    TEM=0

    
    python main.py \
        --is_test \
        --infer_type ${INFER} \
        --task ${TASK} \
        --retrieved_topk ${TOPK} \
        --knowledge_type ${KNOWLEDGE} \
        --per_gpu_eval_batch_size ${INFER_BATCH} \
        --infer_batch ${INFER_BATCH} \
        --reader_model_path ${READER_PATH} \
        --gen_temperature ${TEM} \
        --generation_max_length ${MAX_NEW_TOKEN} \
        --openai_api ${openai_api} \
        --apikey ${apikey} \
        --llm_name ${LLM_NAME} \
        --log_path ${LOG_PATH}
        
}

llm_root='../../plms' #your root

for llm in "llama-2-7b-chat-hf" "llama-2-13b-chat-hf" "Baichuan2-7B-Chat" "chatglm2-6b-32k" "llama-2-70b-chat-hf" "gpt-3.5-turbo-1106"
do
    if [ $llm = "gpt-3.5-turbo-1106" ]; then
        infer="gpt"
    else
        infer="fastchat"
    fi

    if [ $llm = "llama-2-70b-chat-hf" ]; then
        bsz=2
    else
        bsz=32
    fi

    for task in "basic" "time" 
    do
        for topk in 1 3 
        do
            for ret in "basic" "time" "structure" "faithful" "multidoc" "conversation" 
            do
                run $llm $llm_root $topk $infer $task $ret 500 $bsz "logs/${ret}.${topk}.${task}.${llm}.log"
            done
        done
    done
    
done
