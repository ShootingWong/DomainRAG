openai_api=''
apikey=''
run() {

    LLM_NAME=$1
    LLM_ROOT=$2
    READER_PATH=${LLM_ROOT}/${LLM_NAME}
    INFER=$3
    TASK=$4
    KNOWLEDGE=$5

    MAX_NEW_TOKEN=$6
    INFER_BATCH=$7
    PER_GPU_EVAL_BATCH=${INFER_BATCH}
    LOG_PATH=$8
    TEM=0

    python main.py \
        --is_test \
        --infer_type ${INFER} \
        --task ${TASK} \
        --knowledge_type ${KNOWLEDGE} \
        --per_gpu_eval_batch_size ${PER_GPU_EVAL_BATCH} \
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

for llm in "llama-2-7b-chat-hf" "llama-2-13b-chat-hf" "llama-2-70b-chat-hf" "Baichuan2-7B-Chat" "chatglm2-6b-32k" "gpt-3.5-turbo-1106"
do
    
    if [ $llm = "gpt-3.5-turbo-1106" ]; then
        infer="gpt"
    else
        infer="fastchat"
    fi
    
    if [ $llm = "llama-2-70b-chat-hf" ]; then
        bsz=4
    else
        bsz=32
    fi
    
    for task in "basic" "time" "structure" "faithful" "multidoc" "conversation"
    do
        run ${llm} ${llm_root} ${infer} ${task} "GOLDEN" 500 ${bsz} "logs/GOLDEN.${task}.${llm}.log"
    done
done
