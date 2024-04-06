openai_api=''
apikey=''
run() {
    LLM_NAME=$1
    LLM_ROOT=$2
    READER_PATH=${LLM_ROOT}/${LLM_NAME}
    TOPK=1 
    INFER=$3
    TASK=$4 
    KNOWLEDGE='GOLDEN' 

    MAX_NEW_TOKEN=$5 
    INFER_BATCH=$6 
    LOG_PATH=$7
    noisy_cnt=$8
    noisy_gold_idx=$9
    repeat=0
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
        --log_path ${LOG_PATH} \
        --noisy_cnt ${noisy_cnt} \
        --noisy_gold_idx ${noisy_gold_idx} \
        --repeat ${repeat}
        
}

llm_root='../../plms' #your root

for llm in "llama-2-7b-chat-hf" "llama-2-13b-chat-hf" "llama-2-70b-chat-hf" "Baichuan2-7B-Chat" "chatglm2-6b-32k" "gpt-3.5-turbo-1106"
do
    if [ $llm = "gpt-3.5-turbo-1106" ]; then
        infer="gpt"
    else
        infer="fastchat"
    fi
    
    if [ ${llm} =  "chatglm2-6b-32k" ]; then
        bsz=4
    else
        bsz=32
    fi

    task="noisy"
    gold_idx=0
    for noise_cnt in 4 9 14 19 24
    do
        run $llm $llm_root $infer $task 500 $bsz "logs/${task}.${llm}.noise_cnt${noise_cnt}-gold_idx${gold_idx}.log" ${noise_cnt} ${gold_idx}
    done
done

