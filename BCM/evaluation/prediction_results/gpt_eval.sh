eval_root=$1
openai_api=$2 
apikey=$3 
model=$4 
python gpt_eval.py \
    --eval_root ${eval_root}\
    --openai_api ${openai_api} \
    --apikey ${apikey} \
    --model ${model}