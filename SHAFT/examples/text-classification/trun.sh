#/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
eval "$(conda shell.bash hook)"
conda activate ADSMPC
python ./run_glue_private.py  --model_name_or_path andeskyl/bert-base-cased-qnli --task_name qnli --len_data 128 --num_data 1 --max_length 128 --per_device_eval_batch_size 1 --output_dir eval_private/qnli/  2>&1 | tee ./output.log 
