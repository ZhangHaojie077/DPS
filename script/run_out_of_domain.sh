generalization_type=dense # choose dense or mix
model_name=bert-large-cased 
gpu_id=0

declare -A dic
dic=([snli]=1500 [mnli]=1500)

subnetwork_pro=0.3
update_ratio=0.05
task=mnli # choose mnli or snli
seed=1


CUDA_VISIBLE_DEVICES=${gpu_id} python3 run_trainer.py \
--model_name_or_path ${model_name} \
--task_name ${task} \
--do_train \
--train_out_of_domain True \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 2e-5 \
--max_steps ${dic["${task}"]} \
--save_steps 30000 \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--fp16 \
--output_dir ./output \
--reserve_p ${subnetwork_pro} \
--seed ${seed} \
--generalization_type ${generalization_type} \
--update_ratio ${update_ratio}


for eval_task in {mnli,snli,sick,scitail,rte}
do
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 run_trainer.py \
    --model_name_or_path ./output \
    --task_name ${eval_task} \
    --train_out_of_domain True \
    --do_eval \
    --max_seq_length 128 \
    --seed ${seed} \
    --reserve_p ${subnetwork_pro} \
    --generalization_type ${generalization_type} \
    --output_dir ./output/eval_results \
    --update_ratio ${update_ratio} 
done