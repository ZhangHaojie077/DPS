
generalization_type=dense # choose dense or mix 
model_name=bert-large-cased 
gpu_id=0

declare -A dic
dic=([mrpc]=690 [rte]=468 [stsb]=1080 [cola]=1605) # max steps

subnetwork_pro=0.3
update_ratio=0.05
task=rte
seed=666

CUDA_VISIBLE_DEVICES=${gpu_id} python3 run_trainer.py \
--model_name_or_path ${model_name} \
--task_name ${task} \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 2e-5 \
--max_steps ${dic["${task}"]} \
--save_steps 30000 \
--evaluation_strategy epoch \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
--fp16 \
--output_dir ./output \
--seed ${seed} \
--reserve_p ${subnetwork_pro} \
--generalization_type ${generalization_type} \
--update_ratio ${update_ratio} 
