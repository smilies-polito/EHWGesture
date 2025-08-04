declare -a models=("resnet50" "resnext" "phinet3d" "resnet")

for model in "${models[@]}"; do
	for time_downsample in 1 2 3 4
	do
		CUDA_VISIBLE_DEVICES=5 python main.py \
			--root_path ~/ \
			--video_path ~/EHWDataset/ \
			--result_path exp4_time_downsample_${model}/d${time_downsample} \
			--dataset ehwgesture \
			--n_classes 11 \
			--model "${model}" \
			--groups 3 \
			--width_mult 1.0 \
			--train_crop random \
			--learning_rate 0.01 \
			--sample_duration 12 \
			--downsample 1 \
			--time_downsample $time_downsample \
			--batch_size 16 \
			--checkpoint 1 \
			--ehw_cam "event,master,sub2" \
			--ehw_input "event,depth,rgb" \
			--n_epochs 20 \
			--pretrain_epochs 5 \
			--random_mask_fraction 0 \
			--n_threads 32
	done
done