declare -a models=("resnet50" "resnext" "phinet3d" "resnet")

for model in "${models[@]}"; do
	for sample_duration in 4 8 12 16 24
	do
		CUDA_VISIBLE_DEVICES=7 python main.py \
			--root_path ~/ \
			--video_path ~/EHWDataset/ \
			--result_path exp5_num_frames_${model}/f${sample_duration} \
			--dataset ehwgesture \
			--n_classes 11 \
			--model "${model}" \
			--groups 3 \
			--width_mult 1.0 \
			--train_crop random \
			--learning_rate 0.01 \
			--sample_duration $sample_duration \
			--downsample 1 \
			--time_downsample 2 \
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