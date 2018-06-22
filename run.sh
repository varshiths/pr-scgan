export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--tensorboard 1 \
	--seed 0 \
	--config configs/cseqgan.json \
	--architecture cseqgan \
	--dataset jsla \
	--mode train \
	--model trial0 \
	--save trial0 \
	# --test_index 6 \
