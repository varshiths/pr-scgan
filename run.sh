export CUDA_VISIBLE_DEVICES=0
time python3 main.py \
	--seed 0 \
	--dataset jsl \
	--config configs/seqgan.json \
	--architecture seqgan \
	--mode train \
	# --test_index 1 \
	# --save trial \
	# --model trial \
