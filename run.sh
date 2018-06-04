export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--dataset jsl \
	--config configs/seqgan.json \
	--architecture seqgan \
	--save trial \
	--test_index 1 \
	# --mode train \
	# --model trial \
