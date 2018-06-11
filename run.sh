export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--dataset jsl \
	--architecture seqgan \
	--config configs/seqgan_quart.json \
	--model trial1 \
	--save trial1 \
	--mode test \
	--test_index 2 \
