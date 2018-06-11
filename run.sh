export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--dataset jsl \
	--architecture seqgan \
	--config configs/seqgan_quart.json \
	--model qwgan0 \
	--save qwgan0 \
	--mode train \
	--test_index 4 \
