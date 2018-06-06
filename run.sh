export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--dataset jsl \
	--config configs/seqgan.json \
	--architecture seqgan \
	--mode test \
	--model shuwa.wgan0 \
	--save shuwa.wgan0 \
	--test_index 2 \
