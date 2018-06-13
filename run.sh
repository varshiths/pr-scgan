export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--architecture seqgan \
	--config configs/seqgan_quart.json \
	--mode test \
	--dataset jsl \
	--test_index 4 \
	--model shuwa.qwgan0 \
	# --save shuwa.qwgan0 \
