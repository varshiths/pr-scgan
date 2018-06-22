export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--config configs/cseqgan.json \
	--architecture cseqgan \
	--dataset jsla \
	--mode test \
	--model shuwa.trial0 \
	--save shuwa.trial0 \
	--test_index 6 \
