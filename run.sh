export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--config configs/cseqgan.json \
	--dataset jsla \
	--mode test \
	--test_index -1 \
	# --model trial \
	# --save trial \
	# --architecture cseqgan \
