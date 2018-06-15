export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--dataset jsla \
	--config configs/cseqgan.json \
	# --architecture cseqgan \
	# --mode train \
	# --model trial \
	# --save trial \
	# --test_index 4 \
