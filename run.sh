export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--log \
	--seed 0 \
	--config configs/cseqgan.json \
	--architecture cseqgan \
	--dataset jsla \
	--mode train \
	# --model rninit1 \
	# --save rninit1 \
	# --test_index -1 \
