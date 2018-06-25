export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--log \
	--seed 0 \
	--config configs/cseqgan.json \
	--architecture cseqgan \
	--dataset jsla \
	--mode train \
	--model rninit0-445 \
	--save rninit0 \
	# --test_index 6 \
