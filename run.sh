export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--dataset jslad \
	--config configs/cseqgan_dz.json \
	--architecture cseqgan \
	# --mode train \
	# --model dvyi0 \
	# --save dvyi0 \
	# --test_index 5 \
	# --log \
