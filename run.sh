export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--cpu \
	--seed 0 \
	--architecture cseqgan \
	--config configs/cseqgan_dz.json \
	# --dataset jslad \
	# --test_index -1 \
	# --mode train \
	# --model dvyi0 \
	# --save dvyi0 \
	# --log \
