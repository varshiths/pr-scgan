export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--dataset jslad \
	--architecture cseqgan \
	--config configs/cseqgan_dz.json \
	--model dvyi0 \
	--save dvyi0 \
	--test_index 5 \
	# --mode train \
	# --log \
