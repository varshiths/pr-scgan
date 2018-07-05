export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--cpu \
	--seed 0 \
	--dataset jslad \
	--architecture cseqgan \
	--config configs/cseqgan_dz.json \
	--mode train \
	--test_index -1 \
	# --model dvyi0 \
	# --save dvyi0 \
	# --log \
