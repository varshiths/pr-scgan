export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--cpu \
	--seed 0 \
	--dataset jslad \
	--config configs/cseqgan_dz.json \
	--mode test \
	--test_index -1 \
	# --architecture cseqgan \
	# --model dvyi0 \
	# --save dvyi0 \
	# --log \
