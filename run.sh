export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--cpu \
	--seed 0 \
	--architecture cseqgan \
	--config configs/csg_dz_s.json \
	--dataset jslads \
	# --test_index -1 \
	# --mode train \
	# --model dvyi0 \
	# --save dvyi0 \
	# --log \
