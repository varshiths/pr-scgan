export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--cpu \
	--seed 0 \
	--architecture cseqgan \
	--config configs/csg_dz_s.json \
	--dataset jslads \
	--mode train \
	# --test_index 5 \
	# --model dvyi0 \
	# --save dvyi0 \
	# --log \
