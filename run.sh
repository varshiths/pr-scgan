export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--cpu \
	--seed 0 \
	--config configs/csg_dz_s.json \
	--dataset jslads \
	# --mode test \
	# --test_index 5 \
	# --architecture cseqgan \
	# --model dvyi0 \
	# --save dvyi0 \
	# --log \
