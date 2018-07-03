export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--dataset jsla \
	--architecture cseqgan \
	--config configs/cseqgan.json \
	--mode test \
	--model dvyi0 \
	--save dvyi0 \
	--test_index -1 \
	# --log \
