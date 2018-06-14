export CUDA_VISIBLE_DEVICES=0,1,2,3
time python3 main.py \
	--seed 0 \
	--dataset jslw \
	--config configs/seqgan_qw.json \
	--architecture seqgan \
	--mode train \
	--model qw_conv_disc_phrs_0 \
	--save qw_conv_disc_phrs_0 \
	# --test_index 4 \
