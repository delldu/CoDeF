restruction()
{
	GPUS=0

	NAME=$1
	EXP_NAME=base

	echo "Restruction $NAME ..."

	ROOT_DIRECTORY="all_sequences/$NAME/$NAME"
	LOG_SAVE_PATH="logs/test_all_sequences/$NAME"

	MASK_DIRECTORY="all_sequences/$NAME/${NAME}_masks_0 all_sequences/$NAME/${NAME}_masks_1"

	WEIGHT_PATH=all_sequences/$NAME/${EXP_NAME}/${NAME}.ckpt
	# WEIGHT_PATH=ckpts/all_sequences/$NAME/${EXP_NAME}/step=10000.ckpt

	python test_multi.py --test --encode_w \
	                --root_dir $ROOT_DIRECTORY \
	                --log_save_path $LOG_SAVE_PATH \
	                --mask_dir $MASK_DIRECTORY \
	                --weight_path $WEIGHT_PATH \
	                --gpus $GPUS \
	                --config configs/${NAME}/${EXP_NAME}.yaml \
	                --exp_name ${EXP_NAME} \
	                --save_deform False
}


# restruction beauty_0
# restruction beauty_1
# restruction lemon_hit
# restruction scene_0
restruction white_smoke
