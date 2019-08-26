#!/bin/sh

OUTPUT_DIR='./outputs/m1base_7.5w'
TRAIN_DATA_PATH='./datas/7.5w/pfld_train_data.rec'
VALID_DATA_PATH='./datas/7.5w/pfld_valid_data.rec'
LEARNING_RATE=0.0004
BATCH_SIZE=128
EPOCHES=1000
GPU_IDS='0,1'
IMAGE_SIZE=96
MODEL_TYPE='M1BASE'
WITH_ANGLE=1
WITH_CATEGORY=0
ALPHA=1.0

python train.py --output_dir=${OUTPUT_DIR} --train_data_path=${TRAIN_DATA_PATH} --valid_data_path=${VALID_DATA_PATH} --learning_rate=${LEARNING_RATE} --batch_size=${BATCH_SIZE} --epoches=${EPOCHES} --gpu_ids=${GPU_IDS} --image_size=${IMAGE_SIZE} --model_type=${MODEL_TYPE} --with_angle_loss=${WITH_ANGLE} --with_category_loss=${WITH_CATEGORY} --alpha=${ALPHA}
