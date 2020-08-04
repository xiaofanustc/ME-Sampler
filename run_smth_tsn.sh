#!/bash/sh

CUDA_VISIBLE_DEVICES=0 python train.py \
                     -c configs/pretrained/config_model1.json \
                     -g 0 -e --use_cuda \
                     --data-dir /data/uts700/hu/smth-smth/smth-smth-val-frames \
                     --val-list /home/huzhang/dataset/smth-smth/tsn_index_attack.txt \
                     --mpeg4_video_file /data/uts700/hu/smth-smth/smth-smth-val-mpeg4/ \
                     --frame_save_file /data/uts700/hu/smth-smth/save_smth_imgs_tsn \
                     --model resnet50_v1b_sthsthv2 \
                     --model_type tsn2d \
                     --num-classes 174 \
                     --mode hybrid \
                     --dtype float32 \
                     --prefetch-ratio 1.0 \
                     --batch-size 1 \
                     --num-gpus 1 \
                     --num-data-workers 4 \
                     --new-height 256 \
                     --new-width 340 \
                     --new-length 1 \
                     --new-step 1 \
                     --input-size 256 \
                     --num-segments 8 \
                     --use-pretrained \
                     --interval 10
