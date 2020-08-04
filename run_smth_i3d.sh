#!/bash/sh

CUDA_VISIBLE_DEVICES=3 python train.py \
                     -c configs/pretrained/config_model1.json \
                     -g 0 -e --use_cuda \
                     --data-dir /data/uts700/hu/smth-smth/smth-smth-val-frames \
                     --val-list /home/huzhang/dataset/smth-smth/i3d_index_attack.txt \
                     --mpeg4_video_file /data/uts700/hu/smth-smth/smth-smth-val-mpeg4/ \
                     --frame_save_file /data/uts700/hu/smth-smth/save_smth_imgs_i3d \
                     --model i3d_resnet50_v1_sthsthv2 \
                     --model_type i3d \
                     --num-classes 174 \
                     --mode hybrid \
                     --dtype float32 \
                     --prefetch-ratio 1.0 \
                     --batch-size 1 \
                     --num-gpus 1 \
                     --num-data-workers 4 \
                     --new-height 256 \
                     --new-width 340 \
                     --new-length 16 \
                     --new-step 2 \
                     --input-size 256 \
                     --num-segments 2 \
                     --use-pretrained \
                     --interval 10
