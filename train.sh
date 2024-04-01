# pretrain cifar10
# CUDA_VISIBLE_DEVICES=1 python train.py -c configs/datasets/cifar10.yml --model vit_7_4_32 /home/xts/code/dataset/cifar10/

# pretrain cifar100
# CUDA_VISIBLE_DEVICES=2 python train.py -c configs/datasets/cifar100.yml --model vit_7_4_32_c100 /home/xts/code/dataset/cifar100/

# pretrain tiny
# CUDA_VISIBLE_DEVICES=4 python train.py -c configs/datasets/tiny_imagenet.yml --model vit_9_12_64 /home/xts/code/dataset/tiny-imagenet-200

# fix_block_size
CUDA_VISIBLE_DEVICES=7 python train_nas.py -c configs/datasets/cifar10_nas.yml --model vit_7_4_32 /home/xts/code/dataset/cifar10/
