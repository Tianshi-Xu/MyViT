# pretrain cifar10
# CUDA_VISIBLE_DEVICES=1 python train.py -c configs/datasets/cifar10.yml --model vit_7_4_32 /home/xts/code/dataset/cifar10/

# pretrain cifar100
# CUDA_VISIBLE_DEVICES=2 python train.py -c configs/datasets/cifar100.yml --model vit_7_4_32_c100 /home/xts/code/dataset/cifar100/

# pretrain tiny
# CUDA_VISIBLE_DEVICES=4 python train.py -c configs/datasets/tiny_imagenet.yml --model vit_9_12_64 /home/xts/code/dataset/tiny-imagenet-200

# nas
CUDA_VISIBLE_DEVICES=3 python train_nas.py -c configs/datasets/ViT/cifar10_nas.yml --model vit_7_4_32 /home/xts/code/dataset/cifar10/

CUDA_VISIBLE_DEVICES=5 python train_nas.py -c configs/datasets/ViT/cifar100_nas.yml --model vit_7_4_32_c100 /home/xts/code/dataset/cifar100/

# MBV2
# CUDA_VISIBLE_DEVICES=5 python train.py -c configs/datasets/MBV2/tiny_imagenet.yml --model tiny_mobilenetv2 /home/xts/code/dataset/tiny-imagenet-200/
# fix block size
# CUDA_VISIBLE_DEVICES=4 python train_nas.py -c configs/datasets/MBV2/cifar100_nas.yml --model c100_nas_mobilenetv2 /home/xts/code/dataset/cifar100/
