export CUDA_VISIBLE_DEVICES=0

for noise in "0.0" "0.2" "0.4" "0.6" "0.8"
do
#python tsne.py -b 512 --save-fig ./table --title SupCon_${noise} \
# --ckpt ./save/SupCon/cifar10_models/SupCon_cifar10_${noise}_resnet50_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth
#python tsne.py -b 512 --save-fig ./table --title SimCLR_${noise} \
# --ckpt ./save/SupCon/cifar10_models/SimCLR_cifar10_${noise}_resnet50_lr_0.5_decay_0.0001_bsz_1024_temp_0.5_trial_0_cosine_warm/last.pth
python tsne.py -b 512 --save-fig ./table --title SimCLR_${noise}-0.1 \
 --ckpt ./save/SupCon/cifar10_models/SimCLR_cifar10_${noise}_resnet50_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth
python tsne.py -b 512 --save-fig ./table --title SupCE_${noise} \
 --ckpt ./save/SupCon/cifar10_models/SupCE_cifar10_${noise}_resnet50_lr_0.8_decay_0.0001_bsz_1024_trial_0_cosine_warm/last.pth
done