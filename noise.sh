
# Training SupCon features (using labels)
python main_supcon.py --batch_size 1024 --learning_rate 0.5 --temp 0.1 --cosine \
  --noise 0.0 --data_folder ./data/cifar10 --method SupCon

python main_supcon.py --batch_size 1024 --learning_rate 0.5 --temp 0.1 --cosine \
  --noise 0.2 --data_folder ./data/cifar10 --method SupCon

python main_supcon.py --batch_size 1024 --learning_rate 0.5 --temp 0.1 --cosine \
  --noise 0.4 --data_folder ./data/cifar10 --method SupCon

python main_supcon.py --batch_size 1024 --learning_rate 0.5 --temp 0.1 --cosine \
  --noise 0.8 --data_folder ./data/cifar10 --method SupCon

# Training SimCLR features (without labels)
python main_supcon.py --batch_size 1024 --learning_rate 0.5 --temp 0.5 --cosine \
  --noise 0.0 --data_folder ./data/cifar10 --method SupCon

python main_supcon.py --batch_size 1024 --learning_rate 0.5 --temp 0.5 --cosine \
  --noise 0.2 --data_folder ./data/cifar10 --method SupCon

python main_supcon.py --batch_size 1024 --learning_rate 0.5 --temp 0.5 --cosine \
  --noise 0.4 --data_folder ./data/cifar10 --method SupCon

python main_supcon.py --batch_size 1024 --learning_rate 0.5 --temp 0.5 --cosine \
  --noise 0.8 --data_folder ./data/cifar10 --method SupCon


# Training last linear layer
#python main_linear.py --batch_size 512 --learning_rate 5 \
#  --ckpt ./save/SupCon/cifar10_models/SupCon_cifar10_0.4_resnet50_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/last.pth \
#  --yfile ./save/SupCon/cifar10_models/SupCon_cifar10_0.4_resnet50_lr_0.5_decay_0.0001_bsz_1024_temp_0.1_trial_0_cosine_warm/y.npy