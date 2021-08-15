CUDA_VISIBLE_DEVICES=0 python main_ce.py --batch_size 1024 --learning_rate 0.8 --cosine --syncBN --noise 0.0 --data_folder ./data/cifar10
CUDA_VISIBLE_DEVICES=0 python main_ce.py --batch_size 1024 --learning_rate 0.8 --cosine --syncBN --noise 0.2 --data_folder ./data/cifar10
CUDA_VISIBLE_DEVICES=0 python main_ce.py --batch_size 1024 --learning_rate 0.8 --cosine --syncBN --noise 0.4 --data_folder ./data/cifar10
CUDA_VISIBLE_DEVICES=0 python main_ce.py --batch_size 1024 --learning_rate 0.8 --cosine --syncBN --noise 0.6 --data_folder ./data/cifar10
CUDA_VISIBLE_DEVICES=0 python main_ce.py --batch_size 1024 --learning_rate 0.8 --cosine --syncBN --noise 0.8 --data_folder ./data/cifar10