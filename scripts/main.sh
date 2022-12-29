#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=ImagenetR18
#SBATCH --mail-user=xcui32@fordham.edu
#SBATCH --mail-type=END
#SBATCH --output=/u/erdos/students/xcui32/SequentialTraining/results/VOCR18less_epoch/output2.out

module purg
module load gcc5 cuda10.1
module load openmpi/cuda/64
module load pytorch-py36-cuda10.1-gcc
module load ml-pythondeps-py36-cuda10.1-gcc


python3 /u/erdos/cnslab/xcui32/FSL/main.py \
 --model 'ProtoNet' --backbone_class 'convnet' --dataset_name 'omniglot' --root './data/data' \
 --n_ways_train 5 --n_ways_test 5 --n_shots_train 1 --n_shots_test 1  --n_queries_train 15  --n_queries_test 15 --temperature 1 \
 --start_epoch 0 --max_epoch 200 --episodes_per_epoch 100 --num_val_episodes 600 --num_test_episodes 10000 \
 --lr 0.001 --optimizer adam --lr_scheduler  --step_size 20 --gamma 0.2 \
 --momentum 0.9  --weight_decay 0.0005 --val_interval 1 \
 --num_workers 4 --device 'gpu' --download False --result_dir './results/run1' --save False --resume --init_backbone