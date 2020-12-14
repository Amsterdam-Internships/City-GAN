#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 1:00:00
#SBATCH -p gpu_shared

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python

#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_default

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/CLEVR_default/images "$TMPDIR"/datasets/CLEVR_default/

python -m visdom.server -p 8097

# execute training script
python $HOME/City-GAN/train.py --model copypasteGAN \
       --dataroot "$TMPDIR"/datasets/CLEVR_default/images\
       --batch_size 50 --n_epochs 1 --save_epoch_freq 1 \
       --checkpoints_dir "$TMPDIR"/checkpoints\
       --print_freq 50 --update_html 50 \
       --max_dataset_size 500\
       --display_freq 50 --verbose

# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/checkpoints/visdom_test
cp -r "$TMPDIR"/checkpoints $HOME/City-GAN/checkpoints/visdom_test
