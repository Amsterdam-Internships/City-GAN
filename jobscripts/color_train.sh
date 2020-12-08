#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 5:00:00
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

# execute training script
python $HOME/City-GAN/train.py --model copypasteGAN \
       --dataroot "$TMPDIR"/datasets/CLEVR_default/images\
       --batch_size 50 --n_epochs 5 --save_epoch_freq 1 \
       --checkpoints_dir "$TMPDIR"/checkpoints\
       --print_freq 1000 --update_html 10000 \
       --display_freq 1000 --verbose

# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/checkpoints/run3
cp -r "$TMPDIR"/checkpoints $HOME/City-GAN/checkpoints/run3
