#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 15:00:00
#SBATCH -p gpu_shared

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python

# declare run
RUN = 10

#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_colorized

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/CLEVR_colorized/images "$TMPDIR"/datasets/CLEVR_colorized/

# execute training script
python $HOME/City-GAN/train.py --model copypasteGAN \
    --dataroot "$TMPDIR"/datasets/CLEVR_colorized/images\
    --batch_size 50\
    --n_epochs 12\
    --n_epochs_decay 5\
    --save_epoch_freq 10\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --print_freq 1000\
    --update_html 5000 \
    --display_freq 5000\
    --verbose\
    --sigma_blur 1 \
    --D_headstart 80000

# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/checkpoints/run$RUN
cp -r "$TMPDIR"/checkpoints $HOME/City-GAN/checkpoints/run$RUN
