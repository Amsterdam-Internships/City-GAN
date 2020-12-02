#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 59:00
#SBATCH -p gpu_short

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load Python



#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_default

#Copy input file to scratch
cp -r $HOME/City-GAN/datasets/CLEVR_default/images "$TMPDIR"/datasets/CLEVR_default/

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python $HOME/City-GAN/train.py --model copypasteGAN --dataroot "$TMPDIR"/datasets/CLEVR_default/images --batch_size 80 --n_epochs 3 --save_epoch_freq 1


#Copy output directory from scratch to home

mkdir -p $HOME/City-GAN/checkpoints/test_run


# copy checkpoints to home directory
cp -r "$TMPDIR"/checkpoints $HOME/City-GAN/checkpoints/test_run
