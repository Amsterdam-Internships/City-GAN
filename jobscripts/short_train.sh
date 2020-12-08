#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 59:00
#SBATCH -p gpu_short
#SBATCH -N 1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

# Loading modules
module load 2020
module load Python

# print host to create ssh tunnel
echo "Running on following host:"
cat /etc/hosts

# Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_default

#Copy data files to scratch
cp -r $HOME/City-GAN/datasets/CLEVR_default/images "$TMPDIR"/datasets/CLEVR_default/

# Execute training script 
python $HOME/City-GAN/train.py --model copypasteGAN \
       --dataroot "$TMPDIR"/datasets/CLEVR_default/images \
       --n_epochs 1 --batch_size 50\
       --save_epoch_freq 1\
       --max_dataset_size 100\
       --checkpoints_dir "$TMPDIR"/checkpoints\
       --display_freq 10 --print_freq 10 --update_html_freq 10\
       --verbose


# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/checkpoints/test_run
cp -r "$TMPDIR"/checkpoints $HOME/City-GAN/checkpoints/test_run
