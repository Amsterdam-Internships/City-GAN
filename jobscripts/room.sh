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

# declare run
run=1
echo "starting ROOM training run $run"

#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/ROOM

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/ROOM/ "$TMPDIR"/datasets/ROOM

# execute training script
python $HOME/City-GAN/train.py --model copypasteGAN \
    --dataroot "$TMPDIR"/datasets/ROOM/images\
    --name CopyGAN_room\
    --batch_size 128\
    --n_epochs 1\
    --n_epochs_decay 1\
    --save_epoch_freq 5\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --print_freq 20\
    --update_html 100 \
    --display_freq 100\
    --verbose \
    --sigma_blur 0.0\
    --load_size 70\
    --crop_size 64\
    --D_headstart 0\
    --confidence_weight 1.0\
    --val_batch_size 128\
    --accumulation_steps 1\
    --display_id 0\
    --lambda_aux 0.1\
    --D_threshold 0.5\
    --netD copy\
    --real_target 0.8\
    --patch_D\
    --seed 42\
    --no_alternate

# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/checkpoints/room/run"${run}"
cp -r "$TMPDIR"/checkpoints $HOME/City-GAN/checkpoints/room/run"${run}"
