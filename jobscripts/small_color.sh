#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 10:00:00
#SBATCH -p gpu_shared

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python

# declare run
run=78
echo "starting training run $run"

#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_colorized

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/CLEVR_colorized/images "$TMPDIR"/datasets/CLEVR_colorized/

# execute training script
python $HOME/City-GAN/train.py --model copy \
    --dataroot "$TMPDIR"/datasets/CLEVR_colorized/images\
    --batch_size 64\
    --n_epochs 5 \
    --n_epochs_decay 0 \
    --save_epoch_freq 10\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --print_freq 100\
    --update_html 100 \
    --display_freq 100\
    --verbose \
    --sigma_blur 1\
    --load_size 70\
    --crop_size 64\
    --D_headstart 0\
    --confidence_weight 0.0\
    --val_batch_size 128\
    --accumulation_steps 1\
    --display_id 0\
    --lambda_aux 1\
    --D_threshold 0.5\
    --netD copy\
    --real_target 0.75\
    --seed 42\
    --pool_D\

# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/checkpoints/run"${run}"
cp -r "$TMPDIR"/checkpoints $HOME/City-GAN/checkpoints/run"${run}"
