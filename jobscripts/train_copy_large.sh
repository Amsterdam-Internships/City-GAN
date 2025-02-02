#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 30:00:00
#SBATCH -p gpu_shared

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python

# declare run
run=63
echo "starting training run $run"

#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_colorized

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/CLEVR_colorized/images "$TMPDIR"/datasets/CLEVR_colorized/

# execute training script
python $HOME/City-GAN/train.py --model copy \
    --dataroot "$TMPDIR"/datasets/CLEVR_colorized/images\
    --batch_size 32\
    --n_epochs 15\
    --n_epochs_decay 30\
    --save_epoch_freq 15\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --print_freq 20\
    --update_html 100 \
    --display_freq 100\
    --verbose \
    --sigma_blur 0.4\
    --load_size 130\
    --crop_size 128\
    --D_headstart 0\
    --confidence_weight 0.1\
    --val_batch_size 128\
    --accumulation_steps 1\
    --display_id 0\
    --lambda_aux 0.1\
    --D_threshold 0.5\
    --real_target 0.8\
    --patch_D\
    --no_alternate\

# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/checkpoints/run"${run}"
cp -r "$TMPDIR"/checkpoints $HOME/City-GAN/checkpoints/run"${run}"
