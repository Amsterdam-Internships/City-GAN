#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 5:00:00
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python

# declare run
run=7
seed=42
img_dim=256

echo "Starting run $run"

#Create output directory on scratch
mkdir -p "$TMPDIR"/datasets/Cityscapes/src_imgs/images
mkdir -p "$TMPDIR"/datasets/Cityscapes/src_imgs/annotations
mkdir -p "$TMPDIR"/datasets/Cityscapes/gtFine
mkdir "$TMPDIR"/CopyGAN

# Copy data file to scratch
cp -r $HOME/City-GAN/datasets/Cityscapes/leftImg8bit_trainvaltest.zip "$TMPDIR"/datasets/Cityscapes/data.zip
# unzip the data and remove zip archive
unzip -q "$TMPDIR"/datasets/Cityscapes/data.zip -d "$TMPDIR"/datasets/Cityscapes/
rm "$TMPDIR"/datasets/Cityscapes/data.zip

echo "Cityscapes data copied to scratch"

# copy the source images to scratch
cp -r $HOME/City-GAN/datasets/COCO/images/* "$TMPDIR"/datasets/Cityscapes/src_imgs/images/
cp -r $HOME/City-GAN/datasets/COCO/annotations/* "$TMPDIR"/datasets/Cityscapes/src_imgs/annotations

# Min obj surface was 60

# execute training script
python $HOME/City-GAN/train.py --model move \
    --dataroot "$TMPDIR"/datasets/Cityscapes/ \
    --dataset_mode move_coco\
    --name Move\
    --max_dataset_size 10000\
    --batch_size 64\
    --n_epochs 10\
    --n_epochs_decay 20\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --display_id 0\
    --num_threads 4\
    --min_obj_surface 100\
    --print_freq 20\
    --display_freq 10\
    --update_html 10\
    --theta_dim 6\
    --fake_target 0.1\
    --real_target 0.9\
    --verbose\
    --seed "${seed}"\
    --preprocess resize_and_crop \
    --load_size ${img_dim} \
    --crop_size ${img_dim} \
    --lr 2e-5

# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/checkpoints/MoveCityscapes/run"${run}"
cp -r "$TMPDIR"/checkpoints/Move/* $HOME/City-GAN/checkpoints/MoveCityscapes/run"${run}"/
