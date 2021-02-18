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
run=1
echo "starting ROOM training run $run"

#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/ROOM
mkdir "$TMPDIR"/datasets/ROOM/images
mkdir "$TMPDIR"/datasets/ROOM/images/train
mkdir "$TMPDIR"/datasets/ROOM/images/val

#Copy data file to scratch

# old way of copying, in loops
# for i in {0..9}; do
  #  cp -r $HOME/City-GAN/datasets/ROOM/images/train/images_train/99"$i"*.jpg "$TMPDIR"/datasets/ROOM/images/train/
#done

# perhaps this can be done using tar file:
# create tar file from training data, move to scratch, and unzip the archive

# this tar file is already in the home folder
# tar -zcf tar_train.tar.gz $HOME/City-GAN/datasets/ROOM/images/train/images_train/

# move the tar file in home to scratch
cp $HOME/City-GAN/datasets/ROOM/images/train/10k_train.tar.gz "$TMPDIR"/datasets/ROOM/images/train/

echo "Tar file moved to scratch"
now=$(date +"%T")
echo "Current time : $now"
echo

# unpack the tar on scratch
tar -zxf "$TMPDIR"/datasets/ROOM/images/train/10k_train.tar.gz --strip-components 1 --directory "$TMPDIR"/datasets/ROOM/images/train/

echo "Tar file extracted on scratch"
now=$(date +"%T")
echo "Current time : $now"
echo

# old way of copying
# cp -r $HOME/City-GAN/datasets/ROOM/images/train/images_train/111**.jpg "$TMPDIR"/datasets/ROOM/images/val/

cp cp $HOME/City-GAN/datasets/ROOM/images/val/1k_val.tar.gz "$TMPDIR"/datasets/ROOM/images/val/

echo "Validation tar copied to scratch"
now=$(date +"%T")
echo "Current time : $now"
echo

tar -zxf "$TMPDIR"/datasets/ROOM/images/val/1k_val.tar.gz --strip-components 1 --directory "$TMPDIR"/datasets/ROOM/images/val/


echo "validation tar extracted on scratch"
now=$(date +"%T")
echo "Current time : $now"
echo

# execute training script
python $HOME/City-GAN/move.py --model copypasteGAN \
    --dataroot "$TMPDIR"/datasets/ROOM/images/\
    --max_dataset_size 10000\
    --name CopyGAN_room\
    --batch_size 64\
    --n_epochs 10\
    --n_epochs_decay 5\
    --save_epoch_freq 5\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --print_freq 20\
    --update_html 100 \
    --display_freq 100\
    --verbose \
    --sigma_blur 0.0\
    --load_size 64\
    --crop_size 64\
    --D_headstart 0\
    --confidence_weight 1.0\
    --val_batch_size 128\
    --val_freq 200\
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
