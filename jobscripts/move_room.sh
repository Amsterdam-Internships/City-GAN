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
run=7
echo "starting MoveGAN training run $run"

#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/ROOM
mkdir "$TMPDIR"/datasets/ROOM/images
mkdir "$TMPDIR"/datasets/ROOM/images/train
mkdir "$TMPDIR"/datasets/ROOM/images/val


# move the tar file in home to scratch
cp $HOME/City-GAN/datasets/ROOM/images/train/10k_train.tar.gz "$TMPDIR"/datasets/ROOM/images/train/

# ls  "$TMPDIR"/datasets/ROOM/images/train | head -n 5

echo "Tar file moved to scratch"
now=$(date +"%T")
echo "Current time : $now"
echo

# unpack the tar on scratch
tar -zxf "$TMPDIR"/datasets/ROOM/images/train/10k_train.tar.gz --strip-components 1 --directory "$TMPDIR"/datasets/ROOM/images/train/

ls "$TMPDIR/datasets/ROOM/images/train" | head -n 5

echo "Tar file extracted on scratch"
now=$(date +"%T")
echo "Current time : $now"
echo

cp $HOME/City-GAN/datasets/ROOM/images/val/1k_val.tar.gz "$TMPDIR"/datasets/ROOM/images/val/

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
python $HOME/City-GAN/train.py --model move \
    --dataroot "$TMPDIR"/datasets/ROOM/images/\
    --name Move\
    --max_dataset_size 10000\
    --batch_size 64\
    --n_epochs 5\
    --n_epochs_decay 15\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --display_id 0\
    --num_threads 4\
    --min_obj_surface 60\
    --print_freq 20\
    --display_freq 100\
    --update_html 100\
    --theta_dim 6


# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/checkpoints/room/run"${run}"
cp -r "$TMPDIR"/checkpoints $HOME/City-GAN/checkpoints/room/run"${run}"
