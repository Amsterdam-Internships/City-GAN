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
echo "starting Move room training run $run"

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

ls  "$TMPDIR"/datasets/ROOM/images/train | head -n 5

echo "Tar file moved to scratch"
now=$(date +"%T")
echo "Current time : $now"
echo

# unpack the tar on scratch
tar -zxf "$TMPDIR"/datasets/ROOM/images/train/10k_train.tar.gz --strip-components 1 --directory "$TMPDIR"/datasets/ROOM/images/train/

ls "$TMPDIR/datasets/ROOM/images/train"

echo "Tar file extracted on scratch"
now=$(date +"%T")
echo "Current time : $now"
echo

# old way of copying
# cp -r $HOME/City-GAN/datasets/ROOM/images/train/images_train/111**.jpg "$TMPDIR"/datasets/ROOM/images/val/

# cp cp $HOME/City-GAN/datasets/ROOM/images/val/1k_val.tar.gz "$TMPDIR"/datasets/ROOM/images/val/

# echo "Validation tar copied to scratch"
# now=$(date +"%T")
# echo "Current time : $now"
# echo

# tar -zxf "$TMPDIR"/datasets/ROOM/images/val/1k_val.tar.gz --strip-components 1 --directory "$TMPDIR"/datasets/ROOM/images/val/


# echo "validation tar extracted on scratch"
# now=$(date +"%T")
# echo "Current time : $now"
# echo

# execute training script
python $HOME/City-GAN/train_move.py --model move \
    --dataroot "$TMPDIR"/datasets/ROOM/images/\
    --name Move\
    --max_dataset_size 10\
    --batch_size 5\
    --epoch 20\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --display_id 0\
    --num_threads 1\
    --min_obj_surface 55


# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/checkpoints/room/run"${run}"
cp -r "$TMPDIR"/checkpoints $HOME/City-GAN/checkpoints/room/run"${run}"
