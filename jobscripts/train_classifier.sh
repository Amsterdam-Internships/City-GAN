#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 15:00:00
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python

# declare run
run=21
seed=42
echo "starting training classifier (run ${run})"

#Create output directory on scratch
mkdir -p "$TMPDIR"/datasets/ROOM_composite

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/ROOM_composite/* "$TMPDIR"/datasets/ROOM_composite/

# execute training script
python $HOME/City-GAN/train.py --model classifier \
    --dataroot "$TMPDIR"/datasets/ROOM_composite\
    --batch_size 64\
    --n_epochs 10\
    --n_epochs_decay 0\
    --save_epoch_freq 10\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --print_freq 100\
    --update_html 100\
    --display_freq 100\
    --verbose \
    --display_id 0\
    --seed "${seed}"\
    --run ${run}


# copy results to home directory
cp -r "$TMPDIR"/checkpoints/* $HOME/City-GAN/Classifier/run"${run}"/


# test the performance of the classifier
# python $HOME/City-GAN/test.py --model classifier \
