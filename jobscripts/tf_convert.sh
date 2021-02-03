#!/bin/bash
#Set job requirements
#SBATCH -n 1
#SBATCH -t 5:00:00
#SBATCH -p gpu_shared

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python


# Create output directory on scratch
mkdir "$TMPDIR"/datasets

# Copy data file to scratch
cp -r $HOME/City-GAN/datasets/ROOM "$TMPDIR"/datasets/

# execute conversion script
python tf_convert.py --data_dir "datasets/ROOM/test"

# copy files to home directory
cp -r "$TMPDIR"/datasets/ROOM/test/* $HOME/City-GAN/datasets/ROOM/test/
