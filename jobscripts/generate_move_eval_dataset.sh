#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 3:00:00
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python

run=21

#Create data directory on scratch
mkdir "$TMPDIR"/datasets

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/ROOM/images "$TMPDIR"/datasets/ROOM/

# execute training script
python $HOME/City-GAN/create_move_eval_dataset.py.py --model move \
    --dataroot "$TMPDIR"/datasets/ROOM/images\
    --checkpoints_dir "$TMPDIR"/checkpoints/room/run"${run}"/checkpoints/\
    --verbose \
    --min_obj_surface 50












