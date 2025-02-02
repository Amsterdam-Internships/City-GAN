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

run=20
phase="val"

#Create data directory on scratch
mkdir -p "$TMPDIR"/datasets/ROOM/images/"${phase}"

#Copy data file to scratch
if [ $phase = "train" ];
then
    cp $HOME/City-GAN/datasets/ROOM/tars/10k_train.tar.gz "$TMPDIR"/datasets/ROOM/images/test/
    tar -zxf "$TMPDIR"/datasets/ROOM/images/test/10k_train.tar.gz --strip-components 1 --directory "$TMPDIR"/datasets/ROOM/images/test/
elif [ $phase = "test" ];
then
    cp $HOME/City-GAN/datasets/ROOM/tars/1k_test.tar.gz "$TMPDIR"/datasets/ROOM/images/test/
    tar -zxf "$TMPDIR"/datasets/ROOM/images/test/1k_test.tar.gz --strip-components 1 --directory "$TMPDIR"/datasets/ROOM/images/test/
else
    cp $HOME/City-GAN/datasets/ROOM/tars/1k_val.tar.gz "$TMPDIR"/datasets/ROOM/images/val/
    tar -zxf "$TMPDIR"/datasets/ROOM/images/val/1k_val.tar.gz --strip-components 1 --directory "$TMPDIR"/datasets/ROOM/images/val/
fi

# copy model to scratch
mkdir -p $TMPDIR/run"${run}"/MoveModel
cp $HOME/City-GAN/checkpoints/room/run"${run}"/checkpoints/Move/latest_net_Conv.pth $TMPDIR/run"${run}"/MoveModel/

echo $phase

# execute training script
python $HOME/City-GAN/data/create_move_eval_dataset.py --model move \
    --dataroot "$TMPDIR"/datasets/ROOM/images\
    --checkpoints_dir "$TMPDIR"/run"${run}"/\
    --verbose \
    --min_obj_surface 50\
    --run "${run}"\
    --data_phase "${phase}"\
    --phase "${phase}"












