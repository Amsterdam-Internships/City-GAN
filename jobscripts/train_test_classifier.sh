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
run=20
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
    --n_epochs_decay 10\
    --save_epoch_freq 10\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --print_freq 50\
    --update_html 100\
    --display_freq 50\
    --verbose \
    --display_id 0\
    --seed "${seed}"\
    --run ${run}\
    --use_resnet18\
    --use_pretrained\


# copy results to home directory
mkdir -p $HOME/City-GAN/checkpoints/Classifier/run"${run}"/
cp -r "$TMPDIR"/checkpoints/Classifier/* $HOME/City-GAN/checkpoints/Classifier/run"${run}"/


# test the performance of the classifier
python $HOME/City-GAN/test.py --model classifier\
    --dataroot "$TMPDIR"/datasets/ROOM_composite\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --results_dir "$TMPDIR"/results/ \
    --run ${run}\
    --batch_size 64\
    --seed "${seed}"\
    --use_resnet18\
    --use_pretrained
    > "$TMPDIR"/test_results_run"${run}".txt

echo "finished testing run '${run}'"

mkdir -p $HOME/City-GAN/results/Classifier/run"${run}"/
    cp -r "$TMPDIR"/results/Classifier/test_latest/* $HOME/City-GAN/results/Classifier/run"${run}"/
    cp "$TMPDIR"/test_results_run"${run}".txt $HOME/City-GAN/results/Classifier/run"${run}"/

echo "everything copied back to Home folder"




