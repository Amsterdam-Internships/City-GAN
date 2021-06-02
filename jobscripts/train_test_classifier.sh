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
# mode="Resnet18"
mode="Resnet18_pretrained"
# mode="Default"
echo "starting training classifier (run ${run})"

#Create output directory on scratch
mkdir -p "$TMPDIR"/datasets/ROOM_composite

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/ROOM_composite/* "$TMPDIR"/datasets/ROOM_composite/

# execute training script
python $HOME/City-GAN/train.py --model classifier \
    --dataroot "$TMPDIR"/datasets/ROOM_composite\
    --batch_size 64\
    --n_epochs 100\
    --n_epochs_decay 200\
    --save_epoch_freq 100\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --print_freq 50\
    --update_html 100\
    --display_freq 50\
    --verbose \
    --display_id 0\
    --seed "${seed}"\
    --run ${run}\
    --model_type ${mode} \
    --freeze_resnet


# copy results to home directory
mkdir -p $HOME/City-GAN/checkpoints/Classifier/run"${run}"/"${mode}"
cp -r "$TMPDIR"/checkpoints/Classifier/* $HOME/City-GAN/checkpoints/Classifier/run"${run}"/"${mode}"/


# test the performance of the classifier
python $HOME/City-GAN/test.py --model classifier\
    --dataroot "$TMPDIR"/datasets/ROOM_composite\
    --checkpoints_dir "$TMPDIR"/checkpoints\
    --results_dir "$TMPDIR"/results/ \
    --run ${run}\
    --batch_size 64\
    --seed "${seed}"\
    --model_type "${mode}" \
    > "$TMPDIR"/test_results_run"${run}"_"${mode}".txt

echo "finished testing run '${run}'"

mkdir -p $HOME/City-GAN/results/Classifier/run"${run}"/"${mode}"
    cp -r "$TMPDIR"/results/Classifier/test_latest/* $HOME/City-GAN/results/Classifier/run"${run}"/"${mode}"
    cp "$TMPDIR"/test_results_run"${run}"_"${mode}".txt $HOME/City-GAN/results/Classifier/run"${run}"/"${mode}"

echo "everything copied back to Home folder $HOME/City-GA\
N/results/Classifier/run"${run}"/"${mode}""
 




