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


#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_colorized
mkdir "$TMPDIR"/CopyGAN

# declare run
for run in 13 14 55 74
do
    echo "Testing CopyGAN saved in run $run"

    #Copy data file to scratch
    cp -r $HOME/City-GAN/datasets/CLEVR_colorized/images "$TMPDIR"/datasets/CLEVR_colorized/

    # copy the model to scratch
    cp $HOME/City-GAN/checkpoints/run"${run}"/checkpoints/CopyGAN/latest_net_G.pth "$TMPDIR"/CopyGAN/

    # execute training script
    python $HOME/City-GAN/test.py \
        --model copy \
        --num_test 5000\
        --dataroot "$TMPDIR"/datasets/CLEVR_colorized/images\
        --checkpoints_dir "$TMPDIR"\
        --results_dir "$TMPDIR"/results/ \
        --display_freq 10\
        --seed 42\
        --verbose\
        > "$TMPDIR"/test_results_run"${run}".txt

    # copy results to home directory
    mkdir -p $HOME/City-GAN/results/CopyGAN/run"${run}"
    cp -r "$TMPDIR"/results/CopyGAN/test_latest/* $HOME/City-GAN/results/CopyGAN/run"${run}"
    cp "$TMPDIR"/test_results_run"${run}".txt $HOME/City-GAN/results/CopyGAN/run"${run}"/
done

echo "finished all runs"
