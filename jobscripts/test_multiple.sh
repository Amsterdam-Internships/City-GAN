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

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/CLEVR_colorized/images "$TMPDIR"/datasets/CLEVR_colorized/

# define seed for all experiments, make sure this is correct in the paths for loading the models TODO
seed=42
epoch="latest"

# declare run
for run in 100 101 102 103 110 111 112
do
    echo "Testing CopyGAN saved in run $run"
    for min_iou in 0.4 0.5 0.6 0.7 0.8 0.9 0.99
    do
        echo "Testing min_iou of ${min_iou}"


        # copy the model to scratch
        cp $HOME/City-GAN/checkpoints/run"${run}"/seed"${seed}"/latest_net_G.pth "$TMPDIR"/CopyGAN/

        # execute training script
        python $HOME/City-GAN/test.py \
            --model copy \
            --num_test 5000\
            --dataroot "$TMPDIR"/datasets/CLEVR_colorized/images\
            --checkpoints_dir "$TMPDIR"\
            --results_dir "$TMPDIR"/results/ \
            --display_freq 10\
            --epoch "${epoch}"\
            --seed "${seed}"\
            --verbose\
            > "$TMPDIR"/test_results_run"${run}"_seed"${seed}"_epoch"${epoch}"_"${min_iou}".txt

        # copy results to home directory
        mkdir -p $HOME/City-GAN/results/CopyGAN/run"${run}"/seed"${seed}"
        cp -r "$TMPDIR"/results/CopyGAN/test_"${epoch}" $HOME/City-GAN/results/CopyGAN/run"${run}"/seed"${seed}"
        cp "$TMPDIR"/test_results_run"${run}"_seed"${seed}"_epoch"${epoch}"_"${min_iou}".txt $HOME/City-GAN/results/CopyGAN/run"${run}"/
    done
done

echo "finished all runs"
