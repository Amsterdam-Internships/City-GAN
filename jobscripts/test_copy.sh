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

# declare run
run=92
# epoch="latest"
seed=0
echo "Testing CopyGAN saved in run $run"

#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_colorized
mkdir "$TMPDIR"/CopyGAN

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/CLEVR_colorized/images "$TMPDIR"/datasets/CLEVR_colorized/

# copy all models to scratch
cp $HOME/City-GAN/checkpoints/run"${run}"/checkpoints/CopyGAN/* "$TMPDIR"/CopyGAN/

# create directory in home for results
mkdir -p $HOME/City-GAN/results/CopyGAN/run"${run}"/seed"${seed}"

for epoch in "10" "20" "30" "latest"
do
    echo "${epoch}"
    # execute training script
    python $HOME/City-GAN/test.py \
        --model copy \
        --num_test 5000\
        --dataroot "$TMPDIR"/datasets/CLEVR_colorized/images\
        --checkpoints_dir "$TMPDIR"\
        --epoch "${epoch}"\
        --results_dir "$TMPDIR"/results/ \
        --display_freq 10\
        --seed "${seed}"\
        --verbose\
        > "$TMPDIR"/test_results_run"${run}"_seed"${seed}"_epoch"${epoch}".txt



    cp -r "$TMPDIR"/results/CopyGAN/test_"${epoch}" $HOME/City-GAN/results/CopyGAN/run"${run}"/"${seed}"/
    cp "$TMPDIR"/test_results_run"${run}"_seed"{seed}"_epoch"${epoch}".txt $HOME/City-GAN/results/CopyGAN/run"${run}"/
done

