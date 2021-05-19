#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 100:00:00
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python

# declare run
run=100
pred_type="baseline"
netD="basic"
epoch="latest"
# seed=42

echo "starting training and testing run $run"

#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/Cityscapes
mkdir "$TMPDIR"/CopyGAN

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/CLEVR_colorized/images "$TMPDIR"/datasets/CLEVR_colorized/

for seed in 1 10 20 30 42
do
    echo "Seed: $seed"

    # execute training script
    python $HOME/City-GAN/train.py --model copy \
        --dataroot "$TMPDIR"/datasets/CLEVR_colorized/images\
        --batch_size 64\
        --n_epochs 10\
        --n_epochs_decay 30\
        --save_epoch_freq 10\
        --checkpoints_dir "$TMPDIR"/checkpoints/run"${run}"/seed"${seed}"\
        --print_freq 100\
        --update_html 100\
        --display_freq 100\
        --verbose \
        --sigma_blur 1\
        --load_size 70\
        --crop_size 64\
        --D_headstart 0\
        --confidence_weight 0.0\
        --val_batch_size 128\
        --accumulation_steps 1\
        --display_id 0\
        --lambda_aux 0.0\
        --D_threshold 0.5\
        --netD "${netD}"\
        --real_target 0.9\
        --fake_target 0.1\
        --seed "${seed}"\
        --pred_type_D "${pred_type}"\
        --use_amp\
        --noisy_labels\
        --n_alternating_batches 20\
        --val_freq 20


    # copy checkpoints to home directory
    mkdir -p $HOME/City-GAN/checkpoints/run"${run}"/seed"${seed}"
    cp -r "$TMPDIR"/checkpoints/run"${run}"/seed"${seed}"/CopyGAN/* $HOME/City-GAN/checkpoints/run"${run}"/seed"${seed}"/


    #### TESTING PART

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
        --seed "${seed}"\
        --epoch "${epoch}"\
        --verbose\
        > "$TMPDIR"/test_results_run"${run}"_seed"${seed}"_epoch"${epoch}".txt


    # copy results to home directory
    mkdir -p $HOME/City-GAN/results/CopyGAN/run"${run}"/seed"${seed}"
    cp -r "$TMPDIR"/results/CopyGAN/test_"${epoch}" $HOME/City-GAN/results/CopyGAN/run"${run}"/seed"${seed}"/
    cp "$TMPDIR"/test_results_run"${run}"_seed"${seed}"_epoch"${epoch}".txt $HOME/City-GAN/results/CopyGAN/run"${run}"/
done







