#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 12:00:00
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python

# declare run
run=112
aux=0.1
epoch="latest"

echo "starting training and testing run $run with lambda_aux $aux"

#Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_colorized
mkdir "$TMPDIR"/CopyGAN

#Copy data file to scratch
cp -r $HOME/City-GAN/datasets/CLEVR_colorized/images "$TMPDIR"/datasets/CLEVR_colorized/

# for type in "baseline" "pool" "conv"
for type in "conv"
do
    echo "\n\nTraining run ${run} with pred-type ${type}"
    # set aux loss correctly
    # for seed in 1 10 20 30 42
    for seed in 30 42
    do
        echo "Seed: $seed"
        # execute training script
        python $HOME/City-GAN/train.py --model copy \
            --dataroot "$TMPDIR"/datasets/CLEVR_colorized/images\
            --batch_size 64\
            --n_epochs 10\
            --n_epochs_decay 30\
            --save_epoch_freq 20\
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
            --accumulation_steps 1\
            --display_id 0\
            --lambda_aux "${aux}"\
            --D_threshold 0.5\
            --netD copy\
            --real_target 0.9\
            --fake_target 0.1\
            --seed "${seed}"\
            --pred_type_D "${type}"\
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
            --lambda_aux "${aux}"\
            --verbose\
            > "$TMPDIR"/test_results_run"${run}"_seed"${seed}"_epoch"${epoch}".txt


        # copy results to home directory
        mkdir -p $HOME/City-GAN/results/CopyGAN/run"${run}"/seed"${seed}"
        cp -r "$TMPDIR"/results/CopyGAN/test_"${epoch}" $HOME/City-GAN/results/CopyGAN/run"${run}"/seed"${seed}"/
        cp "$TMPDIR"/test_results_run"${run}"_seed"${seed}"_epoch"${epoch}".txt $HOME/City-GAN/results/CopyGAN/run"${run}"/
    done
    # Increment run number
    ((run=run+1))
done










