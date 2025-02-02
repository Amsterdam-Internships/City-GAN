#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 120:00:00
#SBATCH -p gpu_shared
#SBATCH --gpus-per-node=1

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

#Loading modules
module load 2020
module load Python

# declare run
run=4
pred_type="baseline"
netD="copy"
epoch="latest"
img_size=256
# seed=42

echo "starting training and testing run $run"

#Create output directory on scratch
mkdir -p "$TMPDIR"/datasets/Cityscapes/src_imgs/images
mkdir -p "$TMPDIR"/datasets/Cityscapes/src_imgs/annotations
mkdir -p "$TMPDIR"/datasets/Cityscapes/gtFine
mkdir "$TMPDIR"/CopyGAN

# Copy data file to scratch
cp -r $HOME/City-GAN/datasets/Cityscapes/leftImg8bit_trainvaltest.zip "$TMPDIR"/datasets/Cityscapes/data.zip
# unzip the data and remove zip archive
unzip -q "$TMPDIR"/datasets/Cityscapes/data.zip -d "$TMPDIR"/datasets/Cityscapes/
rm "$TMPDIR"/datasets/Cityscapes/data.zip

# copy the masks to scratch
cp -r $HOME/City-GAN/datasets/Cityscapes/gtFine.zip "$TMPDIR"/datasets/Cityscapes/gtFine.zip
unzip -q "$TMPDIR"/datasets/Cityscapes/gtFine.zip -d "$TMPDIR"/datasets/Cityscapes
rm "$TMPDIR"/datasets/Cityscapes/gtFine.zip

ls "$TMPDIR"/datasets/Cityscapes/

# copy the source images to scratch
cp -r $HOME/City-GAN/datasets/COCO/images/* "$TMPDIR"/datasets/Cityscapes/src_imgs/images/
cp -r $HOME/City-GAN/datasets/COCO/annotations/* "$TMPDIR"/datasets/Cityscapes/src_imgs/annotations

for seed in 42
do
    echo "Seed: $seed"

    # execute training script
    python $HOME/City-GAN/train.py --model copy \
        --dataroot "$TMPDIR"/datasets/Cityscapes\
        --dataset_mode cityscapes\
        --batch_size 32\
        --n_epochs 100\
        --n_epochs_decay 100\
        --save_epoch_freq 10\
        --checkpoints_dir "$TMPDIR"/checkpoints/ \
        --print_freq 100\
        --update_html 100\
        --display_freq 100\
        --verbose \
        --sigma_blur 1\
        --load_size "${img_size}"\
        --crop_size "${img_size}"\
        --D_headstart 0\
        --confidence_weight 0.0\
        --val_batch_size 64\
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
        --val_freq 20\


    # copy checkpoints to home directory
    mkdir -p $HOME/City-GAN/checkpoints/CopyGANCityscapes/run"${run}"/seed"${seed}"
    cp -r "$TMPDIR"/checkpoints/CopyGAN/* $HOME/City-GAN/checkpoints/CopyGANCityscapes/run"${run}"/seed"${seed}"/


    #### TESTING PART

    # copy the model to scratch
    cp $HOME/City-GAN/checkpoints/CopyGANCityscapes/run"${run}"/seed"${seed}"/latest_net_G.pth "$TMPDIR"/CopyGAN/

    # execute training script
    python $HOME/City-GAN/test.py \
        --model copy \
        --dataset_mode cityscapes \
        --num_test 5000\
        --dataroot "$TMPDIR"/datasets/Cityscapes/ \
        --checkpoints_dir "$TMPDIR"\
        --results_dir "$TMPDIR"/results/ \
        --display_freq 21\
        --load_size "${img_size}"\
        --crop_size "${img_size}"\
        --seed "${seed}"\
        --epoch "${epoch}"\
        --verbose\
        > "$TMPDIR"/test_results_cityscapes_run"${run}"_seed"${seed}"_epoch"${epoch}".txt


    # copy results to home directory
    mkdir -p $HOME/City-GAN/results/CopyGANCityscapes/run"${run}"/seed"${seed}"
    cp -r "$TMPDIR"/results/CopyGAN/test_"${epoch}" $HOME/City-GAN/results/CopyGANCityscapes/run"${run}"/seed"${seed}"/
    cp "$TMPDIR"/test_results_cityscapes_run"${run}"_seed"${seed}"_epoch"${epoch}".txt $HOME/City-GAN/results/CopyGANCityscapes/run"${run}"/
done







