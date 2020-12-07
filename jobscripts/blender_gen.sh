#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 59:00
#SBATCH -p gpu_short

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

# Loading modules
module load 2020
module load Python

# Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_colorized
mkdir "$TMPDIR"/datasets/CLEVR_colorized/jsons

# Execute training script 
blender --background -noaudio --python data/render_images.py -- \
	--output_image_dir "$TMPDIR"/datasets/CLEVR_colorized\
	--output_scene_dir "$TMPDIR"/datasets/CLEVR_colorized/jsons\
	--base_scene_blendfile data/data/base_scene.blend\
	--filename_prefix "CLEVR_color"\
	--split ""\
	--min_objects 2\
	--max_objects 5\
	--num_images 10\
	--use_gpu 1

ls "$TMPDIR"/datasets/CLEVR_colorized


# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/datasets/
cp -r "$TMPDIR"/datasets/CLEVR_colorized $HOME/City-GAN/datasets
