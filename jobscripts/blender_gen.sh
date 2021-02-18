#!/bin/bash
#Set job requirements
#SBATCH -t 25:00:00
#SBATCH -p gpu_shared
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=tom.lotze@gmail.com

# Loading modules
module load 2020
module load Python
module load Blender

# Create output directory on scratch
mkdir "$TMPDIR"/datasets
mkdir "$TMPDIR"/datasets/CLEVR_colorized
mkdir "$TMPDIR"/datasets/CLEVR_colorized/jsons
mkdir "$TMPDIR"/datasets/CLEVR_colorized/images


# Execute training script
blender --background -noaudio --python data/render_images.py -- \
	--output_image_dir "$TMPDIR"/datasets/CLEVR_colorized/images\
	--output_scene_dir "$TMPDIR"/datasets/CLEVR_colorized/jsons\
	--output_scene_file "$TMPDIR"/datasets/CLEVR_colorized/CLEVR_scenes.json\
	--base_scene_blendfile data/data/base_scene.blend\
	--filename_prefix "CLEVR_color"\
	--split ""\
	--min_objects 1\
	--max_objects 6\
	--num_images 100\
	--start_idx 28000\
	--use_gpu 1\
	--render_num_samples 20\
	--render_min_bounces 0\
	--render_max_bounces 4\
	--width 270 --height 270\
	--min_dist 0.1\
	--margin 0.2
# ultimately, print output to dev 0... (and set verbosity)

ls "$TMPDIR"/datasets/CLEVR_colorized


# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/datasets/CLEVR_colorized
mkdir -p $HOME/City-GAN/datasets/CLEVR_colorized/images
mkdir -p $HOME/City-GAN/datasets/CLEVR_colorized/jsons
cp -r "$TMPDIR"/datasets/CLEVR_colorized/images/* $HOME/City-GAN/datasets/CLEVR_colorized/images/with_masks/
cp -r "$TMPDIR"/datasets/CLEVR_colorized/jsons/* $HOME/City-GAN/datasets/CLEVR_colorized/jsons/
