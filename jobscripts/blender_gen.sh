#!/bin/bash
#Set job requirements
#SBATCH -N 1
#SBATCH -t 20:00
#SBATCH -p gpu_shared

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
	--output_scene_file "$TMPDIR"/datasets/CLEVR_colorized/CLEVR_scenes.json\
	--base_scene_blendfile data/data/base_scene.blend\
	--filename_prefix "CLEVR_color"\
	--split ""\
	--min_objects 1\
	--max_objects 4\
	--num_images 5\
	--use_gpu 1\
	--render_num_samples 10000\
	--render_min_bounces 0\
	--render_max_bounces 4\
	--width 270 --height 270\
	--min_dist 0.1\
	--margin 0.2
# ultimately, print output to dev 0... (and set verbosity)

ls "$TMPDIR"/datasets/CLEVR_colorized


# copy checkpoints to home directory
mkdir -p $HOME/City-GAN/datasets/
cp -r "$TMPDIR"/datasets/CLEVR_colorized $HOME/City-GAN/datasets
