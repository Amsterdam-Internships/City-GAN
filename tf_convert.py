import os
import tensorflow as tf
import data.objects_room as room
import argparse
import glob


if __name__ == '__main__':

    flags = argparse.ArgumentParser()

    flags.add_argument('--data_dir', required=True, help="where are the tfrecord files located?")



    flags = flags.parse_args()

    tfrecords = glob.glob(os.path.join(flags.data_dir, "*.tfrecords"))


    for binary in tfrecords:
        # extract filename and dataset variant
        print(f"processing binary: {binary}...")
        dataset_variant = os.path.basename(binary).split(".")[0]


        # create dataset using helper code, and convert to iterator
        dataset = room.dataset(binary, dataset_variant)
        iterator = iter(dataset)

        # make new directory for image files
        image_dir = os.path.join(flags.data_dir, f"images_{dataset_variant}")
        os.makedirs(image_dir, exist_ok=True)

        for i, item in enumerate(iterator):
            image = item['image']
            masks = (item['mask'])
            tf.keras.preprocessing.image.save_img(
                os.path.join(image_dir, f"{i}_img.jpg"), image, scale=True)
            for j, mask in enumerate(masks):
                tf.keras.preprocessing.image.save_img(
                    os.path.join(image_dir, f"{i}_mask_{j}.jpg"), mask, scale=True)

        print(f"Saved all files from {dataset_variant} to {image_dir}\n")
