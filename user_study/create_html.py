# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2021-06-15 15:34
# @Last Modified by:   TomLotze
# @Last Modified time: 2021-06-16 12:52

import sys
if ".." not in sys.path:
    sys.path.append("..")
from util.html import HTML
import random
import os


page = HTML('.', "User study for MoveGAN on ROOM Objects dataset")

id2class = {
    0 : "real",
    1 : "move",
    2 : "random",
    3 : "scanline"
    }


possible_tuples = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 3),
    (3, 0),
    (3, 1),
    (3, 2)
    ]

# generate list with 300 tuples to determine which pairs is used
all_pairs = (5 * possible_tuples)
random.shuffle(all_pairs)

page.add_comment(f"{all_pairs}")
page.add_header("Please choose the most realistic image from each pair. If you don't have a preference, please note \"3\"")
page.add_header(f"You can enter your preferences in the following Google Sheet:")

page.add_link("https://docs.google.com/spreadsheets/d/1pstBEeDHH3JeLsVgMi4RM1yciVF1GX3084QmNp86i6c/edit?usp=sharing")

def id_to_path(class_id, sample_nr):
    category = id2class[class_id]
    return os.path.join(category, f"{category}_{sample_nr}.png")


# 60 pairs are to be added
for i, (left_id, right_id) in enumerate(all_pairs, start=1):
    page.add_header(f"Pair {i}")

    imgs = [id_to_path(left_id, i), id_to_path(right_id, i)]
    page.add_images(imgs, ["1", "2"], imgs)

    # json file maken met pairs, welke is welke
print(all_pairs)

page.save()







