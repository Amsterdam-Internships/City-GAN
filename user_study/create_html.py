# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2021-06-15 15:34
# @Last Modified by:   TomLotze
# @Last Modified time: 2021-06-17 15:23

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

win_size = 200

def id_to_path(class_id, sample_nr):
    category = id2class[class_id]
    return os.path.join(category, f"{category}_{sample_nr}.png")

# generate list with 300 tuples to determine which pairs is used
all_pairs = (5 * possible_tuples)
random.shuffle(all_pairs)

page.add_comment(f"{all_pairs}")
page.add_header("Please choose the most realistic image from each pair. If you don't have a preference, please note \"3\"")
page.add_header(f"The questionnaire consist of 60 pairs and should take bout 5 minites (i.e. ~5 sec per pair).")
page.add_header(f"You can enter your preferences in the Google Sheet below")
page.add_header(f"If possible, please time your answers, and mention your total time in the Google Sheet")


page.add_link("https://docs.google.com/spreadsheets/d/1pstBEeDHH3JeLsVgMi4RM1yciVF1GX3084QmNp86i6c/edit?usp=sharing")

page.add_header("Here are some real images to give you an impression:")

imgs = [id_to_path(0, 100), id_to_path(0, 101), id_to_path(0, 102), id_to_path(0, 103)]
page.add_images(imgs, ["", "", "", ""], imgs, width=win_size)
imgs = [id_to_path(0, 104), id_to_path(0, 105), id_to_path(0, 106), id_to_path(0, 107)]
page.add_images(imgs, ["", "", "", ""], imgs, width=win_size)

page.add_header("All pairs below are numbered, please indicate the most realistic one in the google sheet, thanks!")




# 60 pairs are to be added
for i, (left_id, right_id) in enumerate(all_pairs, start=1):
    page.add_header(f"Pair {i}")

    imgs = [id_to_path(left_id, i), id_to_path(right_id, i)]
    page.add_images(imgs, ["1", "2"], imgs, width=win_size)

    # json file maken met pairs, welke is welke
print(all_pairs)

page.save()







