
# these are the images with the right classes
used_imgs

imgToAnns
catToImgs

out_ans = {}


for id_ in used_imgs:
    anns = imgToAnns[id_]
    new_anns = []
    for ann in anns:
        if ann['category_id'] in used_cats and ann['area'] > 100:
            new_anns.append(ann)
    if new_anns:
        out_anns[id_] = new_anns

# moving the images
for root, _, files in os.walk("/Users/TomLotze/Downloads/val2017/"):
      for img_name in files:
          if not img_name.endswith(".jpg"):
              continue
          id_ = int(img_name.split("." )[0])
          if str(id_) in img_2000:
              shutil.copy(os.path.join(root, img_name), "datasets/COCO/images/complete/")
