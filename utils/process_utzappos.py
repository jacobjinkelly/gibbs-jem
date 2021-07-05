"""
One-time processing for utzappos.

Reorganize the UT-Zappos dataset to resemble the MIT-States dataset
root/attr_obj/img1.jpg
root/attr_obj/img2.jpg
root/attr_obj/img3.jpg
...
"""

import os
import torch
import shutil

root = os.path.join("/Users/jacobkelly/Downloads")
data_root = os.path.join(root, "utzappos")
os.makedirs(os.path.join(data_root, "images"))

data = torch.load(os.path.join(root, "attr-ops-data", "data", "ut-zap50k", "metadata.t7"))
for instance in data:
    image, attr, obj = instance['_image'], instance['attr'], instance['obj']

    old_file = os.path.join(data_root, "_images", image)

    if not os.path.exists(old_file):
        *dir_names, last_dir_name, img_path = image.split("/")
        last_dir_name = last_dir_name + "%2E"
        new_old_image_name = "/".join(dir_names + [last_dir_name, img_path])
        old_file = os.path.join(data_root, "_images", new_old_image_name)
        assert os.path.exists(old_file)

    new_dir = os.path.join(data_root, "images", f"{attr}_{obj}")
    os.makedirs(new_dir, exist_ok=True)
    shutil.copy(old_file, new_dir)
