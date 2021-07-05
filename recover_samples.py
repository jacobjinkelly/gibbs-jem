"""
Get images from saved grid of samples.
"""

import argparse
import math
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image


def get_grid(imgs, img_size, nmaps, nrow):
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))

    h, w = img_size, img_size
    padding = 2
    height, width = int(h + padding), int(w + padding)

    s_imgs = []
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break

            s_imgs.append(imgs.narrow(1,
                                      y * height + padding, height - padding).narrow(
                2, x * width + padding, width - padding)[None])
            k = k + 1

    return torch.cat(s_imgs, dim=0)


def get_grid_from_fn(fn, img_size, nmaps, nrow):
    img = Image.open(fn)

    img = TF.to_tensor(img)

    return get_grid(img, img_size, nmaps, nrow)


def main():
    parser = argparse.ArgumentParser("Zero Shot Learning yeet")
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--fn_base", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--num_ch", type=int, default=3)
    parser.add_argument("--nrow", type=int, default=-1)
    parser.add_argument("--orig_nrow", type=int, default=10)
    parser.add_argument("--orig_n_img", type=int, default=100)
    parser.add_argument("--n_crop", type=int, default=5)

    args = parser.parse_args()

    if args.nrow < 0:
        args.nrow = args.n_crop

    fn = os.path.join(args.dir, args.fn_base)

    grid = get_grid_from_fn(fn, args.img_size, args.orig_n_img, args.orig_nrow)

    new_cond = grid[:args.n_crop].view(args.n_crop, args.num_ch, args.img_size, args.img_size)

    save_image(new_cond, os.path.join(args.dir, f"crop_{args.fn_base}"), normalize=False, nrow=args.nrow)


if __name__ == "__main__":
    main()
