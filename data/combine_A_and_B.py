import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool
import re


def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, 1)
    im_B = cv2.imread(path_B, 1)
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)


parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', type=str, default='/data/UG/Jiya/pix2pix/datasets/NIRVCIP_DATASET/trainNIR')
parser.add_argument('--fold_B', type=str, default='/data/UG/Jiya/pix2pix/datasets/NIRVCIP_DATASET/trainRGB')
parser.add_argument('--fold_AB', type=str, default='/data/UG/Jiya/pix2pix/datasets/NIRVCIP_DATASET/train')
parser.add_argument('--num_imgs', type=int, default=1000000)
parser.add_argument('--no_multiprocessing', action='store_true', default=False)
args = parser.parse_args()


nir_files = sorted(os.listdir(args.fold_A))

if not args.no_multiprocessing:
    pool = Pool()

os.makedirs(args.fold_AB, exist_ok=True)

count = 0
for name_A in nir_files:

    path_A = os.path.join(args.fold_A, name_A)
    if not os.path.isfile(path_A):
        continue

    # build RGB filename by replacing the suffix
    if "_nir" not in name_A:
        print("Skipping (no _nir):", name_A)
        continue

    name_B = name_A.replace("_nir", "_rgb_reg")
    path_B = os.path.join(args.fold_B, name_B)

    if not os.path.isfile(path_B):
        print("Missing RGB for:", name_A)
        continue

    # keep the original naming or use ID-only output
    out_name = name_A.replace("_nir", "")
    path_AB = os.path.join(args.fold_AB, out_name)

    if not args.no_multiprocessing:
        pool.apply_async(image_write, args=(path_A, path_B, path_AB))
    else:
        im_A = cv2.imread(path_A, 1)
        im_B = cv2.imread(path_B, 1)
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(path_AB, im_AB)

    count += 1
    if count >= args.num_imgs:
        break

if not args.no_multiprocessing:
    pool.close()
    pool.join()

print("Completed:", count, "pairs created.")
