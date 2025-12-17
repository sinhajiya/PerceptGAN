import os
import numpy as np
from collections import defaultdict

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
from util.visualizer import save_images

import torch
from PIL import Image


# ----------------------------
# Stitcher utility
# ----------------------------
class ImageStitcher:
    def __init__(self, image_size=256):
        self.image_size = image_size
        self.canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
        self.count = np.zeros((image_size, image_size, 1), dtype=np.float32)

    def add_patch(self, patch, x, y):
        h, w = patch.shape[:2]
        self.canvas[y:y+h, x:x+w] += patch
        self.count[y:y+h, x:x+w] += 1

    def get_image(self):
        return np.clip(self.canvas / np.maximum(self.count, 1), 0, 1)


# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    opt = TestOptions().parse()

    # Hard-coded test parameters (same as original)
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()

    # Website
    web_dir = os.path.join(
        opt.results_dir,
        opt.name,
        f'{opt.phase}_{opt.epoch}'
    )
    if opt.load_iter > 0:
        web_dir = f'{web_dir}_iter{opt.load_iter}'

    print('creating web directory', web_dir)
    webpage = html.HTML(
        web_dir,
        f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}'
    )

    # Stitchers per image
    stitchers = {}
    processed_images = set()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)
        model.test()

        visuals = model.get_current_visuals()
        fake = visuals['fake_B']  # (1,3,H,W)

        fake = fake[0].permute(1, 2, 0).cpu().numpy()
        fake = (fake + 1.0) / 2.0  # [-1,1] → [0,1]

        img_id = data['img_id'][0]
        x = int(data['x'][0])
        y = int(data['y'][0])

        if img_id not in stitchers:
            stitchers[img_id] = ImageStitcher(image_size=opt.load_size)

        stitchers[img_id].add_patch(fake, x, y)

        if i % 50 == 0:
            print(f'processing patch {i:04d} of image {img_id}')

    # ----------------------------
    # Save reconstructed images
    # ----------------------------
    for img_id, stitcher in stitchers.items():
        final_img = stitcher.get_image()
        final_img_uint8 = (final_img * 255).astype(np.uint8)

        save_path = os.path.join(web_dir, img_id)
        Image.fromarray(final_img_uint8).save(save_path)

        processed_images.add(save_path)

    webpage.save()

    print(f'Saved {len(processed_images)} reconstructed images.')
