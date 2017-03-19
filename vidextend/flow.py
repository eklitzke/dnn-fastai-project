import os.path
import re

import numpy as np
from PIL import Image

NUM_RE = re.compile(r'(\d+)')

maxint = 999999


WHITE_LIST_FORMATS = {'png', 'jpg', 'jpeg', 'bmp'}


def hstack_images(input_filenames, target_size=(224, 224)):
    """
    Horizontally stack all images from @input_filenames in order and write to @output_filename
    """
    images = list(map(lambda i: i.resize(target_size), map(Image.open, input_filenames)))
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


def should_include_image(path, start_num, end_num):
    """Returns true if an image path should be included in the set, false
    otherwise.
    """
    fname = path.lower()

    for extension in WHITE_LIST_FORMATS:
        if fname.endswith('.' + extension):
            num_match = NUM_RE.search(fname)
            if num_match:
                num, = num_match.groups()
                num = int(num, 10)
                return start_num <= num <= end_num
    return False


def flow_from_directory(directory, a, b, c, target_size=(224, 224)):
    for dirpath, dirnames, fnames in os.walk(directory):
        if len(dirnames) == 0:
            # we are at a top-level directory, extract the images in our range
            xs = []
            ys = []

            for fname in fnames:
                if should_include_image(fname, a, b):
                    xs.append(os.path.join(dirpath, fname))
                elif should_include_image(fname, b + 1, c):
                    ys.append(os.path.join(dirpath, fname))
            xs.sort()
            ys.sort()
            x_imgs = np.asarray(hstack_images(xs, target_size=target_size))
            y_imgs = np.asarray(hstack_images(ys, target_size=target_size))
            yield (x_imgs, y_imgs)
