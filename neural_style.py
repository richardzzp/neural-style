# Copyright (c) 2015-2019 Anish Athalye. Released under GPLv3.

import os
from argparse import ArgumentParser
from collections import OrderedDict

from PIL import Image
import numpy as np
import scipy.misc

from stylize import stylize

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e1
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
POOLING = 'max'
CONTENT_PATH = 'contents'
STYLE_PATH = 'styles'
OUTPUT_PATH = 'outputs'


def main():
    count = 0
    while 1:
        count += 1
        content_path = os.path.join(CONTENT_PATH, str(count) + "-content.jpg")
        style_path = os.path.join(STYLE_PATH, str(count) + "-style.jpg")
        output_path=os.path.join(OUTPUT_PATH,str(count)+"-output.jpg")
        if not os.path.exists(content_path) or not os.path.exists(style_path):
            print("完成所有图片处理")
            return

        if not os.path.isfile(VGG_PATH):
            print("Where is the imagenet-vgg-verydeep-19.mat")
            return

        content_image = imread(content_path)
        style_image = imread(style_path)

        target_shape = content_image.shape
        style_scale = STYLE_SCALE
        style_image = scipy.misc.imresize(style_image, style_scale *
                                          target_shape[1] / style_image.shape[1])

        for image in stylize(
            network=VGG_PATH,
            content=content_image,
            style=style_image,
            iterations=ITERATIONS,
            content_weight=CONTENT_WEIGHT,
            content_weight_blend=CONTENT_WEIGHT_BLEND,
            style_weight=STYLE_WEIGHT,
            style_layer_weight_exp=STYLE_LAYER_WEIGHT_EXP,
            tv_weight=TV_WEIGHT,
            learning_rate=LEARNING_RATE,
            beta1=BETA1,
            beta2=BETA2,
            epsilon=EPSILON,
            pooling=POOLING,
        ):
            continue
        print("第"+str(count)+"张图片完成")
        imsave(output_path, image)


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


if __name__ == '__main__':
    main()
