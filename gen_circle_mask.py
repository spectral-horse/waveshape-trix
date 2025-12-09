from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("width", type = int)
parser.add_argument("height", type = int)
parser.add_argument("center_x", type = float)
parser.add_argument("center_y", type = float)
parser.add_argument("radius", type = float)
parser.add_argument("n", type = int)
parser.add_argument("out_path")

args = parser.parse_args()

import numpy as np
from PIL import Image



img = np.zeros((args.height, args.width), dtype = "u1")
ns = np.arange(args.n)+1
angles = 4*np.pi*ns/(3+np.sqrt(5))
radii = np.sqrt(ns)*args.radius/np.sqrt(args.n)

for angle, radius in zip(angles, radii):
    x = int(args.center_x+radius*np.cos(angle))
    y = int(args.center_y+radius*np.sin(angle))

    img[y, x] = 255

Image.fromarray(img).save(args.out_path)
