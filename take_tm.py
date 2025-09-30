from argparse import ArgumentParser, FileType

parser = ArgumentParser()
parser.add_argument("config", type = FileType("rb"))
parser.add_argument("-o", "--output", type = FileType("wb"))

args = parser.parse_args()

print("Loading imports...")

import shaping_system_config
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from complex_color import imshow_complex



# Calculate the Tikhonov-regularised inverse of a matrix by SVD
def tikhonov_invert(mat, alpha):
    u, s, vh = np.linalg.svd(mat, full_matrices = False)
    s_inv = np.diag(s/(s**2+alpha**2))

    return vh.conj().T @ s_inv @ u.conj().T



print("Loading config & setting up...")

system = shaping_system_config.from_toml(args.config)

print("Measuring reference intensity...")

ref = system.measure_reference(200)
colors = np.random.randint(0, 256, (system.segments, 3))

zs1 = np.exp(2j*np.pi*np.random.uniform(0, 1, system.segments))
zs2 = zs1*np.exp(2j*np.pi/system.hologen.n)
speckle1 = system.measure_field(zs1, ref)
speckle2 = system.measure_field(zs2, ref)

plt.subplot(2, 2, 1)
plt.title("DMD template")
plt.imshow(colors[system.template])
plt.subplot(2, 2, 2)
plt.title("Reference intensity")
plt.imshow(ref, cmap = "gray", vmin = 0)
plt.subplot(2, 2, 3)
plt.title("Random speckle")
imshow_complex(speckle1)
plt.subplot(2, 2, 4)
plt.title("Random speckle shifted back")
imshow_complex(speckle2*np.exp(-2j*np.pi/system.hologen.n))
plt.show()

tm = system.measure_tm(ref, progress = True)

if args.output is not None:
    np.save(args.output, tm)

target = (np.array(Image.open("Targets/duck_20x20.png"))[:, :, 0] > 0)*1

test_zs = tikhonov_invert(tm, 10) @ target.ravel()
test_zs /= np.abs(test_zs).max()
test_zs = system.hologen.get_nearest(test_zs)

test_out = system.measure_field(test_zs, ref)

plt.subplot(2, 3, 1)
plt.title("Target output")
plt.imshow(target)
plt.subplot(2, 3, 2)
plt.title("Expected output")
imshow_complex((tm @ test_zs).reshape(system.output_shape))
plt.subplot(2, 3, 4)
plt.title("Actual output")
imshow_complex(test_out)
plt.subplot(2, 3, 5)
plt.title("Actual output magnitude")
plt.imshow(np.abs(test_out))
plt.subplot(2, 3, 6)
plt.title("Actual output phase")
plt.imshow(np.angle(test_out)%(2*np.pi))
plt.show()
