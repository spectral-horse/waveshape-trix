from argparse import ArgumentParser, FileType

parser = ArgumentParser()
parser.add_argument("config", type = FileType("rb"))

args = parser.parse_args()

from shaping_system import ShapingSystem
from dcam_backend import DcamBackend
from alp4 import AlpError
from PIL import Image
import hologram
import numpy as np
import matplotlib.pyplot as plt
from complex_color import imshow_complex
from pydantic import BaseModel, NonNegativeInt, conlist
from enum import Enum
import tomllib



class DmdConfig(BaseModel):
    segments: int

class HologramType(str, Enum):
    HASKELL = "haskell"
    SUPERPIXEL = "superpixel"

class HologramConfig(BaseModel):
    type: HologramType
    order: NonNegativeInt
    shifts: list[NonNegativeInt]

class CameraConfig(BaseModel):
    roi: conlist(NonNegativeInt, min_length = 4, max_length = 4)
    exposure: float

class Config(BaseModel):
    dmd: DmdConfig
    hologram: HologramConfig
    camera: CameraConfig



# Calculate the Tikhonov-regularised inverse of a matrix by SVD
def tikhonov_invert(mat, alpha):
    u, s, vh = np.linalg.svd(mat, full_matrices = False)
    s_inv = np.diag(s/(s**2+alpha**2))

    return vh.conj().T @ s_inv @ u.conj().T



config_data = tomllib.load(args.config)
config = Config.model_validate(config_data)

match config.hologram.type:
    case "haskell": h = hologram.HaskellGenerator(config.hologram.order)
    case "superpixel": h = hologram.SuperpixelGenerator(config.hologram.order)

shifts = config.hologram.shifts
camera = DcamBackend(config.camera.roi, config.camera.exposure)

try:
    system = ShapingSystem(
        camera,
        config.dmd.segments,   # Input DOFs on the DMD
        1000,                  # DMD FPS (>> camera FPS)
        h                      # Hologram generator
    )
except AlpError as e:
    print("Couldn't open DMD!")
    print(*e.args)

    exit(1)

ref = system.measure_reference(200)
colors = np.random.randint(0, 256, (system.segments, 3))

zs1 = np.exp(2j*np.pi*np.random.uniform(0, 1, system.segments))
zs2 = zs1*np.exp(2j*np.pi/system.hologen.n)
speckle1 = system.measure_field(zs1, ref, shifts)
speckle2 = system.measure_field(zs2, ref, shifts)

plt.subplot(2, 2, 1)
plt.title("DMD template")
plt.imshow(colors[system.template])
plt.subplot(2, 2, 2)
plt.title("Reference intensity")
plt.imshow(ref, cmap = "gray", vmin = 0, vmax = 255)
plt.subplot(2, 2, 3)
plt.title("Random speckle")
imshow_complex(speckle1)
plt.subplot(2, 2, 4)
plt.title("Random speckle shifted back")
imshow_complex(speckle2*np.exp(-2j*np.pi/system.hologen.n))
plt.show()

tm = system.measure_tm(ref, shifts, progress = True)

target = np.zeros(system.output_shape)
#target[8:12, 8:12] = 1
#target[8, 8] = 1
target = (np.array(Image.open("Targets/duck_20x20.png"))[:, :, 0] > 0)*1

#test_zs = np.linalg.lstsq(tm, target.ravel())[0]
test_zs = tikhonov_invert(tm, 2) @ target.ravel()
test_zs /= np.abs(test_zs).max()
test_zs = system.hologen.get_nearest(test_zs)

test_out = system.measure_field(test_zs, ref, shifts)

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
