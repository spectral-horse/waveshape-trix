from dcam_backend import DcamBackend
from shaping_system import ShapingSystem
from alp4 import AlpError
from pydantic import BaseModel, NonNegativeInt, conlist
from enum import Enum
from typing import Optional
from PIL import Image
import numpy as np
import tomllib
import hologram



class DmdConfig(BaseModel):
    segments: int

class HologramType(str, Enum):
    HASKELL = "haskell"
    SUPERPIXEL = "superpixel"

class HologramConfig(BaseModel):
    type: HologramType
    order: NonNegativeInt
    shifts: list[NonNegativeInt]

class ReducerType(str, Enum):
    BLOCK = "block"
    SKIP = "skip"

class CameraConfig(BaseModel):
    roi: conlist(NonNegativeInt, min_length = 4, max_length = 4)
    exposure: float
    mask: Optional[str] = None

class Config(BaseModel):
    dmd: DmdConfig
    hologram: HologramConfig
    camera: CameraConfig



def from_toml(path):
    config_data = tomllib.load(path)
    config = Config.model_validate(config_data)

    match config.hologram.type:
        case "haskell": h = hologram.HaskellGenerator(config.hologram.order)
        case "superpixel": h = hologram.SuperpixelGenerator(config.hologram.order)

    if config.camera.mask is not None:
        mask = np.array(Image.open(config.camera.mask))
        
        match mask.ndim:
            case 2: mask = mask.astype(bool)
            case 3: mask = mask[:, :, :3].mean(axis = 2).astype(bool)
            case _: raise ValueError("mask image must be 2D")
    else:
        mask = None

    print("Opening camera...")

    camera = DcamBackend(config.camera.roi, config.camera.exposure)

    print("Initialising system...")

    try:
        system = ShapingSystem(
            camera,
            config.dmd.segments,    # Input DOFs on the DMD
            1000,                   # DMD FPS (>> camera FPS)
            h,                      # Hologram generator
            config.hologram.shifts, # Phase stepping shifts
            mask
        )
    except AlpError as e:
        print("Couldn't open DMD!")
        print(*e.args)

        exit(1)

    return system
