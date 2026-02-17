from dcam_backend import DcamBackend
from dmd_backend import DmdBackend
from shaping_system import ShapingSystem
from pydantic import BaseModel, NonNegativeInt, conlist
from enum import Enum
from typing import Optional
from PIL import Image
import numpy as np
import tomllib
import hologram
import os



class HologramType(str, Enum):
    HASKELL = "haskell"
    SUPERPIXEL = "superpixel"

class TmProbeType(str, Enum):
    HADAMARD = "hadamard"
    RANDOM_PHASE = "random_phase"

class DmdConfig(BaseModel):
    segments: int
    hologram_type: HologramType
    hologram_order: NonNegativeInt

class CameraConfig(BaseModel):
    roi: conlist(NonNegativeInt, min_length = 4, max_length = 4)
    exposure: float
    mask: Optional[str] = None

class InterferometryConfig(BaseModel):
    shifts: list[float]
    tm_probes: TmProbeType

class Config(BaseModel):
    dmd: DmdConfig
    camera: CameraConfig
    interferometry: InterferometryConfig



def from_toml(path):
    config_data = tomllib.load(open(path, "rb"))
    config = Config.model_validate(config_data)

    match config.dmd.hologram_type:
        case "haskell":
            h = hologram.HaskellGenerator(config.dmd.hologram_order)
        case "superpixel":
            h = hologram.SuperpixelGenerator(config.dmd.hologram_order)

    if config.camera.mask is not None:
        mask_path = os.path.join(os.path.dirname(path), config.camera.mask)
        mask = np.array(Image.open(mask_path))
        
        match mask.ndim:
            case 2: mask = mask.astype(bool)
            case 3: mask = mask[:, :, :3].mean(axis = 2).astype(bool)
            case _: raise ValueError("mask image must be 2D")
    else:
        mask = None

    shifts = np.deg2rad(config.interferometry.shifts)

    print("Opening camera...")

    camera = DcamBackend(config.camera.roi, config.camera.exposure)

    print("Opening DMD...")

    shaper = DmdBackend(config.dmd.segments, 1000, h)

    print("Initialising system...")

    system = ShapingSystem(camera, shaper, shifts, mask)

    return system
