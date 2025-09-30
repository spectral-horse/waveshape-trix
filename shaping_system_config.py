from dcam_backend import DcamBackend
from shaping_system import ShapingSystem, BlockReducer, SkipReducer
from alp4 import AlpError
from pydantic import BaseModel, NonNegativeInt, conlist
from enum import Enum
from typing import Optional
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

class ReducerConfig(BaseModel):
    type: ReducerType
    dx: NonNegativeInt
    dy: NonNegativeInt

class CameraConfig(BaseModel):
    roi: conlist(NonNegativeInt, min_length = 4, max_length = 4)
    exposure: float
    reducer: Optional[ReducerConfig] = None

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

    if config.camera.reducer is not None:
        reducer_dx = config.camera.reducer.dx
        reducer_dy = config.camera.reducer.dy

        match config.camera.reducer.type:
            case "block": reducer = BlockReducer(reducer_dx, reducer_dy)
            case "skip": reducer = SkipReducer(reducer_dx, reducer_dy)
    else:
        reducer = None

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
            reducer
        )
    except AlpError as e:
        print("Couldn't open DMD!")
        print(*e.args)

        exit(1)

    return system
