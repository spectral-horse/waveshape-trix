from dcam import Dcam, DcamError
from dcam.enums import DcamPixelType, DcamPolarity, DcamSource, DcamOutputKind
from camera_backend import CameraBackend
import numpy as np



# Return the number of channels and channel data type for a given pixel format
def pixel_type_info(pixel_type):
    match pixel_type:
        case DcamPixelType.MONO_8: return 1, np.uint8
        case DcamPixelType.MONO_16: return 1, np.uint16
        case DcamPixelType.MONO_12: return 1, np.uint16
        case DcamPixelType.RGB_24: return 3, np.uint8
        case DcamPixelType.RGB_48: return 3, np.uint16
        case DcamPixelType.BGR_24: return 3, np.uint8
        case DcamPixelType.BGR_48: return 3, np.uint16
        case _:
            raise RuntimeError(f"Unsupported pixel type {pixel_type}")

class DcamBackend(CameraBackend):
    def __init__(self, roi, exposure):
        self.dcam = Dcam("C:\\Windows\\System32\\dcamapi.dll")

        try:
            self.dev = self.dcam.open_device(0)
        except DcamError as e:
            self.dcam.close()
            raise e

        self.dev.enable_roi(*roi)
        self.dev.set_property("EXPOSURE TIME", exposure)

        self.frame_buffer = None

    def __enter__(self):
        return self

    def __exit__(self, e_type, e_value, e_trace):
        self.close()

    def close(self):
        self.dcam.close()

    def start_acquisition(self, n):
        frame_bytes = self.dev.get_property("BUFFER FRAMEBYTES").value
        width = self.dev.get_property("IMAGE WIDTH").value
        height = self.dev.get_property("IMAGE HEIGHT").value
        pixel_type = self.dev.get_pixel_type()
        channels, channel_dtype = pixel_type_info(pixel_type)
        pixel_bytes = channels*np.dtype(channel_dtype).itemsize

        if width*height*pixel_bytes != frame_bytes:
            raise RuntimeError("BUFFER FRAMEBYTES not equal to calculated size")

        if channels == 1:
            buf = np.empty((n, width, height), dtype = channel_dtype)
        else:
            buf = np.empty((n, width, height, channels), dtype = channel_dtype)

        buf_addr = buf.__array_interface__["data"][0]

        self.dev.attach_frames(n, buf_addr, frame_bytes)
        self.dev.start_capture()

        self.frame_buffer = buf

    def stop_acquisition(self):
        self.dev.stop_capture()
        self.dev.free_frames()

        buf = self.frame_buffer
        self.frame_buffer = None

        return buf

    def is_acquiring(self):
        return self.dev.get_capturing()

    def set_sync_out(self, enabled):
        if enabled:
            self.dev.set_output_trigger_period(0, 0.001)
            self.dev.set_output_trigger_delay(0, 0)
            self.dev.set_output_trigger_polarity(0, DcamPolarity.POSITIVE)
            self.dev.set_output_trigger_source(0, DcamSource.READOUT_END)
            self.dev.set_output_trigger_kind(0, DcamOutputKind.PROGRAMMABLE)
        else:
            self.dev.set_output_trigger_kind(0, DcamOutputKind.LOW)
