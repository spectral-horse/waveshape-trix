from abc import ABC, abstractmethod



class CameraBackend(ABC):
    @abstractmethod
    def start_acquisition(self, n: int):
        # Begin acquiring a sequence of n frames. This method should set the
        # camera running and then return without blocking on the acquisition.
        pass

    @abstractmethod
    def stop_acquisition(self) -> "ndarray":
        # Stop acquisition and return the acquired frames as a 3D array of
        # shape (n, height, width). This method should block until the camera
        # has fully stopped and should stop it forcefully even if it has not
        # gathered the number of frames originally requested.
        pass

    @abstractmethod
    def is_acquiring(self) -> bool:
        # Test whether the camera is currently acquiring, so that we can do
        # things like wait until it has finished acquiring before retrieving
        # the frame buffer.
        pass

    @abstractmethod
    def set_sync_out(self, enabled: bool):
        # Enable or disable the sync out signal of the camera, which should be
        # the signal that goes high when exposure of a frame completes.
        pass

    @abstractmethod
    def get_roi(self) -> (int, int, int, int):
        # Get the region of interest of the camera as a tuple of the form
        # (x, y, width, height). The width and height must match the width and
        # height of the frames returned by `stop_aquisition`.
        pass

    @abstractmethod
    def get_id_string(self) -> str:
        # Get an arbitrary identification string for this camera. This can be
        # absolutely anything, but ideally a model name and/or serial number.
        pass
