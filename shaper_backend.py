from abc import ABC, abstractmethod



class ShaperBackend(ABC):
    @abstractmethod
    def dofs(self) -> int:
        # Return the number of complex degrees of freedom this shaper controls
        pass

    @abstractmethod
    def set_sync_in(self, enabled: bool):
        # Enable or disable acting on the signal received from the camera. If
        # enabled, the next input pattern will be projected on the falling edge
        # of that signal.
        pass

    @abstractmethod
    def upload_patterns(self, patterns: "ndarray"):
        # Upload N patterns of complex numbers to be displayed in order when
        # projection is started. The pattern array should have shape (DOFs,) for
        # a single pattern or (N, DOFs) for many, and complex data type.
        pass

    @abstractmethod
    def free_patterns(self):
        # Stop projection if it is occurring, and free whatever patterns might
        # currently be loaded.
        pass

    @abstractmethod
    def start(self, continuous = False):
        # Start projecting individual patterns in order, driven either by the
        # internal framerate or by the sync in signal. If set to continuous,
        # the projection will loop.
        pass

    @abstractmethod
    def stop(self):
        # Stop projection if it is occurring.
        pass

    @abstractmethod
    def close(self):
        # Free all connections and resources
        pass
