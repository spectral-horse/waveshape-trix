import hologram
import numpy as np
from alp4 import Alp, AlpDataFormat, AlpTrigger
from shaper_backend import ShaperBackend



class DmdBackend(ShaperBackend):
    def __init__(self, segments, fps, hologen):
        alp = Alp()
        dmd = alp.open_device()
        dmd_size = dmd.get_display_size()
        template = hologram.make_template_grid(dmd_size[::-1], segments)

        dmd.set_trigger(AlpTrigger.NONE)

        self.alp = alp
        self.dmd = dmd
        self.fps = fps
        self.segments = segments
        self.template = template
        self.hologen = hologen

        # Will be an AlpSequence handle to the sequence just used, in case we
        # want to re-use it for efficiency.
        self.seq = None

    def dofs(self):
        return self.segments

    def set_sync_in(self, enabled):
        if enabled:
            self.dmd.set_trigger(AlpTrigger.NONE)
        else:
            self.dmd.set_trigger(AlpTrigger.FALLING)

    def upload_patterns(self, patterns):
        if patterns.ndim == 1:
            patterns = patterns[None, :]
        elif patterns.ndim > 2:
            raise ValueError("patterns array must have shape (D,) or (N, D)")

        holo = self.hologen.gen_from_template(self.template, patterns)
        holo = np.packbits(holo, axis = -1)

        self.free_patterns()

        self.seq = self.dmd.allocate_sequence(1, holo.shape[0])
        self.seq.set_format(AlpDataFormat.BINARY_TOPDOWN)
        self.seq.put(0, holo.shape[0], holo)
        self.seq.set_timing(picture = int(0.5+1_000_000/self.fps))
        self.seq.set_cycles(1)

    def free_patterns(self):
        if self.seq is not None:
            self.dmd.halt()
            self.seq.free()

            self.seq = None

    def start(self, continuous = False):
        if self.seq is None:
            raise RuntimeError("No patterns loaded to the shaper")

        self.dmd.set_trigger(AlpTrigger.FALLING)
        self.seq.start(continuous)

    def stop(self):
        if self.seq is not None:
            self.dmd.halt()

    def close(self):
        self.free_patterns()
        self.dmd.close()
