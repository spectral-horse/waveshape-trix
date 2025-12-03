from alp4 import Alp, AlpDataFormat, AlpTrigger
import hologram
import numpy as np
import time



# Generating the inverse matrix relating a vector of intensity measurements
# to its solution vector [A^2+B^2, 2AB cos(phi), 2AB sin(phi)]
def gen_phase_mat(shifts, n):
    phases = np.array(shifts)*2*np.pi/n
    phase_mat = np.linalg.pinv(np.c_[
        np.ones(len(shifts)), np.cos(phases), -np.sin(phases)
    ])

    return phase_mat

# Create a DMD sequence and initialise it for projection with the given data.
#     dmd : An open instance of alp.AlpDevice
#     data : Array of byte images to upload to the DMD, with shape (n, h, w)
#     data_fmt : Binary format of the data (as supported in alp module)
#     rate : Display rate in images per second
#     cycle_limit : Stop after this many cycles if given, otherwise continuous
# Returns the alp.AlpSequence object, ready to be run.
def make_seq(dmd, data, data_fmt, rate, cycle_limit = None):
    seq = dmd.allocate_sequence(1, data.shape[0])
    seq.set_format(data_fmt)
    seq.put(0, data.shape[0], data)
    seq.set_timing(picture = int(0.5+1_000_000/rate))

    if cycle_limit is not None:
        seq.set_cycles(cycle_limit)

    return seq

# Convert an Aravis buffer to an ndarray, optionally cropping it.
#     buf : An instance of Aravis.Buffer containing image data
#     crop : Optional sequence of the form (x, y, w, h) specifying a region
# Returns a 2D array of pixel values.
def buffer_to_ndarray(buf, crop = None):
    w, h = buf.get_image_width(), buf.get_image_height()
    img = np.frombuffer(buf.get_data(), dtype = "u1").reshape(h, w)

    if crop is not None:
        xi, yi, wi, hi = crop
        img = img[yi:yi+hi, xi:xi+wi]

    return img

# Extract the complex field from a set of n intensity images, each w by h,
# using the given phase retrieval matrix and reference intensity image.
#     imgs : 3D array of shape (n, h, w) 
#     ref : 2D array of shape (h, w)
#     phase_mat : Inverse phase shifting matrix of shape (3, n)
# Returns a 2D complex array of shape (h, w) holding the field.
def extract_z(imgs, ref, phase_mat):
    # Flattened images have shape (n, h*w), solution has shape (3, h, w)
    imgs_flat = imgs.reshape(imgs.shape[0], -1)
    sol = (phase_mat @ imgs_flat).reshape(3, *imgs.shape[1:])

    # Assuming no background in addition to reference
    a1 = np.hypot(sol[1], sol[2])/(2*np.sqrt(ref))
    a2 = np.sqrt(np.maximum(0, sol[0]-ref))
    z = np.empty(imgs.shape[1:], dtype = "c16")
    z.real = sol[1]
    z.imag = sol[2]
    z *= (a1+a2)/(2*np.abs(z))

    # Assuming a non-interfering background in addition to reference
    #a = np.sqrt(np.maximum(0, sol[0]-ref))
    #z = np.empty(imgs.shape[1:], dtype = "c16")
    #z.real = sol[1]
    #z.imag = sol[2]
    #z *= a/np.abs(z)

    return z

# Generate the square Hadamard matrix of the given size, with entries -1 or 1
def hadamard_mat(size):
    xx, yy = np.meshgrid(*2*(np.arange(size),))
    
    return 1-2.0*(np.bitwise_count(xx & yy)%2)

# Generate a "spinner" string, which has the form of an O walking back and
# forth along a line with time.
#     width: Width of the spinner
#     dt: Time interval for each step of the O
# For example, width = 5, dt = 0.5 gives the following:
#     t = 0   : "(--O)"
#     t = 0.5 : "(-O-)"
#     t = 1   : "(O--)"
#     t = 1.5 : "(-O-)"
#     t = 2   : "(--O)"
def spinner(width, dt):
    steps = 2*width-6
    i = abs(int(time.time()/dt)%steps-steps//2)

    return "("+("-"*i)+"O"+("-"*(width-3-i))+")"



class ShapingSystem:
    # Parameters are as follows.
    #     camera:   Instance of a CameraBackend controlling the camera
    #     segments: Number of segments to divide the DMD surface into
    #     dmd_fps:  Framerate of the DMD (should be > camera FPS, to keep up)
    #     hologen:  Hologram generator
    #     shifts:   Integer pixel shifts for phase stepping
    #     mask:     Optional 2D boolean array to select output pixels
    def __init__(
        self, camera, segments, dmd_fps, hologen, shifts,
        mask = None
    ):
        if np.log2(segments)%1 != 0:
            raise ValueError("number of segments must be a power of 2")

        alp = Alp()
        dmd = alp.open_device()
        dmd_size = dmd.get_display_size()

        dmd.set_trigger(AlpTrigger.NONE)

        self.dmd = dmd
        self.dmd_fps = dmd_fps
        self.cam = camera
        self.roi = camera.get_roi()
        self.template = hologram.make_template_grid(dmd_size[::-1], segments)
        self.hologen = hologen
        self.shifts = np.array(shifts)
        self.mask = mask
        self.mask_sum = mask.sum()

        # Will be an AlpSequence handle to a Hadamard pattern sequence when a
        # TM has just been taken, to allow for re-use of this large sequence.
        self.hadamard_seq = None

    @property
    def segments(self):
        return int(self.template.max()+1)

    @property
    def output_shape(self):
        if self.mask is None: return self.roi[2:]
        else: return (self.mask_sum,)

    @property
    def output_size(self):
        if self.mask is None: return self.roi[2]*self.roi[3]
        else: return self.mask_sum

    def _apply_mask(self, imgs):
        if self.mask is None: return imgs
        else: return imgs[..., self.mask]

    # Measure the reference beam intensity with no pattern on the DMD.
    #     n_images - Number of shots to take
    # Returns the mean intensity image over the specified number of shots. This
    # will NOT be reduced using the system's mask.
    def measure_reference(self, n_images):
        self.cam.set_sync_out(False)
        self.cam.start_acquisition(n_images)

        while self.cam.is_acquiring():
            time.sleep(0.1)

        frames = self.cam.stop_acquisition()

        return frames.mean(axis = 0)

    # Measure a transmission matrix.
    #     ref: Full-size (unreduced/unmasked) reference intensity image
    #     progress: If True, print progress messages
    # Returns a complex 2D array of shape (w*h, segments) holding the TM.
    def measure_tm(self, ref, progress = False):
        n = self.hologen.n
        dmd = self.dmd
        dmd_fps = self.dmd_fps
        segments = self.segments
        shifts = self.shifts
        n_frames = len(shifts)*segments

        if self.hadamard_seq is None:
            hadamard = self.hologen.hadamard_template(self.template, shifts)
            hadamard = np.packbits(hadamard, axis = -1)

            if progress: print("Uploading patterns...")

            self.hadamard_seq = make_seq(
                dmd, hadamard, AlpDataFormat.BINARY_TOPDOWN, dmd_fps, 1
            )
        
        if progress: print("Measuring...")

        self.cam.set_sync_out(True)
        self.cam.start_acquisition(n_frames)
        dmd.set_trigger(AlpTrigger.FALLING)
        self.hadamard_seq.start()

        while self.cam.is_acquiring():
            time.sleep(0.1)

            if progress: 
                print("\r"+spinner(7, 0.5), end = "", flush = True)

        if progress: print("\nDone")

        dmd.halt()
        dmd.set_trigger(AlpTrigger.NONE)

        frames = self.cam.stop_acquisition()
        tm = np.empty((self.output_size, segments), dtype = "c16")
        phase_mat = gen_phase_mat(shifts, n)

        for i in range(segments):
            frame_start = i*len(shifts)
            frame_end = (i+1)*len(shifts)
            field = extract_z(frames[frame_start:frame_end], ref, phase_mat)
            field = self._apply_mask(field)

            tm[:, i] = field.ravel()

        # Correct for the fact that shift indices are all offset by an extra
        # phase factor, then do the change of basis back from Hadamard
        tm *= np.exp(-1j*np.pi*(np.floor(n*n/2)-1)/(n*n))
        tm = tm @ np.linalg.inv(hadamard_mat(segments))

        return tm

    # Measure a single complex-valued field in response to a single vector of
    # complex inputs.
    #     zs :    1D array of complex input values, one per template group
    #     ref:    Full-size (unreduced/unmasked) reference intensity image
    #     reduce: Whether to reduce the output with the system's mask
    # Returns a complex 2D array of shape (w*h, segments) holding the TM.
    def measure_field(self, zs, ref, reduce = True):
        if self.hadamard_seq is not None:
            self.hadamard_seq.free()
            self.hadamard_seq = None

        dmd = self.dmd
        dmd_fps = self.dmd_fps
        shifts = self.shifts
        n = self.hologen.n

        phases = np.array(shifts)*2*np.pi/n
        phase_mat = gen_phase_mat(shifts, n)

        zs = np.exp(1j*phases)[:, None]*zs
        holo = self.hologen.gen_from_template(self.template, zs)
        holo = hologram.pack_bits(holo)

        seq = make_seq(dmd, holo, AlpDataFormat.BINARY_TOPDOWN, dmd_fps, 1)

        self.cam.set_sync_out(True)
        self.cam.start_acquisition(len(shifts))
        dmd.set_trigger(AlpTrigger.FALLING)
        seq.start()

        while self.cam.is_acquiring():
            time.sleep(0.1)

        dmd.halt()
        dmd.set_trigger(AlpTrigger.NONE)
        seq.free()

        frames = self.cam.stop_acquisition()
        field = extract_z(frames, ref, phase_mat)

        if reduce: field = self._apply_mask(field)

        return field

    def measure_intensity(self, zs, reduce = True):
        if self.hadamard_seq is not None:
            self.hadamard_seq.free()
            self.hadamard_seq = None

        dmd = self.dmd
        dmd_fps = self.dmd_fps

        holo = self.hologen.gen_from_template(self.template, zs)
        holo = hologram.pack_bits(holo)[None, :, :]

        seq = make_seq(dmd, holo, AlpDataFormat.BINARY_TOPDOWN, dmd_fps, 1)

        self.cam.set_sync_out(True)
        self.cam.start_acquisition(1)
        dmd.set_trigger(AlpTrigger.FALLING)
        seq.start()

        while self.cam.is_acquiring():
            time.sleep(0.1)

        dmd.halt()
        dmd.set_trigger(AlpTrigger.NONE)
        seq.free()

        frames = self.cam.stop_acquisition()

        if reduce: frames = self._apply_mask(frames)

        return frames[0]
