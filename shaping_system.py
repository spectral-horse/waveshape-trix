from alp4 import Alp, AlpDataFormat, AlpTrigger
from collections import deque
from skimage.measure import block_reduce
from abc import ABC, abstractmethod
from dataclasses import dataclass
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



class Reducer(ABC):
    @abstractmethod
    def reduce(img: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reduced_shape(img_shape: (int, int)) -> (int, int):
        pass

class NullReducer(Reducer):
    def reduce(self, img):
        return img

    def reduced_shape(self, img_shape):
        return img_shape

@dataclass
class BlockReducer(Reducer):
    def __init__(self, nx, ny):
        self.shape = (ny, nx)

    def reduce(self, img):
        shape = (1,)*(img.ndim-2)+self.shape
        reduced = block_reduce(img, block_size = shape, func = np.mean)
        height = img.shape[-2]//self.shape[0]
        width = img.shape[-1]//self.shape[1]

        # Necessary to slice this because `block_reduce` will include partial
        # blocks at the edges and we want only full blocks
        return reduced[..., :height, :width]

    def reduced_shape(self, img_shape):
        shape_2d = (img_shape[0]//self.shape[0], img_shape[1]//self.shape[1])

        return (1,)*(len(img_shape)-2)+shape_2d

@dataclass
class SkipReducer(Reducer):
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny

    def reduce(self, img):
        if img.ndim == 2: return img[::self.ny, ::self.nx]
        else: return img[:, ::self.ny, ::self.nx]

    def reduced_shape(self, img_shape):
        height = (img_shape[0]+self.ny-1)//self.ny
        width = (img_shape[1]+self.nx-1)//self.nx

        return (1,)*(len(img_shape)-2)+(height, width)

class ShapingSystem:
    def __init__(self, camera, segments, dmd_fps, hologen, reducer = None):
        if np.log2(segments)%1 != 0:
            raise ValueError("number of segments must be a power of 2")

        if reducer is not None and not isinstance(reducer, Reducer):
            raise TypeError("reducer must be a Reducer")

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
        self.reducer = reducer or NullReducer()

    @property
    def segments(self):
        return int(self.template.max()+1)

    @property
    def output_shape(self):
        return self.reducer.reduced_shape(self.roi[2:])

    @property
    def output_size(self):
        h, w = self.output_shape

        return w*h

    # Measure the reference beam intensity with no pattern on the DMD.
    #     n_images - Number of shots to take
    # Returns the mean intensity image over the specified number of shots.
    def measure_reference(self, n_images):
        self.cam.set_sync_out(False)
        self.cam.start_acquisition(n_images)

        while self.cam.is_acquiring():
            time.sleep(0.1)

        frames = self.cam.stop_acquisition()
        frames = self.reducer.reduce(frames)

        return frames.mean(axis = 0)

    # Measure a transmission matrix.
    #     ref: Reference intensity image
    #     shifts: Integer pixel shifts for phase stepping
    #     progress: If True, print progress messages
    # Returns a complex 2D array of shape (w*h, segments) holding the TM.
    def measure_tm(self, ref, shifts, progress = False):
        n = self.hologen.n
        dmd = self.dmd
        dmd_fps = self.dmd_fps
        segments = self.segments
        n_frames = len(shifts)*segments

        phase_mat = gen_phase_mat(shifts, n)
        hadamard = self.hologen.hadamard_template(self.template, shifts)
        hadamard = hologram.pack_bits(hadamard)

        if progress: print("Uploading patterns...")

        seq = make_seq(dmd, hadamard, AlpDataFormat.BINARY_TOPDOWN, dmd_fps, 1)
        
        if progress: print("Measuring...")

        self.cam.set_sync_out(True)
        self.cam.start_acquisition(n_frames)
        dmd.set_trigger(AlpTrigger.FALLING)
        seq.start()

        while self.cam.is_acquiring():
            time.sleep(0.1)

            if progress: 
                print("\r"+spinner(7, 0.5), end = "", flush = True)

        if progress: print("\nDone")

        dmd.set_trigger(AlpTrigger.NONE)

        frames = self.cam.stop_acquisition()
        frames = self.reducer.reduce(frames)

        tm = np.empty((self.output_size, segments), dtype = "c16")

        for i in range(segments):
            frame_start = i*len(shifts)
            frame_end = (i+1)*len(shifts)
            field = extract_z(frames[frame_start:frame_end], ref, phase_mat)

            tm[:, i] = field.ravel()

        seq.free()

        # Correct for the fact that shift indices are all offset by an extra
        # phase factor, then do the change of basis back from Hadamard
        tm *= np.exp(-1j*np.pi*(np.floor(n*n/2)-1)/(n*n))
        tm = tm @ np.linalg.inv(hadamard_mat(segments))

        return tm

    # Measure a single complex-valued field in response to a single vector of
    # complex inputs.
    #     zs : 1D array of complex input values, one per group in the template
    #     shifts: Integer pixel shifts for phase stepping
    # Returns a complex 2D array of shape (w*h, segments) holding the TM.
    def measure_field(self, zs, ref, shifts):
        dmd = self.dmd
        dmd_fps = self.dmd_fps
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

        dmd.set_trigger(AlpTrigger.NONE)

        frames = self.cam.stop_acquisition()
        frames = self.reducer.reduce(frames)
        field = extract_z(frames, ref, phase_mat)

        seq.free()

        return field

    def measure_intensity(self, zs, reduce = True):
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

        dmd.set_trigger(AlpTrigger.NONE)
        seq.free()

        frames = self.cam.stop_acquisition()

        if reduce: frames = self.reducer.reduce(frames)

        return frames[0]
