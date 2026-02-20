import hologram
import numpy as np
import time



# Generating the inverse matrix relating a vector of intensity measurements
# to its solution vector [A^2+B^2, 2AB cos(phi), 2AB sin(phi)]
def gen_phase_mat(shifts):
    phase_mat = np.linalg.pinv(np.c_[
        np.ones(len(shifts)), np.cos(shifts), -np.sin(shifts)
    ])

    return phase_mat

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

# Extract the complex field(s) from a set of n intensity images, each w by h,
# using the given phase retrieval matrix and reference intensity image. The
# phase matrix solves for a number of shifts, s, which n must be a multiple of.
#     imgs : 3D array of shape (n, h, w) 
#     ref : 2D array of shape (h, w)
#     phase_mat : Inverse phase shifting matrix of shape (3, s)
# Returns a complex array of shape (h, w) if only one field is extracted (n = s)
# or of shape (n/s, h, w) if many are extracted.
def extract_z(imgs, ref, phase_mat):
    s = phase_mat.shape[1]
    n, *img_shape = imgs.shape
    img_size = np.prod(img_shape)
    sol = phase_mat @ imgs.reshape(n//s, s, img_size)
    ref = ref.ravel()

    # Assuming no background in addition to reference
    a1 = np.hypot(sol[:, 1, :], sol[:, 2, :])/(2*np.sqrt(ref))
    a2 = np.sqrt(np.maximum(0, sol[:, 0, :]-ref))
    z = np.empty((n//s, img_size), dtype = "c16")
    z.real = sol[:, 1, :]
    z.imag = sol[:, 2, :]
    z *= (a1+a2)/(2*np.abs(z))

    if z.shape[0] == 1: z.shape = img_shape
    else: z.shape = (n//s, *img_shape)

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
    # Parameters are as follows:
    #     camera: Instance of a CameraBackend controlling the camera
    #     shaper: Instance of a ShaperBackend controlling some SLM/DMD/etc
    #     shifts: Phase shifts for interferometric stepping
    #     mask:   Optional 2D boolean array to select output pixels
    def __init__(self, camera, shaper, shifts, mask = None):
        if mask is not None and mask.dtype != bool:
            raise ValueError("mask must be a 2D boolean array")

        self.shaper = shaper
        self.cam = camera
        self.roi = camera.get_roi()
        self.shifts = np.array(shifts)
        self.mask = mask
        self.mask_sum = mask.sum()
        self.just_measured_tm = False

    @property
    def input_size(self):
        return self.shaper.dofs()

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

    def _upload_tm_patterns(self):
        self.shaper.free_patterns()

        mat = hadamard_mat(self.input_size)
        zs = mat[:, None, :]*np.exp(1j*self.shifts)[:, None]
        zs.shape = (-1, mat.shape[1])

        self.shaper.upload_patterns(zs)

    # Measure a transmission matrix.
    #     ref: Full-size (unreduced/unmasked) reference intensity image
    #     progress: If True, print progress messages
    # Returns a complex 2D array of shape (w*h, segments) holding the TM.
    def measure_tm(self, ref, progress = False):
        if not self.just_measured_tm:
            if progress: print("Uploading probe patterns...")

            self._upload_tm_patterns()
            self.just_measured_tm = True

        if progress: print("Measuring...")

        n_frames = len(self.shifts)*self.input_size

        self.cam.set_sync_out(True)
        self.cam.start_acquisition(n_frames)
        self.shaper.set_sync_in(True)
        self.shaper.start()

        while self.cam.is_acquiring():
            time.sleep(0.1)

            if progress: 
                print("\r"+spinner(7, 0.5), end = "", flush = True)

        if progress: print("\nDone")

        self.shaper.stop()
        self.shaper.set_sync_in(False)

        frames = self.cam.stop_acquisition()
        frames = self._apply_mask(frames)
        ref = self._apply_mask(ref)
        phase_mat = gen_phase_mat(self.shifts)
        fields = extract_z(frames, ref, phase_mat)
        tm = fields.reshape(self.input_size, self.output_size).T

        # Correct for the fact that shift indices are all offset by an extra
        # phase factor, then do the change of basis back from Hadamard
        #tm *= np.exp(-1j*np.pi*(np.floor(n*n/2)-1)/(n*n))

        tm = tm @ np.linalg.inv(hadamard_mat(self.input_size))

        return tm

    # Measure a single complex-valued field in response to a single vector of
    # complex inputs.
    #     zs :    1D array of complex input values, one per template group
    #     ref:    Full-size (unreduced/unmasked) reference intensity image
    #     reduce: Whether to reduce the output with the system's mask
    # Returns a complex 2D array of shape (w*h, segments) holding the TM.
    def measure_field(self, zs, ref, reduce = True):
        if self.just_measured_tm:
            self.shaper.free_patterns()
            self.just_measured_tm = False

        zs = np.exp(1j*self.shifts)[:, None]*zs

        self.shaper.upload_patterns(zs)
        self.cam.set_sync_out(True)
        self.cam.start_acquisition(len(self.shifts))
        self.shaper.set_sync_in(True)
        self.shaper.start()

        while self.cam.is_acquiring():
            time.sleep(0.1)

        self.shaper.free_patterns()
        self.shaper.set_sync_in(False)

        frames = self.cam.stop_acquisition()
        phase_mat = gen_phase_mat(self.shifts)
        field = extract_z(frames, ref, phase_mat)

        if reduce: field = self._apply_mask(field)

        return field

    def measure_intensity(self, zs, reduce = True):
        if self.just_measured_tm:
            self.shaper.free_patterns()
            self.just_measured_tm = False

        self.shaper.upload_patterns(zs[None, :])
        self.cam.set_sync_out(True)
        self.cam.start_acquisition(1)
        self.shaper.set_sync_in(True)
        self.shaper.start()

        while self.cam.is_acquiring():
            time.sleep(0.1)

        self.shaper.free_patterns()
        self.shaper.set_sync_in(False)

        frames = self.cam.stop_acquisition()

        if reduce: frames = self._apply_mask(frames)

        return frames[0]
