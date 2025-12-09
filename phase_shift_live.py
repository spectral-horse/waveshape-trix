from argparse import ArgumentParser

def parse_roi(s):
    w, rest = s.split("x")
    w = int(w)

    if "+" in s:
        h, x, y = map(int, rest.split("+"))
    else:
        h, x, y = int(rest), 0, 0

    return x, y, w, h

parser = ArgumentParser()
parser.add_argument("exposure", type = float)
parser.add_argument("roi", type = parse_roi)

args = parser.parse_args()

from dcam import Dcam, DcamError
from dcam.enums import DcamPolarity, DcamSource, DcamOutputKind
from alp4 import Alp, AlpDataFormat, AlpTrigger
from contextlib import ExitStack
from collections import deque
from threading import Thread
import numpy as np
import subprocess as sp
import complex_color



def extract_z(imgs, ref, phase_mat):
    imgs_flat = imgs.reshape(imgs.shape[0], -1)
    sol = (phase_mat @ imgs_flat).reshape(3, *imgs.shape[1:])

    a1 = np.hypot(sol[1], sol[2])/(2*np.sqrt(ref))
    a2 = np.sqrt(np.maximum(0, sol[0]-ref))
    z = np.empty(imgs.shape[1:], dtype = "c16")
    z.real = sol[1]
    z.imag = sol[2]
    z *= (a1+a2)/(2*np.abs(z))

    return z

def run_ffplay(width, height, pixel_format):
    cmd = [
        "ffplay",
        "-v", "error",
        "-video_size", f"{width}x{height}",
        "-pixel_format", pixel_format,
        "-f", "rawvideo",
        "-i", "-"
    ]

    return sp.Popen(cmd, stdin = sp.PIPE)

def init_dmd_sequence(dmd, k, phases, fps):
    w, h = dmd.get_display_size()
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    imgs = np.zeros((len(phases)+1, h, w), dtype = "u1")

    for i, phase in enumerate(phases):
        imgs[i] = (k[0]*xx+k[1]*yy-phase/(2*np.pi))%1 < 0.5

    seq = dmd.allocate_sequence(1, imgs.shape[0])
    seq.set_format(AlpDataFormat.BINARY_TOPDOWN)
    seq.put(0, imgs.shape[0], np.packbits(imgs, axis = -1))
    seq.set_timing(picture = int(0.5+1_000_000/fps))

    return seq

def camera_thread_fn(camera, queue, n_frames, running):
    for i in range(np.iinfo(int).max):
        camera.wait_frame_ready()

        if i%5 == 4:
            start_idx = ((i//5)*5)%n_frames
            queue.append(start_idx)

        if not running[0]: break

def display_thread_fn(ffplay, queue, buffer, phase_mat, running):
    while running[0]:
        if len(queue) > 0:
            start_idx = queue[0]
            chunk = buffer[start_idx:start_idx+4]
            ref = buffer[start_idx+4]
            z = extract_z(chunk, ref, phase_mat)
            rgb = complex_color.complex_to_rgb(z)

            ffplay.stdin.write(rgb.tobytes())

            if ffplay.poll() is not None:
                print("Error occured in ffplay - exiting")
                break

            queue.clear()

dmd_wavevector = [1/4, 1/16]
dmd_fps = 1000
phases = np.arange(4)*np.pi/2
phase_mat = np.linalg.pinv(np.c_[np.ones(4), np.cos(phases), np.sin(phases)])
n_frames = 100

assert n_frames%5 == 0

with ExitStack() as stack:
    alp = stack.enter_context(Alp())
    dmd = stack.enter_context(alp.open_device())
    seq = init_dmd_sequence(dmd, dmd_wavevector, phases, dmd_fps)
    seq = stack.enter_context(seq)

    dmd.set_trigger(AlpTrigger.NONE)

    print("Initialised DMD")

    dcam = stack.enter_context(Dcam("C:\\Windows\\System32\\dcamapi.dll"))

    width, height = args.roi[2:]
    buffer = np.empty((n_frames, height, width), dtype = "u2")
    buffer_addr = buffer.__array_interface__["data"][0]

    camera = dcam.open_device(0)
    camera.enable_roi(*args.roi)
    camera.set_property("EXPOSURE TIME", args.exposure)
    camera.attach_frames(n_frames, buffer_addr, width*height*2)
    camera.set_output_trigger_period(0, 0.001)
    camera.set_output_trigger_delay(0, 0)
    camera.set_output_trigger_polarity(0, DcamPolarity.NEGATIVE)
    camera.set_output_trigger_source(0, DcamSource.READOUT_END)
    camera.set_output_trigger_kind(0, DcamOutputKind.PROGRAMMABLE)

    print("Initialised camera")

    ffplay = stack.enter_context(run_ffplay(*args.roi[2:], "rgb24"))
    queue = deque()
    running = [True]

    dmd.set_trigger(AlpTrigger.FALLING)
    seq.start(continuous = True)
    camera.start_capture(circular = True)

    camera_thread = Thread(
        target = camera_thread_fn,
        args = (camera, queue, n_frames, running)
    )

    display_thread = Thread(
        target = display_thread_fn,
        args = (ffplay, queue, buffer, phase_mat, running)
    )

    camera_thread.start()
    display_thread.start()

    input("Enter anything to stop...")

    running[0] = False

    camera_thread.join()
    display_thread.join()

    camera.stop_capture()
    camera.wait_capture_stop()
    camera.free_frames()
    camera.close()

    ffplay.kill()
