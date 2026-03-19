from argparse import ArgumentParser, ArgumentTypeError, ArgumentError

parser = ArgumentParser()
parser.add_argument("config")

args = parser.parse_args()

print("Loading imports...")

import shaping_system_config
import numpy as np
import matplotlib.pyplot as plt
import shlex
import time
import subprocess as sp
from alp4 import AlpSequence, AlpDataFormat
from shaping_system import ShapingSystem
from complex_color import imshow_complex
from dataclasses import dataclass, field
from datetime import datetime



@dataclass
class MatrixRecord:
    mat: np.ndarray
    time: datetime

@dataclass
class InputFieldRecord:
    field: np.ndarray
    description: str

@dataclass
class State:
    system: ShapingSystem

    # Data that can be manipulated/used by REPL commands
    ref_img: np.ndarray | None = None
    tms: list[MatrixRecord] = field(default_factory = list)
    inputs: list[InputFieldRecord] = field(default_factory = list)
    outputs: list[np.ndarray] = field(default_factory = list)

    # Miscellaneous flags and such
    pattern_applied: bool = False
    quit: bool = False



# Field distance revival operator
def revival_field(*tms):
    tms = np.array(tms)
    diffs = (tms[1:]-tms[0]).reshape(-1, tms[0].shape[1])
    _, _, vh = np.linalg.svd(diffs)
    null_dim = diffs.shape[1]-diffs.shape[0]

    if null_dim <= 0:
        n = len(tms)

        raise ValueError(f"{n}-TM revival requires cols > {n-1}*rows")

    null_basis = vh.conj().T[:, -null_dim:]
    _, _, vh = np.linalg.svd(tms[0] @ null_basis)
    field = null_basis @ vh.conj()[0]
    field /= np.abs(field).max()

    return field

# Save a video using ffmpeg as a subprocess
def ffmpeg_write(frames, fps, out_path):
    if frames.ndim != 3:
        raise RuntimeError("Unsupported frame buffer shape")

    match frames.dtype:
        case np.uint8: pix_fmt = "gray"
        case np.uint16: pix_fmt = "gray16le"
        case _:
            raise RuntimeError("Unsupported data type")

    cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-video_size", "{}x{}".format(*frames.shape[1:]),
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-qp", "0",
        out_path,
        "-y"
    ]

    proc = sp.run(cmd, input = frames.tobytes(), capture_output = True)

    if proc.returncode != 0:
        raise RuntimeError("Failed to save video: "+proc.stderr.decode())

# Utility for argparse arguments
def positive_int(x):
    x = int(x)

    if x <= 0: raise ArgumentTypeError(f"{x} is not a positive integer")
    else: return x

# Utility for argparse arguments
def positive_float(x):
    x = float(x)

    if x <= 0: raise ArgumentTypeError(f"{x} is not a positive float")
    else: return x



# --- REPL commands ---

def cmd_quit(state, args):
    state.quit = True

def cmd_reference(state, args):
    if args.action == "autoreference":
        if args.enabled == "on":
            state.system.self_referencing = True
        elif args.enabled == "off":
            state.system.self_referencing = False
        elif args.enabled == "toggle":
            state.system.self_referencing = not state.system.self_referencing
            enabled = "on" if state.system.self_referencing else "off"

            print("Toggled self-referencing", enabled)
        
        return
    elif args.action == "take":
        if state.pattern_applied:
            state.system.shaper.free_patterns()
            state.pattern_applied = False

        print("Capturing reference intensity image...")

        state.ref_img = state.system.measure_reference(args.n)
        h, w = state.ref_img.shape

        print(f"Done ({w}x{h})")

def cmd_matrix(state, args):
    if state.ref_img is None:
        print("Must have a reference image first")
        return

    if state.pattern_applied:
        state.system.shaper.free_patterns()
        state.pattern_applied = False

    tm = state.system.measure_tm(state.ref_img, progress = True)
    time = datetime.now()
    
    state.tms.append(MatrixRecord(tm, time))

def cmd_list(state, args):
    print(len(state.tms), "transmission matrice(s):")

    for i, record in enumerate(state.tms):
        time_str = record.time.strftime("%H:%M:%S")
        shape = record.mat.shape

        print("  {}: {}x{} {}".format(i, *shape, time_str))

    print(len(state.inputs), "input field(s):")

    for i, inp in enumerate(state.inputs):
        print("  {}: {}".format(i, inp.description))

    print(len(state.outputs), "output field(s):")

    for i, output in enumerate(state.outputs):
        if len(output.shape) == 2:
            print("  {}: {}x{}".format(i, *output.shape[::-1]))
        elif len(output.shape) == 1:
            print("  {}: {}".format(i, *output.shape))

def cmd_clear(state, args):
    if args.matrices: state.tms.clear()
    if args.outputs: state.outputs.clear()
    if args.inputs: state.inputs.clear()

    if not (args.matrices or args.outputs or args.inputs):
        state.tms.clear()
        state.outputs.clear()
        state.inputs.clear()

def cmd_show(state, args):
    try:
        if args.object == "reference":
            if state.ref_img is None:
                print("No reference image to show")
                return
            
            plt.imshow(state.ref_img, vmin = 0)
        elif args.object == "matrix":
            imshow_complex(state.tms[args.n].mat)
        elif args.object == "input":
            imshow_complex(state.inputs[args.n].field[None, :])
        elif args.object == "output":
            imshow_complex(state.outputs[args.n])
        elif args.object == "mask":
            if state.system.mask is None:
                print("No mask set")
                return

            plt.scatter(*np.nonzero(state.system.mask))
            plt.xlim(0, state.system.roi[2])
            plt.ylim(0, state.system.roi[3])

        plt.show()
    except IndexError:
        print("Index out of bounds")

def cmd_revival(state, args):
    try:
        ns = [args.n0]+args.n
        tms = [ state.tms[n].mat for n in ns ]
    except IndexError:
        print("TM index out of bounds")
        return

    record = InputFieldRecord(revival_field(*tms), "Revival")

    state.inputs.append(record)

def cmd_video(state, args):
    cam = state.system.cam
    old_exposure = cam.get_exposure()

    cam.set_exposure(args.exposure)
    cam.set_sync_out(False)

    fps = cam.get_framerate()
    n_frames = int(args.duration*fps)

    cam.start_acquisition(n_frames)

    while cam.is_acquiring():
        spin_char = r"\|/-"[int(time.time()*2)%4]

        print("Acquiring", spin_char, end = "\r", flush = True)
        time.sleep(0.1)

    frames = cam.stop_acquisition()

    cam.set_exposure(old_exposure)

    print("Acquired frame buffer:")
    print("  Shape:", frames.shape)
    print("  Data type:", frames.dtype)

    try:
        ffmpeg_write(frames, fps, args.out_path)
    except RuntimeError as e:
        print("Video write failed:")
        print(*e.args)

def cmd_exposure(state, args):
    if args.exposure is None:
        exposure = state.system.cam.get_exposure()

        print("Current exposure is", exposure, "s")
    else:
        state.system.cam.set_exposure(args.exposure)

def cmd_generate(state, args):
    n = state.system.input_size

    if args.type == "gaussian":
        zs = np.random.normal(0, 1, n)*1j
        zs += np.random.normal(0, 1, n)
        zs /= np.abs(zs).max()

        name = "Generated (gaussian)"

        state.inputs.append(InputFieldRecord(zs, name))
    elif args.type == "ones":
        name = "Generated (ones)"

        state.inputs.append(InputFieldRecord(np.ones(n), name))
    elif args.type == "focus":
        try:
            tm = state.tms[args.matrix_n].mat
        except IndexError:
            print("Matrix index out of bounds")
            return

        try:
            target = np.zeros(state.system.output_size)
            target[args.pixel_index] = 1
        except IndexError:
            print("Pixel index out of bounds")
            return

        field = np.linalg.lstsq(tm, target)[0]
        field /= np.abs(field).max()
        name = f"Generated (focus at output pixel {args.pixel_index})"

        state.inputs.append(InputFieldRecord(field, name))

def cmd_apply(state, args):
    try:
        input_field = state.inputs[args.n].field
    except IndexError:
        print("Input field index out of bounds")
        return

    print("Uploading...")

    state.system.shaper.free_patterns()
    state.system.shaper.upload_patterns(input_field)
    state.system.shaper.start(continuous = True)

    state.pattern_applied = True

def cmd_measure(state, args):
    if not args.intensity and state.ref_img is None:
        print("Must have a reference image first")
        return

    try:
        input_field = state.inputs[int(args.n)].field
    except IndexError:
        print("Input field index out of bounds")
        return

    if state.pattern_applied:
        state.system.shaper.free_patterns()
        state.pattern_applied = False

    if args.intensity:
        output_field = state.system.measure_intensity(
            input_field, reduce = args.masked
        )
    else:
        output_field = state.system.measure_field(
            input_field, state.ref_img, reduce = args.masked
        )

    state.outputs.append(output_field)

def cmd_save(state, args):
    try:
        if args.object == "reference":
            if state.ref_img is None:
                print("No reference image to save")
                return
            
            plt.imsave(args.out_path, state.ref_img, cmap = "gray", vmin = 0)
        elif args.object == "matrix":
            np.save(args.out_path, state.tms[args.n].mat)
        elif args.object == "input":
            np.save(args.out_path, state.inputs[args.n].field)
        elif args.object == "output":
            np.save(args.out_path, state.outputs[args.n])
        elif args.object == "mask":
            if state.system.mask is None:
                print("No mask set")
                return

            plt.imsave(args.out_path, state.system.mask, cmap = "gray")
    except IndexError:
        print("Index out of bounds")

def cmd_mask(state, args):
    try:
        imgs = [ state.outputs[n] for n in args.imgs ]
    except IndexError:
        print("Index out of bounds")
        return

    if imgs[0].ndim != 2:
        print("Images must be 2D")
        return

    combo = np.full(imgs[0].shape, np.finfo(float).max)

    for img in imgs:
        if img.shape != combo.shape:
            print("Images must have the same shape")
            return

        combo = np.minimum(np.abs(img), combo)

    xx, yy = np.meshgrid(*map(np.arange, combo.shape[::-1]))
    mask = np.zeros(combo.shape, dtype = bool)

    for _ in range(args.n_points):
        pos = np.unravel_index(combo.argmax(), combo.shape)
        mask[pos] = True
        combo[np.hypot(xx-pos[1], yy-pos[0]) < args.radius] = 0

    state.system.mask = mask

# --- REPL commands end ---



print("Loading config & setting up...")

system = shaping_system_config.from_toml(args.config)
state = State(system = system)

parser = ArgumentParser(prog = "", exit_on_error = False)
subparsers = parser.add_subparsers(required = True)

quit_cmd = subparsers.add_parser("quit", aliases = ["q"])
quit_cmd.set_defaults(func = cmd_quit)

ref_cmd = subparsers.add_parser("reference", aliases = ["ref"])
ref_cmd.set_defaults(func = cmd_reference)
ref_subparsers = ref_cmd.add_subparsers(required = True)
ref_take_cmd = ref_subparsers.add_parser("take")
ref_take_cmd.set_defaults(action = "take")
ref_take_cmd.add_argument("n", type = positive_int, default = 100, nargs = "?")
ref_auto_cmd = ref_subparsers.add_parser("autoreference", aliases = ["auto"])
ref_auto_cmd.set_defaults(action = "autoreference")
ref_auto_cmd.add_argument(
    "enabled", default = "toggle", nargs = "?",
    choices = ["on", "off", "toggle"]
)

matrix_cmd = subparsers.add_parser("matrix", aliases = ["mat"])
matrix_cmd.set_defaults(func = cmd_matrix)

list_cmd = subparsers.add_parser("list", aliases = ["ls", "l"])
list_cmd.set_defaults(func = cmd_list)

clear_cmd = subparsers.add_parser("clear")
clear_cmd.set_defaults(func = cmd_clear)
clear_cmd.add_argument("--matrices", "-m", action = "store_true")
clear_cmd.add_argument("--outputs", "-o", action = "store_true")
clear_cmd.add_argument("--inputs", "-i", action = "store_true")

show_cmd = subparsers.add_parser("show")
show_cmd.set_defaults(func = cmd_show)
show_subparsers = show_cmd.add_subparsers(required = True)
show_reference_cmd = show_subparsers.add_parser("reference", aliases = ["ref"])
show_reference_cmd.set_defaults(object = "reference")
show_mask_cmd = show_subparsers.add_parser("mask")
show_mask_cmd.set_defaults(object = "mask")
show_matrix_cmd = show_subparsers.add_parser("matrix", aliases = ["mat"])
show_matrix_cmd.set_defaults(object = "matrix")
show_matrix_cmd.add_argument("n", type = int, default = -1, nargs = "?")
show_input_cmd = show_subparsers.add_parser("input", aliases = ["in"])
show_input_cmd.set_defaults(object = "input")
show_input_cmd.add_argument("n", type = int, default = -1, nargs = "?")
show_output_cmd = show_subparsers.add_parser("output", aliases = ["out"])
show_output_cmd.set_defaults(object = "output")
show_output_cmd.add_argument("n", type = int, default = -1, nargs = "?")

revival_cmd = subparsers.add_parser("revival", aliases = ["rev"])
revival_cmd.set_defaults(func = cmd_revival)
revival_cmd.add_argument("n0", type = int, default = -1, nargs = "?")
revival_cmd.add_argument("n", type = int, default = [-2], nargs = "*")

video_cmd = subparsers.add_parser("video")
video_cmd.set_defaults(func = cmd_video)
video_cmd.add_argument("exposure", type = positive_float)
video_cmd.add_argument("duration", type = positive_float)
video_cmd.add_argument("out_path")

exposure_cmd = subparsers.add_parser("exposure", aliases = ["exp", "ex"])
exposure_cmd.set_defaults(func = cmd_exposure)
exposure_cmd.add_argument("exposure", type = positive_float, nargs = "?")

gen_cmd = subparsers.add_parser("generate", aliases = ["gen"])
gen_cmd.set_defaults(func = cmd_generate)
gen_subparsers = gen_cmd.add_subparsers(required = True)
gen_gaussian_cmd = gen_subparsers.add_parser("gaussian", aliases = ["gauss"])
gen_gaussian_cmd.set_defaults(type = "gaussian")
gen_ones_cmd = gen_subparsers.add_parser("ones", aliases = ["one", "1"])
gen_ones_cmd.set_defaults(type = "ones")
gen_focus_cmd = gen_subparsers.add_parser("focus")
gen_focus_cmd.set_defaults(type = "focus")
gen_focus_cmd.add_argument("pixel_index", type = int)
gen_focus_cmd.add_argument("matrix_n", type = int, default = -1, nargs = "?")

apply_cmd = subparsers.add_parser("apply")
apply_cmd.set_defaults(func = cmd_apply)
apply_cmd.add_argument("n", type = int, default = -1, nargs = "?")

measure_cmd = subparsers.add_parser("measure", aliases = ["meas"])
measure_cmd.set_defaults(func = cmd_measure)
measure_cmd.add_argument("n", type = int, default = -1, nargs = "?")
measure_cmd.add_argument("--intensity", "-i", action = "store_true")
measure_cmd.add_argument("--masked", "-m", action = "store_true")

save_cmd = subparsers.add_parser("save", aliases = ["sav"])
save_cmd.set_defaults(func = cmd_save)
save_cmd.add_argument("out_path")
save_subparsers = save_cmd.add_subparsers(required = True)
save_reference_cmd = save_subparsers.add_parser("reference", aliases = ["ref"])
save_reference_cmd.set_defaults(object = "reference")
save_mask_cmd = save_subparsers.add_parser("mask")
save_mask_cmd.set_defaults(object = "mask")
save_matrix_cmd = save_subparsers.add_parser("matrix", aliases = ["mat"])
save_matrix_cmd.set_defaults(object = "matrix")
save_matrix_cmd.add_argument("n", type = int, default = -1, nargs = "?")
save_input_cmd = save_subparsers.add_parser("input", aliases = ["in"])
save_input_cmd.set_defaults(object = "input")
save_input_cmd.add_argument("n", type = int, default = -1, nargs = "?")
save_output_cmd = save_subparsers.add_parser("output", aliases = ["out"])
save_output_cmd.set_defaults(object = "output")
save_output_cmd.add_argument("n", type = int, default = -1, nargs = "?")

mask_cmd = subparsers.add_parser("mask")
mask_cmd.set_defaults(func = cmd_mask)
mask_subparsers = mask_cmd.add_subparsers(required = True)
mask_common_cmd = mask_subparsers.add_parser("common")
mask_common_cmd.set_defaults(action = "common")
mask_common_cmd.add_argument("n_points", type = positive_int)
mask_common_cmd.add_argument("imgs", type = int, default = [-1], nargs = "+")
mask_common_cmd.add_argument("--radius", "-r", type = float, default = 10.0)



print("Welcome!")

while not state.quit:
    while len(parts := shlex.split(input("> "))) == 0:
        pass

    try:
        args = parser.parse_args(parts)
        args.func(state, args)
    except ArgumentError as e:
        print(e.message)
    except ArgumentTypeError as e:
        print(*e.args)
    except SystemExit:
        pass

system.cam.close()
system.shaper.close()
