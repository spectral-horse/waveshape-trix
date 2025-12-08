from argparse import ArgumentParser

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
    # Things linked to the physical setup. The camera+DMD control system, and
    # possibly a sequence handle if we're displaying something on the DMD.
    system: ShapingSystem
    seq: AlpSequence | None = None

    # Data that can be manipulated/used by REPL commands
    ref_img: np.ndarray | None = None
    tms: list[MatrixRecord] = field(default_factory = list)
    inputs: list[InputFieldRecord] = field(default_factory = list)
    outputs: list[np.ndarray] = field(default_factory = list)

    # Miscellaneous flags and such
    quit: bool = False



# Field distance revival operator
def revival_field(tm1, tm2):
    diff = tm2-tm1
    _, _, vh = np.linalg.svd(diff)
    null_dim = tm1.shape[1]-tm1.shape[0]
    null_basis = vh.conj().T[:, -null_dim:]

    _, _, vh = np.linalg.svd(tm1 @ null_basis)
    field = null_basis @ vh.conj()[0]
    field /= np.abs(field).max()

    return field

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



# --- REPL commands ---

def cmd_quit(state, args):
    state.quit = True

def cmd_reference(state, args):
    try: n_imgs = int(args[0])
    except IndexError: n_imgs = 100
    except ValueError: n_imgs = None

    if n_imgs is None or n_imgs <= 0:
        print("Invalid integer")
        return

    if state.seq is not None:
        state.seq.free()
        state.seq = None

    print("Capturing reference intensity image...")

    state.ref_img = state.system.measure_reference(n_imgs)
    h, w = state.ref_img.shape

    print(f"Done ({w}x{h})")

def cmd_matrix(state, args):
    if state.ref_img is None:
        print("Must have a reference image first")
        return

    if state.seq is not None:
        state.seq.free()
        state.seq = None

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
    if len(args) != 1:
        print("Usage: clear matrices/inputs/outputs/all")
    elif args[0] in ["matrices", "mat"]:
        state.tms.clear()
    elif args[0] in ["inputs", "in"]:
        state.inputs.clear()
    elif args[0] in ["outputs", "out"]:
        state.outputs.clear()
    elif args[0] in ["all", "a"]:
        state.tms.clear()
        state.inputs.clear()
        state.outputs.clear()
    else:
        print("Invalid list to clear")

def cmd_show(state, args):
    if len(args) == 0:
        print("Usage: show reference/matrix/input/output (n)")
        return

    if args[0] in ["reference", "ref"]:
        if state.ref_img is None:
            print("No reference image to show")
            return
        
        plt.imshow(state.ref_img, vmin = 0)
        plt.show()
    else:
        try: n = int(args[1])
        except IndexError: n = -1
        except ValueError:
            print("Index must be an integer")
            return

        if args[0] in ["matrix", "mat"]:
            try:
                imshow_complex(state.tms[n].mat)
                plt.show()
            except IndexError:
                print("TM index out of bounds")
        elif args[0] in ["input", "in"]:
            template = state.system.template

            try:
                imshow_complex(state.inputs[n].field[template])
                plt.show()
            except IndexError:
                print("Input field index out of bounds")
        elif args[0] in ["output", "out"]:
            try:
                imshow_complex(state.outputs[n])
                plt.show()
            except IndexError:
                print("Output field index out of bounds")

def cmd_revival(state, args):
    if len(args) == 0: n1, n2 = -2, -1
    elif len(args) == 2:
        try: n1, n2 = map(int, args)
        except ValueError:
            print("Indices must be valid integers")
            return
    else:
        print("Usage: revival (N1 N2)")
        return

    try:
        tm1 = state.tms[n1].mat
        tm2 = state.tms[n2].mat
    except IndexError:
        print("TM index out of bounds")
        return

    record = InputFieldRecord(revival_field(tm1, tm2), "Revival")

    state.inputs.append(record)

def cmd_video(state, args):
    try:
        exposure, duration, out_path = args
    except ValueError:
        print("Usage: video EXPOSURE DURATION OUT_PATH")
        return

    try:
        exposure = float(exposure)
        duration = float(duration)
    except ValueError:
        print("EXPOSURE and DURATION must be numerical")
        return

    if exposure < 0 or duration < 0:
        print("EXPOSURE and DURATION must be positive")
        return

    cam = state.system.cam
    old_exposure = cam.get_exposure()

    cam.set_exposure(exposure)
    cam.set_sync_out(False)

    fps = state.system.cam.get_framerate()
    n_frames = int(duration*fps)

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
        ffmpeg_write(frames, fps, out_path)
    except RuntimeError as e:
        print("Video write failed:")
        print(*e.args)

def cmd_exposure(state, args):
    if len(args) == 0:
        exposure = state.system.cam.get_exposure()

        print("Current exposure is", exposure, "s")
    elif len(args) == 1:
        try: exposure = float(args[0])
        except ValueError:
            print("Exposure time must be a valid number")
            return

        if exposure <= 0:
            print("Exposure time must be a postive number")
            return

        state.system.cam.set_exposure(exposure)
    else:
        print("Usage: exposure [EXPOSURE_SECONDS]")

def cmd_generate(state, args):
    n = state.system.segments
    zs = np.random.normal(0, 1, n)*1j
    zs += np.random.normal(0, 1, n)
    zs /= np.abs(zs).max()

    state.inputs.append(InputFieldRecord(zs, "Generated"))

def cmd_apply(state, args):
    if len(args) != 1:
        print("Usage: apply N")
        return

    try:
        input_field = state.inputs[int(args[0])].field
    except ValueError:
        print("N must be an integer")
        return
    except IndexError:
        print("Input field index out of bounds")
        return

    template = state.system.template
    hologen = state.system.hologen
    holo = hologen.gen_from_template(template, input_field)
    holo = np.packbits(holo, axis = -1)

    print("Uploading...")

    if state.seq is not None:
        state.seq.free()

    state.seq = system.dmd.allocate_sequence(1, 1)
    state.seq.set_format(AlpDataFormat.BINARY_TOPDOWN)
    state.seq.put(0, 1, holo)
    state.seq.start(continuous = True)

def cmd_measure(state, args):
    if state.ref_img is None:
        print("Must have a reference image first")
        return

    if len(args) != 2:
        print("Usage: measure N REDUCE?")
        return

    try:
        input_field = state.inputs[int(args[0])].field
    except ValueError:
        print("N must be an integer")
        return
    except IndexError:
        print("Input field index out of bounds")
        return

    match args[1]:
        case "yes": reduce = True
        case "no": reduce = False
        case _:
            print("REDUCE? must be 'yes' or 'no'")
            return

    output_field = state.system.measure_field(
        input_field, state.ref_img, reduce = reduce
    )

    state.outputs.append(output_field)

def cmd_save(state, args):
    if len(args) != 2:
        print("Usage: save N PATH")
        return

    try: n = int(args[0])
    except ValueError:
        print("Index must be an integer")
        return

    try: np.save(args[1], state.tms[n].mat)
    except IndexError:
        print("Index out of range")
        return
    except Exception as e:
        print("Failed to save array to file:")
        print(e)

# --- REPL commands end ---



print("Loading config & setting up...")

system = shaping_system_config.from_toml(args.config)
state = State(system = system)

cmds = [
    (["quit", "q"], cmd_quit),
    (["reference", "ref"], cmd_reference),
    (["matrix", "mat"], cmd_matrix),
    (["list", "ls", "l"], cmd_list),
    (["clear", "c"], cmd_clear),
    (["show", "s"], cmd_show),
    (["revival", "rev"], cmd_revival),
    (["video", "v"], cmd_video),
    (["generate", "gen"], cmd_generate),
    (["exposure", "exp", "ex"], cmd_exposure),
    (["apply", "a"], cmd_apply),
    (["measure", "meas"], cmd_measure),
    (["save", "sav"], cmd_save)
]

print("Welcome!")

while not state.quit:
    while len(parts := shlex.split(input("> "))) == 0:
        pass

    cmd, *args = parts

    for aliases, func in cmds:
        if cmd in aliases:
            func(state, args)
            break
    else:
        print("Unknown command")

system.cam.close()
system.dmd.close()
