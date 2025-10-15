from argparse import ArgumentParser, FileType

parser = ArgumentParser()
parser.add_argument("config", type = FileType("rb"))

args = parser.parse_args()

print("Loading imports...")

from alp4 import AlpDataFormat
from complex_color import imshow_complex
import shaping_system_config
import hologram
import numpy as np
import matplotlib.pyplot as plt



def tikhonov_invert(mat, alpha):
    u, s, vh = np.linalg.svd(mat)
    s_inv = np.diag(s/(s**2+alpha**2))
    s_inv = np.pad(s_inv, [(0, vh.shape[0]-len(s)), (0, u.shape[1]-len(s))])

    return vh.conj().T @ s_inv @ u.conj().T

def revival_field(tm1, tm2, method = "field_distance"):
    if tm1.shape != tm2.shape:
        raise ValueError("TMs must be the same shape")

    match method:
        case "field_distance":
            free_dofs = tm1.shape[1]-tm1.shape[0]
            dist_mat = (tm2-tm1).conj().T @ (tm2-tm1)
            power_mat = tm1.conj().T @ tm1

            eigvals, eigvecs = np.linalg.eig(dist_mat)
            idcs = np.argsort(np.abs(eigvals))
            null_mat = eigvecs[:, idcs[:free_dofs]]
            mat = null_mat @ np.linalg.pinv(null_mat) @ power_mat
            eigvals, eigvecs = np.linalg.eig(mat)

            field = eigvecs[:, np.argmax(np.abs(eigvals))]
            field /= np.abs(field).max()

            print("POWER:", (field.conj() @ (power_mat @ field))/(field.conj() @ field))
            print("DIST:", (field.conj() @ (dist_mat @ field))/(field.conj() @ field))

            return field
        case "devival":
            eigvals, eigvecs = np.linalg.eigh(tm2.conj().T @ tm1)
            field = eigvecs[:, 0]
            field /= np.abs(field).max()

            return field


print("Loading config & setting up...")

system = shaping_system_config.from_toml(args.config)

ref = system.measure_reference(200)
colors = np.random.randint(0, 256, (system.segments, 3))

plt.subplot(1, 2, 1)
plt.title("DMD template")
plt.imshow(colors[system.template])
plt.subplot(1, 2, 2)
plt.title("Reference intensity")
plt.imshow(ref, cmap = "gray", vmin = 0)
plt.show()

tm1 = system.measure_tm(ref, progress = True)

np.save("revival_tm1.npy", tm1)

print("Measured 1st TM")
input("Enter anything to measure 2nd TM... ")

tm2 = system.measure_tm(ref, progress = True)

np.save("revival_tm2.npy", tm2)

print("Measured 2nd TM")
print("Computing revival input field...")

field_in = revival_field(tm1, tm2, method = "devival")

imshow_complex(field_in[system.template])
plt.show()

input("Enter anything to capture response (1/2)...")

out1 = system.measure_field(field_in, ref)

input("Enter anything to capture response (2/2)...")

out2 = system.measure_field(field_in, ref)
cor = np.corrcoef([out1.ravel(), out2.ravel()])[0, 1]

print("Correlation:", np.abs(cor))

plt.subplot(1, 2, 1)
imshow_complex(out1)
plt.subplot(1, 2, 2)
imshow_complex(out2)
plt.show()

print("Generating hologram...")

holo = system.hologen.gen_from_template(system.template, field_in)
holo = hologram.pack_bits(holo)

print("Uploading...")

seq = system.dmd.allocate_sequence(1, 1)
seq.set_format(AlpDataFormat.BINARY_TOPDOWN)
seq.put(0, 1, holo)
seq.start(continuous = True)

system.cam.close()

print("Hologram is now displaying")
input("Enter anything to stop... ")
