import gi

gi.require_version("GLib", "2.0")

from gi.repository import GLib as glib
from shaping_system import ShapingSystem
from alp import AlpError
import hologram
import numpy as np
import matplotlib.pyplot as plt



def tikhonov_invert(mat, alpha):
    u, s, vh = np.linalg.svd(mat)
    s_inv = np.diag(s/(s**2+alpha**2))
    s_inv = np.pad(s_inv, [(0, vh.shape[0]-len(s)), (0, u.shape[1]-len(s))])

    return vh.conj().T @ s_inv @ u.conj().T

def revival_field(tm1, tm2):
    if tm1.shape != tm2.shape:
        raise ValueError("TMs must be the same shape")

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



h = hologram.SuperpixelGenerator(4)
shifts = [0, 1, 2, 3] # phase shifts in pixels

try:
    system = ShapingSystem(
        1024,             # Input DOFs on the DMD
        h,                # Hologram generator
        (0, 0, 128, 128), # Region of interest of the camera
        (48, 48, 20, 20), # Sub-region of interest within camera ROI
        100,              # Camera FPS
        0.01              # Camera exposure in seconds
    )
except AlpError as e:
    print("Couldn't open DMD!")
    print(*e.args)

    exit(1)
except glib.Error as e:
    print("Couldn't initialise camera!")
    print(e.message)

    exit(1)

ref = system.measure_reference(200)
colors = np.random.randint(0, 256, (system.segments, 3))

plt.subplot(1, 2, 1)
plt.title("DMD template")
plt.imshow(colors[system.template])
plt.subplot(1, 2, 2)
plt.title("Reference intensity")
plt.imshow(ref, cmap = "gray", vmin = 0, vmax = 255)
plt.show()

tm1 = system.measure_tm(ref, shifts, progress = True)

np.save("revival_tm1.npy", tm1)

print("Measured 1st TM")
input("Enter anything to measure 2nd TM... ")

tm2 = system.measure_tm(ref, shifts, progress = True)

np.save("revival_tm2.npy", tm2)

print("Measured 2nd TM")
print("Computing revival input field...")

#field_in = revival_field(tm1, tm2)

# Maximum sensitivity recipe
#mat = tm2.conj().T @ tm1
#eigvals, eigvecs = np.linalg.eig(mat)
#idcs = np.argsort(np.abs(eigvals))
#free_dofs = tm1.shape[1]-tm1.shape[0]
#eigvecs = eigvecs[:, idcs[:free_dofs]]
#eigvecs /= np.abs(eigvecs).max(axis = 0)
#powers = (np.abs(tm1 @ eigvecs)**2).sum(axis = 0)
#field_in = eigvecs[:, np.argmax(powers)]

#plt.plot(np.sort(np.abs(eigvals)))
#plt.plot([0, free_dofs], [0, 0])
#plt.show()

# Popoff-Wigner-Smith operator
mat = tikhonov_invert(tm1, 1) @ (tm2-tm1)
mat += mat.conj().T
eigvals, eigvecs = np.linalg.eig(mat)
field_in = eigvecs[:, np.argmin(np.abs(eigvals))]
field_in /= np.abs(field_in).max()

print("Generating hologram...")

holo = system.hologen.gen_from_template(system.template, field_in)
holo = hologram.pack_bits(holo)

print("Uploading...")

seq = system.dmd.allocate_sequence(1, 1)
seq.set_format("binary_topdown")
seq.put(0, 1, holo)
seq.start(continuous = True)

del system.cam

print("Hologram is now displaying")
input("Enter anything to stop... ")
