# --- Imports ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import fluidfoam as fl

# If you really need LaTeX rendering, keep this; otherwise it can slow/complicate things
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'Times New Roman',
    'text.usetex': True
})

# --- Paths (fix Path vs path, and use os.path.join) ---
path = r"D:\blueCFD-Core\2024\ofuser-of12\run\3_bird_90_deg_1_sep"
save_path = os.path.join(path, "pod_test")
os.makedirs(save_path, exist_ok=True)

times_txt = os.path.join(path, "times.txt")
if not os.path.isfile(times_txt):
    raise FileNotFoundError(f"Could not find {times_txt}")

# Load times (strip blanks/comments)
Times = [t.strip() for t in open(times_txt).read().splitlines() if t.strip() and not t.strip().startswith('#')]
Snapshots = len(Times)
print("Snapshots:", Snapshots)
if Snapshots == 0:
    raise ValueError("No times found in times.txt")

# --- Read one velocity to get sizes (case-independent) ---
vel = fl.readvector(path, time_name='latestTime', name='U', structured=False)
# vel is typically shape (3, n_cells)
if vel.ndim != 2 or vel.shape[0] != 3:
    raise ValueError(f"Unexpected vel shape {vel.shape}. Expected (3, n_cells) for structured=False.")

n_comp = vel.shape[0]         # 3
n_cells = vel.shape[1]        # Number of cells
total  = vel.size             # n_comp * n_cells
print(f"n_cells={n_cells}, components={n_comp}, total entries={total}")

# --- Snapshot matrix B: flattened (n_cells*3) by Snapshots ---
B = np.zeros((total, Snapshots), dtype=np.float64)

# --- Mean field: prefer UMean if present, else compute temporal mean from data ---
try:
    Mean_vel1 = fl.readvector(path, time_name=str(Times[0]), name='UMean', structured=False)
    if Mean_vel1.shape != vel.shape:
        raise ValueError("UMean shape mismatch with U.")
    mean_flat = Mean_vel1.T.reshape(total, order='F')
    mean_source = "UMean"
except Exception:
    # Compute mean over snapshots if UMean is unavailable
    acc = np.zeros(total, dtype=np.float64)
    for ti in Times:
        ui = fl.readvector(path, time_name=str(ti), name='U', structured=False)
        if ui.shape != vel.shape:
            raise ValueError(f"U({ti}) shape {ui.shape} mismatches first snapshot {vel.shape}")
        acc += ui.T.reshape(total, order='F')
    mean_flat = acc / Snapshots
    mean_source = "temporal mean of U"
print("Mean field source:", mean_source)

# --- Build fluctuation matrix B ---
for i, ti in enumerate(Times):
    ui = fl.readvector(path, time_name=str(ti), name='U', structured=False)
    u_flat = ui.T.reshape(total, order='F')       # (n_cells*3,)
    B[:, i] = u_flat - mean_flat

# Save B for reuse
np.save(os.path.join(save_path, 'B.npy'), B)
print("B saved at:", os.path.join(save_path, 'B.npy'))

# --- Correlation matrix method (method of snapshots) ---
# C is symmetric PSD, use eigh; divide by number of snapshots for proper scaling
C = (B.T @ B) / Snapshots              # shape (Snapshots, Snapshots)
eigvals, eigvecs = np.linalg.eigh(C)   # eigh since C is symmetric
# Sort descending
idx = np.argsort(eigvals)[::-1]
S = eigvals[idx].real
U_t = eigvecs[:, idx].real             # temporal eigenvectors

# --- POD spatial modes (columns) ---
# Modes = B @ U_t / sqrt(Snapshot scaling * eigenvalue) gives orthonormal spatial modes
# Guard against tiny/zero eigenvalues
eps = 1e-14 * S.max() if S.size else 0.0
sigma = np.sqrt(np.maximum(S, 0.0))
sigma[sigma < eps] = np.inf  # avoid division by tiny numbers
Modes = B @ (U_t / sigma)    # shape (total, Snapshots)

# --- Energy content (fraction per mode) ---
if S.sum() > 0:
    Energy = S / S.sum()
else:
    Energy = S * 0.0
cumEnergy = np.cumsum(Energy)

print("Top 10 energy fractions:", Energy[:10])
print("Cumulative energy (top 10):", cumEnergy[:10])

# --- Normalized amplitudes (time coefficients) ---
# With orthonormal Modes, time coefficients are:
Amp = Modes.T @ B             # shape (Snapshots, Snapshots)
# Often people also project each snapshot onto each mode individually:
# a_k(t_j) = mode_k^T * snapshot_j; which is Amp here.

# --- Optional: per-mode normalization (each spatial mode to unit L2) ---
# (Usually unnecessary if constructed as above, but shown for clarity)
mode_norms = np.linalg.norm(Modes, axis=0)
mode_norms[mode_norms == 0] = 1.0
Modes_normalized = Modes / mode_norms

# --- Save some outputs ---
np.save(os.path.join(save_path, 'eigenvalues.npy'), S)
np.save(os.path.join(save_path, 'energy.npy'), Energy)
np.save(os.path.join(save_path, 'modes.npy'), Modes)  # (total, Snapshots)
np.save(os.path.join(save_path, 'amps.npy'), Amp)     # (Snapshots, Snapshots)

# --- Write first 10 modes back in your original mesh ordering (n_cells x 3) ---
header_path = os.path.join(path, 'header.txt')
footer_path = os.path.join(path, 'footer.txt')
if not (os.path.isfile(header_path) and os.path.isfile(footer_path)):
    print("Warning: header.txt/footer.txt not found; skipping OpenFOAM-style saves.")
else:
    # Read the entire files (not just the first line!)
    with open(header_path, 'r', encoding='utf-8', newline='') as fh:
        header = fh.read()
    with open(footer_path, 'r', encoding='utf-8', newline='') as ff:
        footer = ff.read()

    saveTime = os.path.join(path, "pod_test_time")
    os.makedirs(saveTime, exist_ok=True)

    n_write = min(10, Modes.shape[1])
    for i in range(n_write):
        mode_i = Modes[:, i].real.reshape((n_cells, 3), order='F')
        np.savetxt(
            os.path.join(saveTime, f"Mode{i+1}"),
            mode_i,
            fmt='(% .8e % .8e % .8e)',
            header=header,            # full header content
            footer=footer,            # full footer content
            comments=''               # keep OpenFOAM files clean (no leading '#')
        )
    print(f"Wrote {n_write} modes to:", saveTime)


# --- Quick plots (optional) ---
plt.figure()
plt.plot(np.arange(1, len(Energy)+1), Energy, marker='o')
plt.xlabel('Mode index')
plt.ylabel('Energy fraction')
plt.title('POD Energy Spectrum')
plt.grid(True)

plt.figure()
plt.plot(np.arange(1, len(cumEnergy)+1), cumEnergy, marker='o')
plt.xlabel('Mode index')
plt.ylabel('Cumulative energy')
plt.title('Cumulative POD Energy')
plt.grid(True)
plt.show()
