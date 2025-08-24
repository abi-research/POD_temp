# --- Imports ---
import os
import numpy as np
import matplotlib.pyplot as plt
import fluidfoam as fl

# --- Matplotlib style (optional) ---
plt.rcParams.update({
    'font.size': 18,
    'font.family': 'Times New Roman',
    'text.usetex': True
})

# --- Paths ---
path = r"D:\blueCFD-Core\2024\ofuser-of12\run\3_bird_90_deg_1_sep"
save_path = os.path.join(path, "pod_test_raw_including_mean")
os.makedirs(save_path, exist_ok=True)

times_txt = os.path.join(path, "times.txt")
if not os.path.isfile(times_txt):
    raise FileNotFoundError(f"Could not find {times_txt}")

# --- Load times (strip blanks/comments) ---
Times = [t.strip() for t in open(times_txt).read().splitlines()
         if t.strip() and not t.strip().startswith('#')]
Snapshots = len(Times)
if Snapshots == 0:
    raise ValueError("No times found in times.txt")

# --- Read one velocity to get sizes ---
vel = fl.readvector(path, time_name='latestTime', name='U', structured=False)
if vel.ndim != 2 or vel.shape[0] != 3:
    raise ValueError(f"Unexpected vel shape {vel.shape}. Expected (3, n_cells) for structured=False.")
n_comp, n_cells = vel.shape
total = vel.size  # 3 * n_cells

# =====================================================================
#               POD ON THE FULL (UN-CENTERED) FIELD (PURE)
# =====================================================================
# Snapshot matrix X: shape (total, Snapshots); *no* mean removal.
X = np.zeros((total, Snapshots), dtype=np.float64)
for j, ti in enumerate(Times):
    ui = fl.readvector(path, time_name=str(ti), name='U', structured=False)
    if ui.shape != vel.shape:
        raise ValueError(f"U({ti}) shape {ui.shape} mismatches first snapshot {vel.shape}")
    X[:, j] = ui.T.reshape(total, order='F')

np.save(os.path.join(save_path, "X.npy"), X)

# Empirical mean of snapshots (verification only; not used to build POD)
mu = np.mean(X, axis=1)

# --- Method of Snapshots on X (not centered) ---
M = Snapshots
C = (X.T @ X) / M
eigvals, eigvecs = np.linalg.eigh(C)
idx = np.argsort(eigvals)[::-1]
S = eigvals[idx].real
V = eigvecs[:, idx].real
sigma = np.sqrt(np.maximum(S, 0.0))

# Guard against tiny/zero eigenvalues to avoid division issues
eps = 1e-14 * (sigma.max() if sigma.size else 0.0)
sigma_safe = sigma.copy()
sigma_safe[sigma_safe < eps] = np.inf

# Spatial modes (columns): Φ = X (V / σ)
Modes = X @ (V / sigma_safe)             # (total, M), columns orthonormal up to √M scaling
# Time coefficients: A = Σ V^T
A = (sigma[:, None] * V.T)                # (M, M); row k is a_k(t)

# --- Energy content ---
Energy = S / S.sum() if S.sum() > 0 else S * 0.0
cumEnergy = np.cumsum(Energy)

# =====================================================================
#                     PURE DIAGNOSTICS (NO RESCALING)
# =====================================================================
# With C = (1/M) X^T X, Φ^T X = M A  → compare A to (Φ^T X)/M
A_check = (Modes.T @ X) / M
coeff_mismatch = np.linalg.norm(A - A_check) / (np.linalg.norm(A) + 1e-30)

# Reconstruction: X ≈ Φ A
X_hat = Modes @ A
rel_rec_err = np.linalg.norm(X - X_hat) / (np.linalg.norm(X) + 1e-30)

# Orthonormal SVD view: U = Φ / √M,  A_svd = √M A
U = Modes / np.sqrt(M)
A_svd = np.sqrt(M) * A
coeff_mismatch_orth = np.linalg.norm(U.T @ X - A_svd) / (np.linalg.norm(A_svd) + 1e-30)
orth_err = np.linalg.norm(U.T @ U - np.eye(U.shape[1]))

# "Is Mode0 the mean?" — scale-aware metrics (still pure; just reading outputs)
phi0 = Modes[:, 0]
phi0_norm = np.linalg.norm(phi0) + 1e-30
mu_norm  = np.linalg.norm(mu) + 1e-30
cos_sim = (phi0 @ mu) / (phi0_norm * mu_norm)

# Use POD time coefficients to extract mean contribution (PURE):
mean_coeffs = np.mean(A, axis=1)             # \bar{A}
mu_from_coeffs = Modes @ mean_coeffs         # should match mu
rel_err_mu_from_coeffs = np.linalg.norm(mu - mu_from_coeffs) / (mu_norm + 1e-30)

# The **Mode-0 contribution to the mean** (no gymnastics):  φ0 * mean(a0)
a0_bar = mean_coeffs[0]
mode0_meanPart = phi0 * a0_bar               # this is what to compare in magnitude to mean

# For reference only, best-fit scalar via projection (not used to save/modify):
s0 = (phi0 @ mu) / (phi0_norm**2)
rel_err_mu_vs_mode0 = np.linalg.norm(mu - s0 * phi0) / (mu_norm + 1e-30)

# Mode-0 coefficient constancy
a0 = A[0, :]
cv_a0 = (np.std(a0) / (np.mean(np.abs(a0)) + 1e-30)) if np.mean(np.abs(a0)) > 0 else np.nan

# =====================================================================
#                              SAVE OUTPUTS
# =====================================================================
np.save(os.path.join(save_path, 'eigenvalues.npy'), S)
np.save(os.path.join(save_path, 'energy.npy'), Energy)
np.save(os.path.join(save_path, 'modes.npy'), Modes)               # normalized modes (Mode0 first)
np.save(os.path.join(save_path, 'amps.npy'), A)                    # time coefficients
np.save(os.path.join(save_path, 'mu_empirical.npy'), mu)
np.save(os.path.join(save_path, 'mu_from_coeffs.npy'), mu_from_coeffs)
np.save(os.path.join(save_path, 'mode0_meanPart.npy'), mode0_meanPart)  # φ0 * mean(a0)  ← use this for magnitude

# Write OpenFOAM-style fields if header/footer exist
header_path = os.path.join(path, 'header.txt')
footer_path = os.path.join(path, 'footer.txt')
save_info_foam = ""
if os.path.isfile(header_path) and os.path.isfile(footer_path):
    with open(header_path, 'r', encoding='utf-8', newline='') as fh:
        header = fh.read()
    with open(footer_path, 'r', encoding='utf-8', newline='') as ff:
        footer = ff.read()

    saveTime = os.path.join(path, "pod_test_time_raw")
    os.makedirs(saveTime, exist_ok=True)

    # Save the canonical (normalized) Mode0 shape
    mode0_shape_field = phi0.reshape((n_cells, 3), order='F')
    np.savetxt(os.path.join(saveTime, "Mode0_shape"),
               mode0_shape_field, fmt='(% .8e % .8e % .8e)',
               header=header, footer=footer, comments='')

    # Save Mode0's *pure POD* contribution to the mean: φ0 * mean(a0)
    mode0_meanPart_field = mode0_meanPart.reshape((n_cells, 3), order='F')
    np.savetxt(os.path.join(saveTime, "Mode0_meanPart"),
               mode0_meanPart_field, fmt='(% .8e % .8e % .8e)',
               header=header, footer=footer, comments='')

    # Save the empirical mean and the POD-mean
    mu_field = mu.reshape((n_cells, 3), order='F')
    np.savetxt(os.path.join(saveTime, "Mean_empirical"),
               mu_field, fmt='(% .8e % .8e % .8e)',
               header=header, footer=footer, comments='')

    mu_coeffs_field = mu_from_coeffs.reshape((n_cells, 3), order='F')
    np.savetxt(os.path.join(saveTime, "Mean_from_coeffs"),
               mu_coeffs_field, fmt='(% .8e % .8e % .8e)',
               header=header, footer=footer, comments='')

    # (Optional) also dump a few more normalized modes for reference
    n_write = min(10, Modes.shape[1])
    for i in range(1, n_write):
        mode_i = Modes[:, i].reshape((n_cells, 3), order='F')
        np.savetxt(os.path.join(saveTime, f"Mode{i}_shape"),
                   mode_i, fmt='(% .8e % .8e % .8e)',
                   header=header, footer=footer, comments='')

    save_info_foam = (f"Wrote Mode0_shape, Mode0_meanPart, Mean_empirical, Mean_from_coeffs, "
                      f"and Mode1_shape..Mode{n_write-1}_shape to: {saveTime}")
else:
    save_info_foam = "header.txt/footer.txt not found; skipped OpenFOAM-style saves."

# =====================================================================
#                            OPTIONAL PLOTS
# =====================================================================
plt.figure()
plt.plot(np.arange(1, len(Energy)+1), Energy, marker='o')
plt.xlabel('Mode index'); plt.ylabel('Energy fraction')
plt.title('POD Energy Spectrum (uncentered)'); plt.grid(True)

plt.figure()
plt.plot(np.arange(1, len(cumEnergy)+1), cumEnergy, marker='o')
plt.xlabel('Mode index'); plt.ylabel('Cumulative energy')
plt.title('Cumulative POD Energy (uncentered)'); plt.grid(True)

plt.figure()
plt.plot(np.arange(M), A[0, :], marker='o', ms=3)
plt.xlabel('Snapshot index'); plt.ylabel('a0(t)')
plt.title('Mode 0 time coefficients'); plt.grid(True)

plt.show()

# =====================================================================
#                        FINAL DIAGNOSTIC PRINT
# =====================================================================
top = min(10, len(Energy))
energy_str = ", ".join([f"{Energy[i]:.6f}" for i in range(top)])
cum_str    = ", ".join([f"{cumEnergy[i]:.6f}" for i in range(top)])

phi0_mu_cos = cos_sim
rel_mu_via_coeffs = rel_err_mu_from_coeffs
rel_mu_vs_mode0bf = rel_err_mu_vs_mode0

print("\n" + "="*72)
print("POD (uncentered, PURE) — FINAL DIAGNOSTICS")
print("="*72)
print(f"Snapshots                 : {Snapshots}")
print(f"n_cells                   : {n_cells}")
print(f"components                : {n_comp}")
print(f"total entries             : {total}")
print("-"*72)
print(f"Saved arrays path         : {save_path}")
print(f"{save_info_foam}")
print("-"*72)
print("As-built scaling (Φ, A) with C = (1/M) XᵀX")
print(f"Coeff consistency  ||A - (ΦᵀX)/M||/||A|| : {coeff_mismatch:.3e}")
print(f"Reconstruction    ||X - ΦA||/||X||       : {rel_rec_err:.3e}")
print("-"*72)
print("Orthonormal SVD view (U = Φ/√M,  A_svd = √M·A)")
print(f"UᵀX vs A_svd mismatch                    : {coeff_mismatch_orth:.3e}")
print(f"Orthonormality  ||UᵀU - I||              : {orth_err:.3e}")
print("-"*72)
print(f"Top-{top} Energy fractions               : {energy_str}")
print(f"Top-{top} Cumulative energy             : {cum_str}")
print("-"*72)
print(f"cos(mean, Mode0_shape)                   : {phi0_mu_cos:.6f}  (1.0 → aligned)")
print(f"||mu - Φ*mean(A)|| / ||mu||              : {rel_mu_via_coeffs:.3e}  (POD mean check)")
print(f"||mu - s0*Mode0_shape|| / ||mu||         : {rel_mu_vs_mode0bf:.3e}  (for reference)")
print("-"*72)
print("Files to compare in ParaView:")
print("  • Mean_empirical            (true mean)")
print("  • Mean_from_coeffs          (POD mean, should match true mean)")
print("  • Mode0_meanPart            (φ0 * mean(a0): Mode0’s pure contribution to the mean, correct magnitude)")
print("  • Mode0_shape               (normalized mode shape; not for magnitude checks)")
print("="*72 + "\n")
