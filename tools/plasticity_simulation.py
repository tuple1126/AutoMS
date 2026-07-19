#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script_parallel_casefit_fit_v4.py

Purpose
-------
Robust small-strain compression simulation for lattice/microstructure specimens,
with an optional apparent-curve correction that accounts for platen/contact/machine
compliance without changing the base material parameters E, nu, and G.

Main changes from the old robust script
--------------------------------------
1) The constitutive model is embedded in this single file, so you only need this script.
2) Boundary update bug is fixed: for smooth_platen/frictionless_platen the top-z
   displacement BC is updated by index, not by blindly replacing the last BC value.
3) Top/bottom surface band detection is thinner and configurable. The old script could
   clamp a thick z-band, which easily makes the response too stiff.
4) Progress is printed after every converged increment: step, strain, stress, force,
   eqps, substeps, and wall time.
5) By default, the code saves both raw FE results and a corrected apparent curve:
       eps_report = eps_FE + c_sigma * sigma_FE_corrected
       sigma_report = k_sigma * sigma_FE
   This correction is fitted to the experimental mean curve and is meant to represent
   system/contact compliance and apparent-area/load normalization uncertainty. It does
   NOT modify E, nu, or G in the material model.

Typical command
---------------
CUDA_VISIBLE_DEVICES=0 python script_parallel_casefit_fit_v4.py \
  --mesh-file ./S-P-30-759_2x2x2.obj_.msh \
  --experiment-files ./S-P-30-759_2x2x2.obj_.msh.txt \
  --case-id micro_B_v4 \
  --output-root ./fit_runs \
  --no-mesh-scaling \
  --specimen-height-mm 40 \
  --specimen-width-mm 40 \
  --specimen-depth-mm 40 \
  --boundary-mode rough_platen \
  --max-engineering-strain 0.05 \
  --target-num-steps 100 \
  --params-json '{"E":1951,"nu":0.4,"sig0":35,"eps0":0.02,"n":0.2}'
"""

import os
import sys
import time
import json
import uuid
import shutil
import argparse
import gc
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

# Keep before importing JAX
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import jax
import jax.numpy as np
import numpy as onp
import meshio
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.sparse.linalg

try:
    import fcntl
except Exception:  # pragma: no cover - Windows fallback
    fcntl = None

from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh


DEFAULT_MATERIAL_PARAMS = {
    "E": 1951.0,
    "nu": 0.4,
    "sig0": 35.0,
    "hardening_model": "swift",
    "eps0": 0.02,
    "n": 0.2,
    "plasticity_tolerance": 1e-8,
    # Legacy linear-Voce parameters are retained for explicit compatibility.
    "H1": 10.0,
    "Q_inf": 18.0,
    "b": 8.0,
    "eta": 0.0,
}


class SmallStrainJ2Plasticity(Problem):
    """Small-strain J2 plasticity with isotropic hardening and optional viscosity.

    Units are inherited from the mesh and material input. If coordinates are in mm
    and E/sig0 are in MPa, nodal reaction is in N, so reaction/area gives MPa.
    """

    def __init__(self, *args, material_params=None, **kwargs):
        self.material_params = dict(DEFAULT_MATERIAL_PARAMS)
        if material_params:
            self.material_params.update(material_params)
        super().__init__(*args, **kwargs)

    def custom_init(self):
        self.fe = self.fes[0]
        n_cells = len(self.fe.cells)
        n_quads = self.fe.num_quads

        self.cell_volumes = np.sum(self.fe.JxW, axis=1)
        self.total_volume = np.sum(self.cell_volumes)

        self.epsp_old = np.zeros((n_cells, n_quads, self.dim, self.dim))
        self.alpha_old = np.zeros((n_cells, n_quads))
        self.internal_vars = [self.epsp_old, self.alpha_old]

        _, update_int_vars_map, compute_stress_map = self.get_maps()
        self._vmap_update_int_vars_map = jax.jit(jax.vmap(jax.vmap(update_int_vars_map)))
        self._vmap_compute_stress = jax.jit(jax.vmap(jax.vmap(compute_stress_map)))

    def get_tensor_map(self):
        tensor_map, _, _ = self.get_maps()
        return tensor_map

    def get_maps(self):
        E = float(self.material_params["E"])
        nu = float(self.material_params["nu"])
        sig0 = float(self.material_params["sig0"])
        hardening_model = str(self.material_params.get("hardening_model", "swift")).lower()
        eps0 = max(float(self.material_params.get("eps0", 0.02)), 1e-12)
        swift_n = float(self.material_params.get("n", 0.2))
        plasticity_tolerance = float(self.material_params.get("plasticity_tolerance", 1e-8))
        H1 = float(self.material_params["H1"])
        Q_inf = float(self.material_params.get("Q_inf", 0.0))
        b = float(self.material_params.get("b", 1.0))
        eta = float(self.material_params.get("eta", 0.0))

        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        G = E / (2.0 * (1.0 + nu))
        I = np.eye(self.dim)
        tiny = 1e-12

        def sym(A):
            return 0.5 * (A + A.T)

        def dev(A):
            return A - np.trace(A) * I / self.dim

        def elastic_stress(eps_e):
            return lam * np.trace(eps_e) * I + 2.0 * G * eps_e

        def hardening(alpha):
            if hardening_model == "swift":
                return sig0 * (1.0 + alpha / eps0) ** swift_n
            return sig0 + H1 * alpha + Q_inf * (1.0 - np.exp(-b * alpha))

        def hardening_slope(alpha):
            if hardening_model == "swift":
                return sig0 * swift_n / eps0 * (1.0 + alpha / eps0) ** (swift_n - 1.0)
            return H1 + Q_inf * b * np.exp(-b * alpha)

        def local_return_map(u_grad, epsp_old, alpha_old):
            eps = sym(np.nan_to_num(u_grad, nan=0.0, posinf=1e6, neginf=-1e6))
            eps_e_trial = eps - epsp_old
            sigma_trial = elastic_stress(eps_e_trial)
            s_trial = dev(sigma_trial)
            seq_trial = np.sqrt(np.maximum(1.5 * np.sum(s_trial * s_trial), 0.0))
            f_trial = seq_trial - hardening(alpha_old)
            dt_eff = np.maximum(getattr(self, "dt", 1.0), 1e-12)

            def elastic_branch():
                return epsp_old, alpha_old, sigma_trial

            def plastic_branch():
                denom0 = np.maximum(3.0 * G + hardening_slope(alpha_old) + eta / dt_eff, 1e-9)
                dp0 = np.maximum(f_trial, 0.0) / denom0

                def body_fun(_, state):
                    dp, converged = state
                    alpha = alpha_old + dp
                    res = seq_trial - 3.0 * G * dp - hardening(alpha) - eta * dp / dt_eff
                    jac = -(3.0 * G + hardening_slope(alpha) + eta / dt_eff)
                    dp_new = dp - res / np.where(np.abs(jac) > tiny, jac, -1.0)
                    upper = seq_trial / np.maximum(3.0 * G, 1e-9) + 1.0
                    dp_new = np.clip(dp_new, 0.0, upper)
                    newly_converged = np.abs(dp_new - dp) < plasticity_tolerance
                    return np.where(converged, dp, dp_new), np.logical_or(converged, newly_converged)

                dp, _ = jax.lax.fori_loop(0, 25, body_fun, (dp0, False))
                n = np.where(seq_trial > tiny, s_trial / seq_trial, np.zeros_like(s_trial))
                s_new = s_trial - 3.0 * G * dp * n
                p_trial = np.trace(sigma_trial) / self.dim
                sigma_new = s_new + p_trial * I
                depsp = 1.5 * dp * n
                epsp_new = sym(epsp_old + depsp)
                alpha_new = alpha_old + dp
                return epsp_new, alpha_new, sigma_new

            epsp_new, alpha_new, sigma = jax.lax.cond(
                f_trial <= 0.0,
                elastic_branch,
                plastic_branch,
            )
            epsp_new = np.nan_to_num(epsp_new, nan=0.0, posinf=1e6, neginf=-1e6)
            alpha_new = np.nan_to_num(alpha_new, nan=0.0, posinf=1e6, neginf=0.0)
            sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e12, neginf=-1e12)
            return epsp_new, alpha_new, sigma

        def tensor_map(u_grad, epsp_old, alpha_old):
            _, _, sigma = local_return_map(u_grad, epsp_old, alpha_old)
            return sigma

        def update_int_vars_map(u_grad, epsp_old, alpha_old):
            epsp_new, alpha_new, _ = local_return_map(u_grad, epsp_old, alpha_old)
            return epsp_new, alpha_new

        def compute_stress_map(u_grad, epsp_old, alpha_old):
            _, _, sigma = local_return_map(u_grad, epsp_old, alpha_old)
            return sigma

        return tensor_map, update_int_vars_map, compute_stress_map

    def _compute_u_grads(self, sol):
        u_grads = (
            np.take(sol, self.fe.cells, axis=0)[:, None, :, :, None]
            * self.fe.shape_grads[:, :, :, None, :]
        )
        return np.sum(u_grads, axis=2)

    def update_int_vars_gp(self, sol, int_vars):
        u_grads = self._compute_u_grads(sol)
        return self._vmap_update_int_vars_map(u_grads, *int_vars)

    def set_params(self, params):
        self.internal_vars = params

    def compute_cell_average_stress(self, sol, int_vars):
        u_grads = self._compute_u_grads(sol)
        sigma = self._vmap_compute_stress(u_grads, *int_vars)
        return np.sum(sigma * self.fe.JxW[:, :, None, None], axis=1) / self.cell_volumes[:, None, None]

    def compute_volume_average_stress(self, sol, int_vars):
        sigma_cell = self.compute_cell_average_stress(sol, int_vars)
        return np.sum(sigma_cell * self.cell_volumes[:, None, None], axis=0) / self.total_volume

    def compute_avg_stress(self, sol, int_vars):
        return self.compute_volume_average_stress(sol, int_vars)

    def compute_stress(self, sol, int_vars):
        u_grads = self._compute_u_grads(sol)
        return self._vmap_compute_stress(u_grads, *int_vars)

    def get_eqps(self, int_vars=None):
        if int_vars is None:
            int_vars = self.internal_vars
        return int_vars[1]


@dataclass
class SimulationState:
    disp: float
    time_value: float
    sol_list: list
    params: list


@dataclass
class ApparentCorrection:
    enabled: bool
    mode: str
    stress_scale: float
    strain_compliance_per_MPa: float
    raw_score: Dict[str, float]
    corrected_score: Dict[str, float]


def sanitize_name(text: str) -> str:
    text = os.path.basename(str(text))
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("_") or "case"


def parse_args():
    parser = argparse.ArgumentParser(description="Robust microstructure compression fitting on stress-strain")
    parser.add_argument("--mesh-file", required=True)
    parser.add_argument("--experiment-files", nargs="+", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output-root", default="./fit_runs")
    parser.add_argument("--run-tag", default=None)

    parser.add_argument("--real-dimension-mm", type=float, default=None, help="Optional scaling target. Omit to keep original mesh size.")
    parser.add_argument("--dimension-axis", choices=["x", "y", "z", "max_span"], default="z")
    parser.add_argument("--anisotropic-scale", action="store_true")
    parser.add_argument("--no-mesh-scaling", action="store_true", default=False)

    parser.add_argument("--specimen-height-mm", type=float, default=40.0)
    parser.add_argument("--specimen-width-mm", type=float, default=40.0)
    parser.add_argument("--specimen-depth-mm", type=float, default=40.0)

    parser.add_argument(
        "--boundary-mode",
        choices=["rough_platen", "smooth_platen", "frictionless_platen"],
        default="rough_platen",
        help="rough: top/bottom x,y,z fixed to platen; smooth/frictionless: only z platen plus bottom anchor nodes.",
    )
    parser.add_argument("--target-num-steps", type=int, default=100)
    parser.add_argument("--max-engineering-strain", type=float, default=0.05)
    parser.add_argument("--max-subdivisions", type=int, default=8)

    parser.add_argument("--zero-shift-experiment", action="store_true", default=False)
    parser.add_argument("--no-zero-shift-experiment", action="store_false", dest="zero_shift_experiment")
    parser.add_argument("--use-experiment-timestamps", action="store_true", default=True)
    parser.add_argument("--no-experiment-timestamps", action="store_false", dest="use_experiment_timestamps")

    parser.add_argument("--save-plot", action="store_true", default=True)
    parser.add_argument("--no-save-plot", action="store_false", dest="save_plot")
    parser.add_argument("--plot-raw-simulation", action="store_true", default=True)
    parser.add_argument("--no-plot-raw-simulation", action="store_false", dest="plot_raw_simulation")

    # Thin surface band is important: thick clamped bands make the specimen too stiff.
    parser.add_argument("--surface-min-nodes", type=int, default=16)
    parser.add_argument("--surface-band-frac", type=float, default=0.003, help="Preferred thin band fraction of model height.")
    parser.add_argument("--surface-max-band-frac", type=float, default=0.02, help="Maximum allowed surface band fraction before quantile fallback.")

    # Apparent curve correction: does not touch material E/nu/G.
    parser.add_argument("--post-fit-correction", action="store_true", default=False)
    parser.add_argument("--no-post-fit-correction", action="store_false", dest="post_fit_correction")
    parser.add_argument(
        "--correction-mode",
        choices=["compliance", "stress_scale", "compliance_stress_scale"],
        default="compliance_stress_scale",
    )
    parser.add_argument("--stress-scale-min", type=float, default=0.45)
    parser.add_argument("--stress-scale-max", type=float, default=1.05)
    parser.add_argument("--strain-compliance-min", type=float, default=0.0)
    parser.add_argument("--strain-compliance-max", type=float, default=0.012, help="Upper bound of strain/MPa for machine/contact compliance.")

    parser.add_argument("--params-json", default=None)
    return parser.parse_args()


def read_text_lines(path: str) -> List[str]:
    last_exc = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.readlines()
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"failed to read {path}: {last_exc}")


def parse_experiment_txt(path: str, zero_shift: bool = True) -> Dict[str, Any]:
    disp, force, t = [], [], []
    for line in read_text_lines(path):
        toks = re.split(r"\s+", line.strip())
        if len(toks) < 3:
            continue
        try:
            d = float(toks[0])
            p = float(toks[1])
            ti = float(toks[2])
        except Exception:
            continue
        if onp.isfinite(d) and onp.isfinite(p) and onp.isfinite(ti):
            disp.append(d)
            force.append(p)
            t.append(ti)

    if len(disp) < 5:
        raise RuntimeError(f"Insufficient data could be parsed from experiment file {path}")

    disp = onp.asarray(disp, dtype=float)
    force = onp.asarray(force, dtype=float)
    t = onp.asarray(t, dtype=float)

    if zero_shift:
        disp = disp - disp[0]
        force = force - force[0]
        t = t - t[0]

    # Keep monotonically increasing displacement branch only.
    keep = [0]
    for i in range(1, len(disp)):
        if disp[i] > disp[keep[-1]]:
            keep.append(i)
    disp = disp[keep]
    force = force[keep]
    t = t[keep]

    return {
        "path": path,
        "disp_mm": disp,
        "force_kN": force,
        "time_s": t,
        "max_disp_mm": float(disp[-1]),
        "max_force_kN": float(force[-1]),
        "n_points": int(len(disp)),
    }


def build_experiment_target(
    experiments: List[Dict[str, Any]],
    num_steps: int,
    specimen_height_mm: float,
    specimen_area_mm2: float,
    use_experiment_timestamps: bool,
    max_engineering_strain: float,
) -> Dict[str, Any]:
    common_max_disp = min(exp["max_disp_mm"] for exp in experiments)
    common_max_disp = min(common_max_disp, max_engineering_strain * specimen_height_mm)
    common_max_disp = max(common_max_disp, 1e-9)

    grid_disp = onp.linspace(0.0, common_max_disp, 250)
    grid_strain = grid_disp / max(specimen_height_mm, 1e-12)

    stress_stack = []
    time_stack = []
    for exp in experiments:
        force_interp = onp.interp(grid_disp, exp["disp_mm"], exp["force_kN"])
        stress_stack.append(force_interp * 1000.0 / max(specimen_area_mm2, 1e-12))
        time_stack.append(onp.interp(grid_disp, exp["disp_mm"], exp["time_s"]))

    stress_stack = onp.vstack(stress_stack)
    time_stack = onp.vstack(time_stack)
    stress_mean = onp.mean(stress_stack, axis=0)
    stress_std = onp.std(stress_stack, axis=0, ddof=1) if len(experiments) > 1 else onp.zeros_like(stress_mean)
    stress_min = onp.min(stress_stack, axis=0)
    stress_max = onp.max(stress_stack, axis=0)
    time_mean = onp.mean(time_stack, axis=0)

    target_strain = onp.linspace(0.0, common_max_disp / max(specimen_height_mm, 1e-12), num_steps)
    if use_experiment_timestamps:
        target_time = onp.interp(target_strain, grid_strain, time_mean)
    else:
        target_time = onp.linspace(0.0, float(time_mean[-1]), num_steps)

    return {
        "full_grid_strain": grid_strain,
        "grid_disp_mm": grid_disp,
        "stress_mean_MPa": stress_mean,
        "stress_std_MPa": stress_std,
        "stress_min_MPa": stress_min,
        "stress_max_MPa": stress_max,
        "target_strain": target_strain,
        "target_time_s": target_time,
        "specimen_height_mm": float(specimen_height_mm),
        "specimen_area_mm2": float(specimen_area_mm2),
        "common_max_disp_mm": float(common_max_disp),
        "common_max_strain": float(common_max_disp / max(specimen_height_mm, 1e-12)),
    }


def get_span(points: onp.ndarray, axis: str) -> float:
    if axis == "x":
        return float(onp.max(points[:, 0]) - onp.min(points[:, 0]))
    if axis == "y":
        return float(onp.max(points[:, 1]) - onp.min(points[:, 1]))
    if axis == "z":
        return float(onp.max(points[:, 2]) - onp.min(points[:, 2]))
    if axis == "max_span":
        return float(onp.max(onp.ptp(points[:, :3], axis=0)))
    raise ValueError(axis)


def load_mesh(mesh_file: str, real_dimension_mm, dimension_axis: str, anisotropic_scale: bool, no_mesh_scaling: bool):
    print(f"[INFO] reading mesh: {mesh_file}", flush=True)
    meshio_mesh = meshio.read(mesh_file)
    points = onp.asarray(meshio_mesh.points, dtype=float).copy()
    spans_before = onp.ptp(points[:, :3], axis=0)

    scale_meta = {
        "mesh_scaled": False,
        "scale_factor": 1.0,
        "reference_axis": dimension_axis,
        "spans_before_mm": spans_before.tolist(),
        "spans_after_mm": spans_before.tolist(),
    }

    if not no_mesh_scaling and real_dimension_mm is not None:
        ref_span = get_span(points, dimension_axis)
        if ref_span <= 0.0:
            raise RuntimeError("mesh reference span must be positive")
        scale = float(real_dimension_mm) / ref_span
        if anisotropic_scale:
            ai = {"x": 0, "y": 1, "z": 2}.get(dimension_axis)
            if ai is None:
                raise ValueError("anisotropic scale only supports x/y/z")
            points[:, ai] *= scale
        else:
            points[:, :3] *= scale
        scale_meta["mesh_scaled"] = True
        scale_meta["scale_factor"] = float(scale)
        scale_meta["spans_after_mm"] = onp.ptp(points[:, :3], axis=0).tolist()
        print(f"[INFO] mesh scaled by {scale:.8f}", flush=True)
    else:
        print("[INFO] mesh scaling disabled; using original mesh coordinates", flush=True)

    return meshio.Mesh(points=points, cells=meshio_mesh.cells), scale_meta


def value_fn(val):
    def f(_point):
        return val
    return f


def compression_value_fn(disp):
    def f(_point):
        return disp
    return f


def point_selector(target_point, tol):
    target = onp.asarray(target_point, dtype=float)
    def selector(point):
        return bool(onp.linalg.norm(onp.asarray(point) - target) <= tol)
    return selector


def _pick_anchor_nodes(points, bottom_nodes):
    bottom_points = points[bottom_nodes]
    x_min = onp.min(bottom_points[:, 0])
    y_min = onp.min(bottom_points[:, 1])
    xy = bottom_points[:, :2]
    p0_local = int(onp.argmin(onp.linalg.norm(xy - onp.array([x_min, y_min]), axis=1)))
    p0 = bottom_nodes[p0_local]

    ref0 = points[p0, :2]
    dist = onp.linalg.norm(xy - ref0[None, :], axis=1)
    p1_local = int(onp.argmax(dist))
    p1 = bottom_nodes[p1_local]

    a = points[p0, :2]
    b = points[p1, :2]
    ab = b - a
    denom = float(onp.dot(ab, ab))
    if denom <= 1e-20:
        p2 = p1
    else:
        tt = onp.clip(((xy - a[None, :]) @ ab) / denom, 0.0, 1.0)
        proj = a[None, :] + tt[:, None] * ab[None, :]
        dist_line = onp.linalg.norm(xy - proj, axis=1)
        p2_local = int(onp.argmax(dist_line))
        p2 = bottom_nodes[p2_local]
    return int(p0), int(p1), int(p2)


def _surface_band_masks(axis_vals, min_nodes=16, preferred_band_frac=0.003, max_band_frac=0.02):
    """Pick a thin top/bottom band.

    The old script required 256 nodes, which may enlarge the band to several percent
    of specimen height and over-constrain a volume slab. Here we use a small minimum
    by default and report the selected band thickness for diagnosis.
    """
    axis_vals = onp.asarray(axis_vals, dtype=float)
    z_min = float(onp.min(axis_vals))
    z_max = float(onp.max(axis_vals))
    L = max(z_max - z_min, 1e-12)

    fractions = sorted(set([
        1e-8, 1e-7, 1e-6, 5e-6, 1e-5, 5e-5,
        1e-4, 2e-4, 5e-4, 1e-3, preferred_band_frac,
        5e-3, 1e-2, max_band_frac,
    ]))
    bottom_mask = top_mask = None
    band = None
    for frac in fractions:
        if frac > max_band_frac + 1e-15:
            continue
        band_try = max(frac * L, 1e-12)
        bmask = axis_vals <= z_min + band_try
        tmask = axis_vals >= z_max - band_try
        if int(bmask.sum()) >= min_nodes and int(tmask.sum()) >= min_nodes:
            bottom_mask, top_mask, band = bmask, tmask, float(band_try)
            break

    if bottom_mask is None:
        # Fallback: choose nearest min_nodes by z coordinate, still thinest possible.
        order_bot = onp.argsort(axis_vals)
        order_top = onp.argsort(-axis_vals)
        bottom_mask = onp.zeros_like(axis_vals, dtype=bool)
        top_mask = onp.zeros_like(axis_vals, dtype=bool)
        bottom_mask[order_bot[:min(min_nodes, len(axis_vals))]] = True
        top_mask[order_top[:min(min_nodes, len(axis_vals))]] = True
        band = max(float(onp.max(axis_vals[bottom_mask]) - z_min), float(z_max - onp.min(axis_vals[top_mask])))

    return bottom_mask, top_mask, float(band)


def _describe_band(points, mask, axis_index):
    pts = points[mask]
    return {
        "n_nodes": int(len(pts)),
        "axis_min": float(onp.min(pts[:, axis_index])),
        "axis_max": float(onp.max(pts[:, axis_index])),
    }


def build_boundary_conditions(mesh, boundary_mode, surface_min_nodes, surface_band_frac, surface_max_band_frac):
    points = mesh.points
    x_min, y_min, z_min = onp.min(points[:, 0]), onp.min(points[:, 1]), onp.min(points[:, 2])
    x_max, y_max, z_max = onp.max(points[:, 0]), onp.max(points[:, 1]), onp.max(points[:, 2])
    Lx, Ly, Lz = x_max - x_min, y_max - y_min, z_max - z_min
    tol_node = max(5e-4 * min(max(Lx, 1e-12), max(Ly, 1e-12), max(Lz, 1e-12)), 1e-10)

    bottom_mask, top_mask, band_thickness = _surface_band_masks(
        points[:, 2],
        min_nodes=max(3, int(surface_min_nodes)),
        preferred_band_frac=float(surface_band_frac),
        max_band_frac=float(surface_max_band_frac),
    )
    bottom_nodes = onp.where(bottom_mask)[0]
    top_nodes = onp.where(top_mask)[0]
    if len(bottom_nodes) == 0 or len(top_nodes) == 0:
        raise RuntimeError("failed to detect top/bottom bands")

    anchor0, anchor1, anchor2 = _pick_anchor_nodes(points, bottom_nodes)
    p0 = points[anchor0]
    p1 = points[anchor1]
    p2 = points[anchor2]

    z_bot_cut = float(onp.max(points[bottom_mask, 2]))
    z_top_cut = float(onp.min(points[top_mask, 2]))

    def bottom_surface(point):
        return point[2] <= z_bot_cut

    def top_surface(point):
        return point[2] >= z_top_cut

    if boundary_mode in ("smooth_platen", "frictionless_platen"):
        # Frictionless/smooth platen: prescribe only z on top/bottom surfaces;
        # use three bottom anchor constraints only to remove rigid body modes.
        dirichlet_bc_info = [
            [bottom_surface, top_surface, point_selector(p0, tol_node), point_selector(p0, tol_node), point_selector(p1, tol_node), point_selector(p2, tol_node)],
            [2, 2, 0, 1, 1, 0],
            [value_fn(0.0), compression_value_fn(0.0), value_fn(0.0), value_fn(0.0), value_fn(0.0), value_fn(0.0)],
        ]
        compression_bc_index = 1
    else:
        # Rough platen: fully tie top/bottom bands to platen in x/y/z.
        dirichlet_bc_info = [
            [bottom_surface, bottom_surface, bottom_surface, top_surface, top_surface, top_surface],
            [0, 1, 2, 0, 1, 2],
            [value_fn(0.0), value_fn(0.0), value_fn(0.0), value_fn(0.0), value_fn(0.0), compression_value_fn(0.0)],
        ]
        compression_bc_index = 5

    bc_meta = {
        "mode": boundary_mode,
        "Lx": float(Lx),
        "Ly": float(Ly),
        "Lz": float(Lz),
        "apparent_area": float(max(Lx * Ly, 1e-12)),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "z_bot_cut": float(z_bot_cut),
        "z_top_cut": float(z_top_cut),
        "band_thickness": float(band_thickness),
        "band_thickness_over_Lz": float(band_thickness / max(Lz, 1e-12)),
        "top_nodes": top_nodes,
        "bottom_nodes": bottom_nodes,
        "top_band_info": _describe_band(points, top_mask, 2),
        "bottom_band_info": _describe_band(points, bottom_mask, 2),
        "anchor0": int(anchor0),
        "anchor1": int(anchor1),
        "anchor2": int(anchor2),
        "compression_bc_index": int(compression_bc_index),
    }
    return dirichlet_bc_info, bc_meta


def initialize_problem(meshio_mesh, material_params, args):
    available_cells = list(meshio_mesh.cells_dict.keys())
    if "tetra" in available_cells:
        cell_type = "tetra"
        ele_type = "TET4"
    elif "hexahedron" in available_cells:
        cell_type = "hexahedron"
        ele_type = "HEX8"
    else:
        raise RuntimeError(f"unsupported cell types: {available_cells}")

    mesh = Mesh(onp.asarray(meshio_mesh.points, dtype=float), meshio_mesh.cells_dict[cell_type])
    dirichlet_bc_info, bc_meta = build_boundary_conditions(
        mesh,
        args.boundary_mode,
        args.surface_min_nodes,
        args.surface_band_frac,
        args.surface_max_band_frac,
    )
    problem = SmallStrainJ2Plasticity(
        mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=dirichlet_bc_info,
        material_params=material_params,
    )
    problem.top_node_inds = onp.array(bc_meta["top_nodes"], dtype=onp.int32)
    problem.bottom_node_inds = onp.array(bc_meta["bottom_nodes"], dtype=onp.int32)

    sol0 = np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))
    res0 = problem.compute_residual([sol0])[0]
    zero_residual_finite = bool(onp.all(onp.isfinite(onp.asarray(jax.device_get(res0)))))
    diagnostics = {
        "zero_residual_finite": zero_residual_finite,
        "n_total_nodes": int(problem.fes[0].num_total_nodes),
        "n_cells": int(len(problem.fe.cells)),
        "n_top_nodes": int(len(problem.top_node_inds)),
        "n_bottom_nodes": int(len(problem.bottom_node_inds)),
        "anchors": [int(bc_meta["anchor0"]), int(bc_meta["anchor1"]), int(bc_meta["anchor2"])],
        "band_thickness": float(bc_meta["band_thickness"]),
        "band_thickness_over_Lz": float(bc_meta["band_thickness_over_Lz"]),
        "bottom_band_info": bc_meta["bottom_band_info"],
        "top_band_info": bc_meta["top_band_info"],
        "model_dims_mm": [bc_meta["Lx"], bc_meta["Ly"], bc_meta["Lz"]],
        "apparent_area_mm2": bc_meta["apparent_area"],
        "compression_bc_index": int(bc_meta["compression_bc_index"]),
    }
    return problem, dirichlet_bc_info, bc_meta, diagnostics


def compute_reaction_force_kN(problem, sol):
    res_list = problem.compute_residual([sol])
    nodal_residual = res_list[0]
    top_reaction = -np.sum(nodal_residual[problem.top_node_inds, 2])
    return float(onp.asarray(jax.device_get(top_reaction))) / 1000.0


def compute_step_metrics(problem, sol, params, bc_meta):
    sigma_bar = onp.asarray(jax.device_get(problem.compute_volume_average_stress(sol, params)))
    eqps = onp.asarray(jax.device_get(problem.get_eqps(params)))
    reaction_kN = compute_reaction_force_kN(problem, sol)
    apparent_stress = reaction_kN * 1000.0 / max(bc_meta["apparent_area"], 1e-12)
    return {
        "sigma_bar_zz": float(sigma_bar[2, 2]),
        "sigma_bar_xx": float(sigma_bar[0, 0]),
        "sigma_bar_yy": float(sigma_bar[1, 1]),
        "reaction_kN": float(reaction_kN),
        "apparent_stress_MPa": float(apparent_stress),
        "eqps_mean": float(eqps.mean()),
        "eqps_max": float(eqps.max()),
    }


def make_regularized_solver(rel_reg: float, abs_reg: float = 1e-10):
    def custom_solver(A, b, x0, solver_options):
        indptr, indices, data = A.getValuesCSR()
        Asp = scipy.sparse.csr_matrix((data, indices, indptr), shape=A.getSize())
        n = Asp.shape[0]
        diag = onp.abs(Asp.diagonal())
        scale = float(diag.mean()) if diag.size else 0.0
        if not onp.isfinite(scale) or scale <= 0.0:
            scale = float(onp.mean(onp.abs(data))) if len(data) else 1.0
        if not onp.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        reg = max(abs_reg, rel_reg * scale)
        Areg = Asp + reg * scipy.sparse.eye(n, format="csr")
        rhs = onp.asarray(b, dtype=float)
        x = None
        try:
            x = scipy.sparse.linalg.spsolve(Areg, rhs)
        except Exception:
            x = None
        if x is None or onp.any(~onp.isfinite(x)):
            x = scipy.sparse.linalg.lsmr(Areg, rhs, atol=1e-10, btol=1e-10, maxiter=8000)[0]
        if onp.any(~onp.isfinite(x)):
            raise RuntimeError(f"regularized linear solve failed with reg={reg:.3e}")
        return x
    return custom_solver


SOLVER_CANDIDATES = [
    {
        "name": "reg_direct_1e-10",
        "options": {
            "custom_solver": make_regularized_solver(1e-10),
            "tol": 1e-4,
            "rel_tol": 1e-5,
            "line_search_flag": False,
        },
    },
    {
        "name": "reg_direct_1e-8",
        "options": {
            "custom_solver": make_regularized_solver(1e-8),
            "tol": 1e-4,
            "rel_tol": 1e-5,
            "line_search_flag": False,
        },
    },
    {
        "name": "reg_direct_1e-6",
        "options": {
            "custom_solver": make_regularized_solver(1e-6),
            "tol": 1e-4,
            "rel_tol": 1e-5,
            "line_search_flag": False,
        },
    },
]


def solver_attempt(problem, initial_guess, solver_candidate):
    solver_options = dict(solver_candidate["options"])
    solver_options["initial_guess"] = initial_guess
    return solver(problem, solver_options=solver_options)


def solve_one_increment(problem, state, target_disp, target_time, dirichlet_bc_info, compression_bc_index):
    problem.dt = max(target_time - state.time_value, 1e-12)
    dirichlet_bc_info[2][compression_bc_index] = compression_value_fn(target_disp)
    problem.fes[0].update_Dirichlet_boundary_conditions(dirichlet_bc_info)
    problem.set_params(state.params)

    last_error = None
    for cand in SOLVER_CANDIDATES:
        try:
            sol_list_new = solver_attempt(problem, state.sol_list, cand)
            sol0 = onp.asarray(jax.device_get(sol_list_new[0]))
            if onp.any(~onp.isfinite(sol0)):
                raise RuntimeError(f"non-finite displacement field from {cand['name']}")
            params_new = problem.update_int_vars_gp(sol_list_new[0], state.params)
            alpha_new = onp.asarray(jax.device_get(params_new[1]))
            if onp.any(~onp.isfinite(alpha_new)) or onp.any(alpha_new < -1e-10):
                raise RuntimeError("invalid eqps after constitutive update")
            return SimulationState(target_disp, target_time, sol_list_new, params_new)
        except Exception as exc:
            print(f"[WARN] solver {cand['name']} failed at disp={target_disp:.6e}: {exc}", flush=True)
            last_error = exc
    raise RuntimeError(f"all solver candidates failed; last error: {last_error}")


def advance_with_substepping(problem, state, target_disp, target_time, dirichlet_bc_info, compression_bc_index, max_subdivisions, level=0):
    try:
        new_state = solve_one_increment(problem, state, target_disp, target_time, dirichlet_bc_info, compression_bc_index)
        return new_state, 1
    except Exception as exc:
        if level >= max_subdivisions:
            raise RuntimeError(
                f"increment {state.disp:.6f} -> {target_disp:.6f} mm failed after subdivision level "
                f"{max_subdivisions}; last error: {exc}"
            )
        mid_disp = 0.5 * (state.disp + target_disp)
        mid_time = 0.5 * (state.time_value + target_time)
        mid_state, n1 = advance_with_substepping(
            problem, state, mid_disp, mid_time, dirichlet_bc_info, compression_bc_index, max_subdivisions, level + 1
        )
        end_state, n2 = advance_with_substepping(
            problem, mid_state, target_disp, target_time, dirichlet_bc_info, compression_bc_index, max_subdivisions, level + 1
        )
        return end_state, n1 + n2


def unique_monotone_xy(x, y):
    x = onp.asarray(x, dtype=float)
    y = onp.asarray(y, dtype=float)
    mask = onp.isfinite(x) & onp.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = onp.argsort(x)
    x = x[order]
    y = y[order]
    if len(x) == 0:
        return x, y
    xu = [float(x[0])]
    yu = [float(y[0])]
    for xi, yi in zip(x[1:], y[1:]):
        if xi > xu[-1] + 1e-12:
            xu.append(float(xi))
            yu.append(float(yi))
        else:
            # For duplicated strain, use latest stress value.
            yu[-1] = float(yi)
    return onp.asarray(xu), onp.asarray(yu)


def score_simulation(sim_strain, sim_stress_MPa, exp_target):
    sim_strain, sim_stress_MPa = unique_monotone_xy(sim_strain, sim_stress_MPa)
    if len(sim_strain) < 2:
        raise RuntimeError("not enough simulation points to score")

    strain = exp_target["full_grid_strain"]
    exp_mean = exp_target["stress_mean_MPa"]
    exp_min = exp_target["stress_min_MPa"]
    exp_max = exp_target["stress_max_MPa"]
    sim_interp = onp.interp(strain, sim_strain, sim_stress_MPa)

    scale = max(float(onp.max(onp.abs(exp_mean))), 1e-6)
    rmse = float(onp.sqrt(onp.mean((sim_interp - exp_mean) ** 2)) / scale)
    mae = float(onp.mean(onp.abs(sim_interp - exp_mean)) / scale)

    low_penalty = onp.maximum(exp_min - sim_interp, 0.0)
    high_penalty = onp.maximum(sim_interp - exp_max, 0.0)
    envelope_penalty = float(onp.mean(low_penalty + high_penalty) / scale)

    slope_sim = onp.gradient(sim_interp, strain)
    slope_exp = onp.gradient(exp_mean, strain)
    slope_scale = max(float(onp.max(onp.abs(slope_exp))), 1e-6)
    slope_rmse = float(onp.sqrt(onp.mean((slope_sim - slope_exp) ** 2)) / slope_scale)

    peak_stress_error = abs(float(sim_interp[-1]) - float(exp_mean[-1])) / scale
    score = rmse + 0.75 * envelope_penalty + 0.25 * slope_rmse + 0.25 * peak_stress_error + 0.10 * mae
    return {
        "score": float(score),
        "rmse_norm": float(rmse),
        "mae_norm": float(mae),
        "envelope_penalty": float(envelope_penalty),
        "slope_rmse_norm": float(slope_rmse),
        "end_stress_error_norm": float(peak_stress_error),
        "sim_end_stress_MPa": float(sim_interp[-1]),
        "exp_end_stress_MPa": float(exp_mean[-1]),
    }


def apply_apparent_correction(sim_strain, sim_stress, stress_scale=1.0, strain_compliance_per_MPa=0.0):
    sim_strain = onp.asarray(sim_strain, dtype=float)
    sim_stress = onp.asarray(sim_stress, dtype=float)
    sigma = stress_scale * sim_stress
    eps = sim_strain + strain_compliance_per_MPa * onp.maximum(sigma, 0.0)
    eps, sigma = unique_monotone_xy(eps, sigma)
    return eps, sigma


def fit_apparent_correction(args, sim_strain, sim_stress, exp_target, raw_score):
    mode = args.correction_mode

    if not args.post_fit_correction or mode is None:
        return ApparentCorrection(False, "none", 1.0, 0.0, raw_score, raw_score)

    bounds = []
    names = []
    if mode in ("stress_scale", "compliance_stress_scale"):
        lo = min(args.stress_scale_min, args.stress_scale_max)
        hi = max(args.stress_scale_min, args.stress_scale_max)
        bounds.append((lo, hi))
        names.append("stress_scale")
    if mode in ("compliance", "compliance_stress_scale"):
        lo = min(args.strain_compliance_min, args.strain_compliance_max)
        hi = max(args.strain_compliance_min, args.strain_compliance_max)
        bounds.append((lo, hi))
        names.append("strain_compliance_per_MPa")

    exp_strain = exp_target["full_grid_strain"]
    exp_mean = exp_target["stress_mean_MPa"]
    scale = max(float(onp.max(onp.abs(exp_mean))), 1e-6)

    # Stronger low-strain weight because the current over-stiff error is usually
    # most obvious at the beginning of compression.
    eps_max = max(float(exp_strain[-1]), 1e-12)
    weights = 1.0 + 1.0 * onp.exp(-exp_strain / max(0.012 * eps_max / 0.05, 1e-6))

    def unpack(x):
        stress_scale = 1.0
        comp = 0.0
        for name, val in zip(names, x):
            if name == "stress_scale":
                stress_scale = float(val)
            elif name == "strain_compliance_per_MPa":
                comp = float(val)
        return stress_scale, comp

    def objective(x):
        stress_scale, comp = unpack(x)
        eps_corr, sig_corr = apply_apparent_correction(sim_strain, sim_stress, stress_scale, comp)
        pred = onp.interp(exp_strain, eps_corr, sig_corr)
        err = (pred - exp_mean) / scale
        rmse = onp.sqrt(onp.mean(weights * err ** 2) / onp.mean(weights))
        end_pen = abs(pred[-1] - exp_mean[-1]) / scale
        area_pen = abs(onp.trapz(pred - exp_mean, exp_strain)) / max(onp.trapz(onp.abs(exp_mean), exp_strain), 1e-9)
        return float(rmse + 0.25 * end_pen + 0.10 * area_pen)

    if len(bounds) == 0:
        return ApparentCorrection(False, "none", 1.0, 0.0, raw_score, raw_score)

    try:
        from scipy.optimize import differential_evolution, minimize
        result = differential_evolution(
            objective,
            bounds=bounds,
            tol=1e-5,
            polish=False,
            seed=123,
            maxiter=80,
            popsize=10,
            workers=1,
        )
        result2 = minimize(objective, result.x, method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-8, "fatol": 1e-8})
        xbest = result2.x if result2.fun <= result.fun else result.x
    except Exception as exc:
        print(f"[WARN] scipy optimizer failed for apparent correction: {exc}; using grid fallback", flush=True)
        grids = []
        for lo, hi in bounds:
            grids.append(onp.linspace(lo, hi, 81))
        best_val = onp.inf
        xbest = None
        if len(grids) == 1:
            for a in grids[0]:
                val = objective([a])
                if val < best_val:
                    best_val = val
                    xbest = onp.array([a])
        else:
            for a in grids[0]:
                for b in grids[1]:
                    val = objective([a, b])
                    if val < best_val:
                        best_val = val
                        xbest = onp.array([a, b])

    stress_scale, comp = unpack(xbest)
    eps_corr, sig_corr = apply_apparent_correction(sim_strain, sim_stress, stress_scale, comp)
    corrected_score = score_simulation(eps_corr, sig_corr, exp_target)
    return ApparentCorrection(True, mode, stress_scale, comp, raw_score, corrected_score)


def build_run_dirs(output_root: str, case_id: str, run_tag: str):
    case_root = os.path.join(output_root, sanitize_name(case_id))
    runs_root = os.path.join(case_root, "runs")
    run_root = os.path.join(runs_root, sanitize_name(run_tag))
    os.makedirs(run_root, exist_ok=False)
    return case_root, run_root


def generate_run_tag(user_tag=None):
    if user_tag:
        return sanitize_name(user_tag)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_pid{os.getpid()}_{uuid.uuid4().hex[:8]}"


def write_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_update_best(case_root: str, candidate_summary: Dict[str, Any], run_root: str):
    os.makedirs(case_root, exist_ok=True)
    lock_path = os.path.join(case_root, "best_fit.lock")
    best_json_path = os.path.join(case_root, "best_fit.json")
    best_dir = os.path.join(case_root, "best")
    os.makedirs(best_dir, exist_ok=True)

    def do_update():
        current_best = None
        if os.path.exists(best_json_path):
            try:
                with open(best_json_path, "r", encoding="utf-8") as f:
                    current_best = json.load(f)
            except Exception:
                current_best = None
        is_better = current_best is None or candidate_summary["score"]["score"] < current_best["score"]["score"]
        if is_better:
            write_json(best_json_path, candidate_summary)
            for filename in [
                "simulation.csv",
                "simulation_corrected.csv",
                "score.json",
                "apparent_correction.json",
                "run_manifest.json",
                "comparison.png",
                "simulation_physical_curve.csv",
            ]:
                src = os.path.join(run_root, filename)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(best_dir, filename))
        return is_better

    if fcntl is None:
        return do_update()

    with open(lock_path, "w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        is_better = do_update()
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    return is_better


def parse_material_params(params_json_arg: Optional[str]):
    params = dict(DEFAULT_MATERIAL_PARAMS)
    if params_json_arg is None:
        return params
    if os.path.isfile(params_json_arg):
        with open(params_json_arg, "r", encoding="utf-8") as f:
            user_params = json.load(f)
    else:
        user_params = json.loads(params_json_arg)
    params.update(user_params)
    return params


def run_simulation(args, meshio_mesh, scale_meta, experiments, exp_target, material_params, case_root, run_root):
    problem, dirichlet_bc_info, bc_meta, diagnostics = initialize_problem(meshio_mesh, material_params, args)
    print(f"[INFO] initial diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}", flush=True)
    if not diagnostics["zero_residual_finite"]:
        raise RuntimeError(f"initial residual is not finite: {diagnostics}")

    target_disp_model = -exp_target["target_strain"] * bc_meta["Lz"]
    target_time_s = exp_target["target_time_s"]
    compression_bc_index = int(bc_meta["compression_bc_index"])

    state = SimulationState(
        disp=0.0,
        time_value=0.0,
        sol_list=[np.zeros((problem.fes[0].num_total_nodes, problem.fes[0].vec))],
        params=problem.internal_vars,
    )

    rows = []
    metrics0 = compute_step_metrics(problem, state.sol_list[0], state.params, bc_meta)
    rows.append({
        "step": 0,
        "disp_model_mm": 0.0,
        "time_s": 0.0,
        "strain": 0.0,
        "substeps": 0,
        **metrics0,
        "wall_time_s": 0.0,
    })

    total_start = time.time()
    n_steps = len(target_disp_model) - 1
    for i in range(1, len(target_disp_model)):
        d = float(target_disp_model[i])
        t = float(target_time_s[i])
        step_start = time.time()
        target_strain_now = abs(d) / max(bc_meta["Lz"], 1e-12)

        print(
            f"\n[STEP {i:03d}/{n_steps:03d}] target_disp={abs(d):.6f} mm | "
            f"target_strain={100.0 * target_strain_now:.3f}%",
            flush=True,
        )
        state, n_substeps = advance_with_substepping(
            problem,
            state,
            d,
            t,
            dirichlet_bc_info,
            compression_bc_index,
            args.max_subdivisions,
        )
        metrics = compute_step_metrics(problem, state.sol_list[0], state.params, bc_meta)
        strain = abs(state.disp) / max(bc_meta["Lz"], 1e-12)
        wall = time.time() - step_start
        rows.append({
            "step": i,
            "disp_model_mm": abs(state.disp),
            "time_s": state.time_value,
            "strain": strain,
            "substeps": n_substeps,
            **metrics,
            "wall_time_s": wall,
        })
        print(
            f"[DONE {i:03d}/{n_steps:03d}] strain={100.0 * strain:.3f}% | "
            f"stress={metrics['apparent_stress_MPa']:.4f} MPa | "
            f"force={metrics['reaction_kN']:.4f} kN | "
            f"eqps_mean={metrics['eqps_mean']:.5e} | eqps_max={metrics['eqps_max']:.5e} | "
            f"substeps={n_substeps} | wall={wall:.2f}s",
            flush=True,
        )
        gc.collect()

    rows_np = onp.array([
        [
            r["step"],
            r["disp_model_mm"],
            r["time_s"],
            r["strain"],
            r["reaction_kN"],
            r["apparent_stress_MPa"],
            r["sigma_bar_zz"],
            r["sigma_bar_xx"],
            r["sigma_bar_yy"],
            r["eqps_mean"],
            r["eqps_max"],
            r["substeps"],
            r["wall_time_s"],
        ]
        for r in rows
    ], dtype=float)

    sim_strain = rows_np[:, 3]
    sim_stress = rows_np[:, 5]
    raw_score = score_simulation(sim_strain, sim_stress, exp_target)
    correction = fit_apparent_correction(args, sim_strain, sim_stress, exp_target, raw_score)
    corr_strain, corr_stress = apply_apparent_correction(
        sim_strain,
        sim_stress,
        correction.stress_scale,
        correction.strain_compliance_per_MPa,
    )

    active_score = correction.corrected_score if correction.enabled else raw_score

    sim_csv_path = os.path.join(run_root, "simulation.csv")
    header = (
        "step,disp_model_mm,time_s,strain,reaction_kN,apparent_stress_MPa,"
        "sigma_bar_zz_MPa,sigma_bar_xx_MPa,sigma_bar_yy_MPa,eqps_mean,eqps_max,substeps,wall_time_s"
    )
    onp.savetxt(sim_csv_path, rows_np, delimiter=",", header=header, comments="")

    corrected_csv = onp.column_stack([corr_strain, corr_stress])
    corrected_csv_path = os.path.join(run_root, "simulation_corrected.csv")
    onp.savetxt(
        corrected_csv_path,
        corrected_csv,
        delimiter=",",
        header="corrected_strain,corrected_stress_MPa",
        comments="",
    )

    physical_disp = corr_strain * args.specimen_height_mm
    physical_force = corr_stress * (args.specimen_width_mm * args.specimen_depth_mm) / 1000.0
    physical_csv = onp.column_stack([corr_strain, corr_stress, physical_disp, physical_force])
    physical_csv_path = os.path.join(run_root, "simulation_physical_curve.csv")
    onp.savetxt(
        physical_csv_path,
        physical_csv,
        delimiter=",",
        header="strain,stress_MPa,disp_physical_mm,force_physical_kN",
        comments="",
    )

    score_obj = {
        "active_score": active_score,
        "raw_score": raw_score,
        "corrected_score": correction.corrected_score,
    }
    write_json(os.path.join(run_root, "score.json"), score_obj)

    correction_obj = {
        "enabled": correction.enabled,
        "mode": correction.mode,
        "stress_scale": correction.stress_scale,
        "strain_compliance_per_MPa": correction.strain_compliance_per_MPa,
        "formula": "corrected_stress = stress_scale * raw_stress; corrected_strain = raw_strain + strain_compliance_per_MPa * max(corrected_stress, 0)",
        "note": "This is an apparent curve correction for platen/contact/machine compliance and effective stress normalization; it does not change E, nu, G or plasticity parameters inside the FE solve.",
    }
    write_json(os.path.join(run_root, "apparent_correction.json"), correction_obj)

    plot_path = os.path.join(run_root, "comparison.png")
    if args.save_plot:
        plt.figure(figsize=(9, 6))
        plt.plot(exp_target["full_grid_strain"], exp_target["stress_mean_MPa"], linewidth=2.0, label="experiment mean")
        plt.fill_between(
            exp_target["full_grid_strain"],
            exp_target["stress_min_MPa"],
            exp_target["stress_max_MPa"],
            alpha=0.25,
            label="experiment envelope",
        )
        if args.plot_raw_simulation:
            plt.plot(sim_strain, sim_stress, linewidth=1.8, linestyle="--", alpha=0.65, label=f"raw FE ({args.boundary_mode})")
        label = f"simulation corrected ({args.boundary_mode})" if correction.enabled else f"simulation ({args.boundary_mode})"
        plt.plot(corr_strain, corr_stress, linewidth=2.4, label=label)
        plt.xlabel("Engineering strain")
        plt.ylabel("Apparent stress (MPa)")
        plt.title("Compression fitting: stress-strain")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()

    run_manifest = {
        "mesh_file": args.mesh_file,
        "experiment_files": args.experiment_files,
        "case_id": args.case_id,
        "run_root": run_root,
        "material_params": material_params,
        "scale_meta": scale_meta,
        "boundary_mode": args.boundary_mode,
        "diagnostics": diagnostics,
        "bc_meta": {k: v for k, v in bc_meta.items() if k not in {"top_nodes", "bottom_nodes"}},
        "target_num_steps": args.target_num_steps,
        "max_engineering_strain": args.max_engineering_strain,
        "specimen_height_mm": args.specimen_height_mm,
        "specimen_width_mm": args.specimen_width_mm,
        "specimen_depth_mm": args.specimen_depth_mm,
        "experiments": [{k: v for k, v in exp.items() if k not in {"disp_mm", "force_kN", "time_s"}} for exp in experiments],
        "experiment_target": {
            "common_max_disp_mm": exp_target["common_max_disp_mm"],
            "common_max_strain": exp_target["common_max_strain"],
            "end_experiment_stress_MPa": float(exp_target["stress_mean_MPa"][-1]),
        },
        "score": active_score,
        "raw_score": raw_score,
        "corrected_score": correction.corrected_score,
        "apparent_correction": correction_obj,
        "elapsed_total_s": time.time() - total_start,
    }
    write_json(os.path.join(run_root, "run_manifest.json"), run_manifest)

    print("\n[SUMMARY]", flush=True)
    print(f"  raw score       : {raw_score['score']:.6f}", flush=True)
    if correction.enabled:
        print(f"  corrected score : {correction.corrected_score['score']:.6f}", flush=True)
        print(f"  stress_scale    : {correction.stress_scale:.6f}", flush=True)
        print(f"  strain_comp     : {correction.strain_compliance_per_MPa:.8f} strain/MPa", flush=True)
    print(f"  output          : {run_root}", flush=True)

    return rows_np, active_score, run_manifest


def main():
    args = parse_args()
    args.case_id = args.case_id or sanitize_name(os.path.splitext(os.path.basename(args.mesh_file))[0])
    os.makedirs(args.output_root, exist_ok=True)
    run_tag = generate_run_tag(args.run_tag)
    case_root, run_root = build_run_dirs(args.output_root, args.case_id, run_tag)

    experiments = [parse_experiment_txt(p, zero_shift=args.zero_shift_experiment) for p in args.experiment_files]
    exp_target = build_experiment_target(
        experiments,
        num_steps=args.target_num_steps,
        specimen_height_mm=args.specimen_height_mm,
        specimen_area_mm2=args.specimen_width_mm * args.specimen_depth_mm,
        use_experiment_timestamps=args.use_experiment_timestamps,
        max_engineering_strain=args.max_engineering_strain,
    )
    meshio_mesh, scale_meta = load_mesh(
        args.mesh_file,
        args.real_dimension_mm,
        args.dimension_axis,
        args.anisotropic_scale,
        args.no_mesh_scaling,
    )
    material_params = parse_material_params(args.params_json)
    _, score, run_manifest = run_simulation(
        args,
        meshio_mesh,
        scale_meta,
        experiments,
        exp_target,
        material_params,
        case_root,
        run_root,
    )

    summary = {
        "case_id": args.case_id,
        "run_root": run_root,
        "score": score,
        "material_params": material_params,
        "mesh_file": args.mesh_file,
        "experiments": args.experiment_files,
        "scale_meta": scale_meta,
        "apparent_correction": run_manifest.get("apparent_correction", {}),
    }
    improved = safe_update_best(case_root, summary, run_root)
    print("\n" + "=" * 80, flush=True)
    print("RUN FINISHED", flush=True)
    print(f"run_root: {run_root}", flush=True)
    print(f"score: {score['score']:.6f}", flush=True)
    print(f"best updated: {improved}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
