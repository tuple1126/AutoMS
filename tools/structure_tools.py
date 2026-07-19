import os
import sys
import json
import subprocess
import shutil
import numpy as np
import gc
from datetime import datetime
from typing import Dict, List, Any, Optional, Annotated
from tools.tree_tracker import get_global_tracker, get_property_table
from tools.project_paths import (
    DATABASE_CSV_PATH,
    DATABASE_OBJ_DIR,
    MIND_GENERATE_ALL_SCRIPT,
    MIND_GEN_FROM_TRI_SCRIPT,
    MIND_NETWORK_PATH,
    MIND_VALIDATION_DIR,
    PETL_RESULTS_DIR,
    TRIPLANE_WORK_DIR,
)


def clear_gpu_memory():
    """
    Helper that releases GPU memory.

    Call this after a subprocess finishes to release GPU resources.
    """
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    except Exception as e:
        print(f"Error while clearing GPU memory: {e}")

# Base materials definition
BASE_MATERIALS = {
    "TPU": {"E": 50, "G": 20, "nu": 0.48, "name": "TPU (thermoplastic polyurethane)"},
    "PCTPE": {"E": 75, "G": 29, "nu": 0.28, "name": "PCTPE (flexible nylon elastomer)"},
    "Silicone": {"E": 10, "G": 4, "nu": 0.47, "name": "Silicone"},
    "PLA+": {"E": 1200, "G": 450, "nu": 0.34, "name": "PLA+ (modified polylactic acid)"},
    "ABS": {"E": 2400, "G": 870, "nu": 0.37, "name": "ABS (acrylonitrile butadiene styrene)"},
    "Al6061": {"E": 68970, "G": 25900, "nu": 0.33, "name": "Aluminum alloy 6061"},
    "PETG": {"E": 2100, "G": 800,  "nu": 0.38, "name": "PETG (polyethylene terephthalate glycol)"},
    "PA12": {"E": 1700, "G": 650,  "nu": 0.40, "name": "PA12 (nylon 12)"},
    "PC": {"E": 2300, "G": 850,  "nu": 0.37, "name": "PC (polycarbonate)"},
    "HIPS": {"E": 2100, "G": 800,  "nu": 0.35, "name": "HIPS (high-impact polystyrene)"},
    "PVA": {"E": 2000, "G": 750,  "nu": 0.33, "name": "PVA (water-soluble support material)"},
    "PEEK": {"E": 3600, "G": 1350, "nu": 0.40, "name": "PEEK (polyether ether ketone)"},
    "Ti6Al4V": {"E": 113800, "G": 44000, "nu": 0.34, "name": "Ti6Al4V (titanium alloy)"},
    "316L": {"E": 200000, "G": 77000, "nu": 0.30, "name": "316L stainless steel"},
    "Inconel718": {"E": 200000, "G": 78000, "nu": 0.29, "name": "Inconel 718 (nickel-based high-temperature alloy)"},
    "AlSi10Mg": {"E": 69000, "G": 26000, "nu": 0.33, "name": "AlSi10Mg (aluminum alloy for additive manufacturing)"},
}

def load_dict(path):
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        return obj.item()
    raise ValueError("The file is neither a dictionary nor a zero-dimensional object array.")

def to_target_cube(vol, N=64):
    assert vol.ndim == 3 and vol.shape[0] == vol.shape[1] == vol.shape[2], "The voxel grid must be cubic."
    D = vol.shape[0]
    if D == 70 and N == 64:
        return vol[2:66, 2:66, 2:66]
    if D == N:
        return vol
    idx = np.floor(np.linspace(0, D - 1, N)).astype(int)
    return vol[np.ix_(idx, idx, idx)]

def auto_fix_sign(sdfN, occN):
    if occN is None:
        return False
    occN = (occN > 0.5)
    pred_a = (sdfN < 0)
    pred_b = (-sdfN < 0)
    match_a = (pred_a == occN).mean()
    match_b = (pred_b == occN).mean()
    return match_b > match_a

def process_npy_to_csv(npy_path):
    try:
        base = os.path.splitext(os.path.basename(npy_path))[0]
        out_csv = os.path.join(os.path.dirname(npy_path), f"{base}.csv")
        existed = os.path.exists(out_csv)

        d = load_dict(npy_path)
        if "voxel_sdf_pr" not in d:
            print(f"[WARN] Missing voxel_sdf_pr: {os.path.basename(npy_path)}")
            return False

        sdf = np.asarray(d["voxel_sdf_pr"], dtype=np.float32)
        sdfN = to_target_cube(sdf, 64)

        occN_hint = None
        if "voxel_occ_pr" in d and isinstance(d["voxel_occ_pr"], np.ndarray):
            occN_hint = to_target_cube(np.asarray(d["voxel_occ_pr"], dtype=np.float32), 64)

        if auto_fix_sign(sdfN, occN_hint):
            sdfN = -sdfN

        if occN_hint is not None:
            occ = (occN_hint > 0.5).astype(np.uint8)
            src = "occ_hint"
        else:
            occ = (sdfN <= 0.0).astype(np.uint8)
            src = "sdf_sign"

        np.savetxt(out_csv, occ.reshape(-1), fmt="%d")

        tag = "[OVERWRITE]" if existed else "[OK]"
        print(f"{tag} {os.path.basename(npy_path)} -> {os.path.basename(out_csv)}  "
              f"shape:{tuple(occ.shape)}  unique:{np.unique(occ).tolist()}  src:{src}")
        return True

    except Exception as e:
        print(f"[FAIL] {os.path.basename(npy_path)} -> {e}")
        return False

def build_stiffness_matrix(E: float, G: float, nu: float) -> np.ndarray:
    """Build an isotropic conditioning tensor from all paper parameters E, G, nu."""
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    C = np.zeros((6, 6))
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * G
    C[0, 1] = C[1, 0] = C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = lam
    C[3, 3] = C[4, 4] = C[5, 5] = G
    return C


def validate_isotropic_conditioning(E: float, G: float, nu: float, tolerance: float = 0.05) -> float:
    """Enforce the Appendix A.1 isotropic relation E = 2 G (1 + nu)."""
    if not (np.isfinite(E) and np.isfinite(G) and np.isfinite(nu)):
        raise ValueError("E, G, and nu must be finite")
    if E <= 0.0 or G <= 0.0:
        raise ValueError("E and G must be positive")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu must be in (-1, 0.5)")
    implied_E = 2.0 * G * (1.0 + nu)
    relative_error = abs(E - implied_E) / max(abs(E), abs(implied_E), 1.0)
    if relative_error > tolerance:
        raise ValueError(
            "inconsistent isotropic conditioning: "
            f"E={E:.6g}, G={G:.6g}, nu={nu:.6g}, "
            f"but 2*G*(1+nu)={implied_E:.6g} (relative error={relative_error:.2%})"
        )
    return relative_error

def convert_physical_to_mat_params(E: float, nu: float, G: float, custom_base_material: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    C11_RANGE = (0.0004, 0.6209)
    C12_RANGE = (-0.0078, 0.2368)
    C44_RANGE = (0.0, 0.1591)
    isotropic_relative_error = validate_isotropic_conditioning(E, G, nu)

    if custom_base_material is not None:
        base_E = custom_base_material.get('E')
        base_G = custom_base_material.get('G') 
        base_nu = custom_base_material.get('nu')
        base_name = custom_base_material.get('name', 'Custom material')
        
        if base_E is None or base_G is None or base_nu is None:
            return {"error": "Custom base-material parameters are incomplete; E, G, and nu are required."}
        
        if E > base_E or G > base_G:
            print("Warning: target parameters exceed custom base-material bounds")
        
        print(f"Using custom base material: {base_name}")
        print(f"Base-material parameters: E={base_E}, G={base_G}, nu={base_nu}")

        E_norm = E / base_E
        G_norm = G / base_G
        nu_norm = nu
        
        D_local = build_stiffness_matrix(E_norm, G_norm, nu_norm)
        c11, c12, c44 = D_local[0, 0], D_local[0, 1], D_local[3, 3]
        in_range = (C11_RANGE[0] <= c11 <= C11_RANGE[1] and
                   C12_RANGE[0] <= c12 <= C12_RANGE[1] and
                   C44_RANGE[0] <= c44 <= C44_RANGE[1])
        
        E_utilization = E / base_E
        G_utilization = G / base_G
        nu_utilization = nu / base_nu if base_nu != 0 else 0
        avg_utilization = (E_utilization + G_utilization + nu_utilization) / 3
        
        selected = {
            "base_key": "custom",
            "material": {"E": base_E, "G": base_G, "nu": base_nu, "name": base_name},
            "E_norm": E_norm,
            "G_norm": G_norm,
            "nu_norm": nu_norm,
            "c11": c11,
            "c12": c12,
            "c44": c44,
            "in_range": in_range,
            "avg_utilization": avg_utilization,
            "E_utilization": E_utilization,
            "G_utilization": G_utilization,
            "nu_utilization": nu_utilization
        }
    else:
        candidates = []
        for material_key, material_props in BASE_MATERIALS.items():
            if (E <= material_props["E"] and G <= material_props["G"]):
                E_norm = E / material_props["E"]
                G_norm = G / material_props["G"]
                nu_norm = nu
                D_local = build_stiffness_matrix(E_norm, G_norm, nu_norm)
                c11_l, c12_l, c44_l = D_local[0, 0], D_local[0, 1], D_local[3, 3]
                in_range = (C11_RANGE[0] <= c11_l <= C11_RANGE[1] and
                           C12_RANGE[0] <= c12_l <= C12_RANGE[1] and
                           C44_RANGE[0] <= c44_l <= C44_RANGE[1])
                E_utilization = E / material_props["E"]
                G_utilization = G / material_props["G"]
                nu_utilization = nu / material_props["nu"]
                avg_utilization = (E_utilization + G_utilization + nu_utilization) / 3
                candidates.append({
                    "base_key": material_key,
                    "material": material_props,
                    "E_norm": E_norm,
                    "G_norm": G_norm,
                    "nu_norm": nu_norm,
                    "c11": c11_l,
                    "c12": c12_l,
                    "c44": c44_l,
                    "in_range": in_range,
                    "avg_utilization": avg_utilization,
                    "E_utilization": E_utilization,
                    "G_utilization": G_utilization,
                    "nu_utilization": nu_utilization
                })
        
        valid_candidates = [c for c in candidates if c["in_range"]]
        if valid_candidates:
            selected = max(valid_candidates, key=lambda x: x["avg_utilization"])
        else:
            def distance_to_range(val, r):
                if val < r[0]: return r[0] - val
                if val > r[1]: return val - r[1]
                return 0.0
            
            if candidates:
                for c in candidates:
                    dist = (distance_to_range(c["c11"], C11_RANGE) +
                           distance_to_range(c["c12"], C12_RANGE) +
                           distance_to_range(c["c44"], C44_RANGE))
                    c["distance"] = dist
                selected = min(candidates, key=lambda x: x["distance"])
            else:
                selected = {
                    "base_key": "Al6061",
                    "material": BASE_MATERIALS["Al6061"],
                    "E_norm": E / BASE_MATERIALS["Al6061"]["E"],
                    "G_norm": G / BASE_MATERIALS["Al6061"]["G"],
                    "nu_norm": nu,
                    "c11": 0.0, "c12": 0.0, "c44": 0.0,
                    "in_range": False, "avg_utilization": 0.0
                }

    base_material = selected["material"]
    E_normalized = selected["E_norm"]
    G_normalized = selected["G_norm"]
    nu_normalized = selected["nu_norm"]
    c11, c12, c44 = selected["c11"], selected["c12"], selected["c44"]

    print("  Stiffness-matrix range check:")
    print(f"    Target ranges: C11{C11_RANGE}, C12{C12_RANGE}, C44{C44_RANGE}")
    print(f"    Selected base material: {base_material['name']} (key={selected['base_key']})")
    print(f"    Normalization: E_norm={E_normalized:.6f}, G_norm={G_normalized:.6f}, nu={nu_normalized:.6f}")
    print(f"    Computed stiffness: C11={c11:.6f}, C12={c12:.6f}, C44={c44:.6f} (in_range={selected['in_range']})")
    print(f"    Utilization: E={selected['E_utilization']:.3f}, G={selected['G_utilization']:.3f}, nu={selected['nu_utilization']:.3f}")

    prop1 = c11 * 1.2
    prop2 = (c12 + 0.01) * 5.0
    prop3 = c44 * 5.0

    print("  Step 3 - material-parameter conversion:")
    print(f"    prop1(mat[0][0])={prop1:.6f}, prop2(mat[0][1])={prop2:.6f}, prop3(mat[3][3])={prop3:.6f}")

    return {
        "prop1": prop1,
        "prop2": prop2,
        "prop3": prop3,
        "original_E": E,
        "original_nu": nu,
        "original_G": G,
        "selected_base_material": {
            **base_material,
            "key": selected["base_key"],
        },
        "E_normalized": E_normalized,
        "G_normalized": G_normalized,
        "isotropic_relative_error": isotropic_relative_error,
        "c11": c11,
        "c12": c12,
        "c44": c44,
        "within_target_range": selected["in_range"],
        "stiffness_target_ranges": {
            "C11": C11_RANGE,
            "C12": C12_RANGE,
            "C44": C44_RANGE
        },
        "adjustment_note": (
            "Target ranges satisfied" if selected["in_range"] else
            "Target ranges were not fully satisfied; the closest base material was selected. Consider adjusting the input physical parameters."
        )
    }

def generate_microstructure_with_ai(
    E: float,
    nu: float,
    G: float,
    custom_base_material,  # Required parameter: base-material information is mandatory.
    custom_name: str = "ai_generated_structures"
) -> Dict[str, Any]:
    """
    Generate microstructures from physical material parameters.

    Args:
        E: Target Young's modulus (MPa).
        nu: Target Poisson's ratio.
        G: Target shear modulus (MPa).
        custom_base_material: Required base-material dictionary containing E, G,
            nu, and name. Example: {"E": 68970, "G": 25900, "nu": 0.33,
            "name": "Al6061", "k": 167}.
        custom_name: Custom generation-batch name.
    """
    # ========== 1. Validate required custom_base_material ==========
    if custom_base_material is None:
        return {
            "error": "Missing required parameter: custom_base_material.",
            "hint": "Provide base-material parameters, for example: custom_base_material={'E': 68970, 'G': 25900, 'nu': 0.33, 'name': 'Al6061', 'k': 167}.",
            "common_materials": {
                "Al6061": {"E": 68970, "G": 25900, "nu": 0.33, "k": 167},
                "Ti6Al4V": {"E": 113800, "G": 44000, "nu": 0.342, "k": 6.7},
                "SS316L": {"E": 200000, "G": 78000, "nu": 0.3, "k": 16}
            }
        }
    
    # Parse custom_base_material; strings and dictionaries are supported.
    parsed_base_material = None
    if isinstance(custom_base_material, str):
        try:
            import json
            parsed_base_material = json.loads(custom_base_material)
        except json.JSONDecodeError:
            return {"error": f"Could not parse custom_base_material string as JSON: {custom_base_material}"}
    elif isinstance(custom_base_material, dict):
        parsed_base_material = custom_base_material
    else:
        return {"error": f"custom_base_material must be a string or dictionary; received {type(custom_base_material)}"}
    
    # Validate base-material parameter completeness.
    required_fields = ['E', 'G', 'nu', 'name']
    missing_fields = [f for f in required_fields if f not in parsed_base_material]
    if missing_fields:
        return {
            "error": f"custom_base_material is missing required fields: {missing_fields}",
            "hint": f"Required fields: {required_fields}",
            "received": parsed_base_material
        }
    
    # ========== 2. Validate target parameters ==========
    if E <= 0: return {"error": f"Young's modulus must be positive: {E}"}
    if not (0.0 < nu < 0.5): return {"error": f"Poisson's ratio must be in (0.0, 0.5): {nu}"}
    if G <= 0: return {"error": f"Shear modulus must be positive: {G}"}
    
    # Ensure target parameters do not exceed base-material parameters.
    base_E = parsed_base_material['E']
    base_G = parsed_base_material['G']
    if E > base_E:
        return {"error": f"Target Young's modulus E={E} exceeds base material E={base_E}; lower the target or choose another base material."}
    if G > base_G:
        return {"error": f"Target shear modulus G={G} exceeds base material G={base_G}; lower the target or choose another base material."}
    
    base_name = parsed_base_material.get('name', 'Custom')
    print(f"Starting microstructure generation - target parameters: E={E}, G={G}, nu={nu}")
    print(f"  Base material: {base_name} (E={base_E}, G={base_G}, nu={parsed_base_material['nu']})")
    print("  Step 1 - physical-parameter validation:")
    
    try:
        conversion_result = convert_physical_to_mat_params(E, nu, G, parsed_base_material)
    except ValueError as exc:
        return {"error": str(exc)}
    prop1 = conversion_result["prop1"]
    prop2 = conversion_result["prop2"] 
    prop3 = conversion_result["prop3"]
        # Stop generation if the stiffness matrix falls outside target ranges.
    if not conversion_result["within_target_range"]:
        return {
            "error": "The stiffness matrix is outside the target range; adjust the parameters and try again.",
            "adjustment_note": conversion_result["adjustment_note"],
            "stiffness_values": {
                "c11": conversion_result["c11"],
                "c12": conversion_result["c12"],
                "c44": conversion_result["c44"]
            },
            "target_ranges": conversion_result["stiffness_target_ranges"]
        }
    
    print("  Step 2 - diffusion-model generation:")
    print(f"    Input conditions: prop1={prop1:.6f}, prop2={prop2:.6f}, prop3={prop3:.6f}")

    # In wo_dual_source ablation mode, generate ten structures.
    ablation_dual_source = os.environ.get("CHATMS_ABLATION_DUAL_SOURCE", "0") == "1"
    each_test_num = 10 if ablation_dual_source else 5
    print(f"    Ablation mode: {'wo_dual_source' if ablation_dual_source else 'full'}, generated structures: {each_test_num}")

    generate_all_script_path = str(MIND_GENERATE_ALL_SCRIPT)
    network_path = str(MIND_NETWORK_PATH)
    outdir = str(PETL_RESULTS_DIR)
    work_dir = str(TRIPLANE_WORK_DIR)
    nfd_vali_dir = str(MIND_VALIDATION_DIR)
    gen_from_tri_script = str(MIND_GEN_FROM_TRI_SCRIPT)

    missing_mind_resources = [
        path for path in (
            generate_all_script_path,
            network_path,
            work_dir,
            nfd_vali_dir,
            gen_from_tri_script,
        )
        if not os.path.exists(path)
    ]
    if missing_mind_resources:
        return {
            "error": "MIND resources are missing",
            "missing_paths": missing_mind_resources,
        }
    
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        generate_all_script_path,
        "--steps=32",
        "--sigma_max=80",
        "--batch=1024",
        f"--network={network_path}",
        f"--outdir={outdir}",
        "--cond_strength=7",
        f"--each_test_num={each_test_num}",
        f"--prop1={prop1}",
        f"--prop2={prop2}",
        f"--prop3={prop3}",
        f"--custom_name={custom_name}",
    ]
    command_env = os.environ.copy()
    command_env.setdefault("OMP_NUM_THREADS", "8")
    
    print(f"    Executing command: {command}")
    
    try:
        result = subprocess.run(
            command,
            check=True, 
            cwd=work_dir,
            env=command_env,
            capture_output=True,
            text=True
        )
        print("Step 1 complete: AI model generation finished")
        # Clear GPU memory after AI-model generation.
        clear_gpu_memory()
        print("GPU memory cleared after AI generation")
        # return {"status": "success", "message": "AI generation completed", "conversion_result": conversion_result} # Removed early return

        # Step 2: run validation and post-processing scripts.
        print("Starting step 2: running validation script...")
        
        # Run gen_from_tri.sh.
        gen_result = subprocess.run(
            ["bash", gen_from_tri_script],
            check=True, 
            cwd=nfd_vali_dir,
            capture_output=True,
            text=True
        )
        print("gen_from_tri.sh completed")
        # Clear GPU memory after gen_from_tri.sh.
        clear_gpu_memory()
        print("GPU memory cleared after validation script")
        print("=" * 50)

        # Step 3: post-process with run_stiffness_analysis for homogenized-property filtering.
        # Use stiffness_analysis_tool.py consistently with the database retrieval path.
        # Supply physical base-material parameters so results are absolute values.
        print("Starting step 3: filtering OBJ files by homogenized properties...")

        # Get base-material parameters.
        base_material = conversion_result["selected_base_material"]
        base_E = base_material["E"]
        base_nu = base_material["nu"]
        base_G = base_material["G"]

        # Use input physical material parameters as target properties.
        E_target = E
        nu_target = nu
        G_target = G

        print(f"  Base material: {base_material.get('name', 'Custom')} (E={base_E}, G={base_G}, nu={base_nu})")
        print(f"  Target elastic properties: E={E_target:.2f}, G={G_target:.2f}, nu={nu_target:.4f}")

        # Get the workshop directory.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        workshop_dir = os.path.join(project_root, "data", "workshop")
        os.makedirs(workshop_dir, exist_ok=True)

        # Step 3.1: preprocess NPY -> CSV -> OBJ and ensure the workshop has OBJ files.
        npy_files = []
        if os.path.exists(workshop_dir):
            for filename in os.listdir(workshop_dir):
                if filename.endswith('.npy') and custom_name in filename:
                    npy_files.append(os.path.join(workshop_dir, filename))

        if npy_files:
            print(f"  Found {len(npy_files)} NPY files for the current batch")
            # Convert NPY files to CSV for later simulation workflows when needed.
            csv_count = 0
            for npy_file in npy_files:
                if process_npy_to_csv(npy_file):
                    csv_count += 1
            print(f"  Converted {csv_count}/{len(npy_files)} NPY files to CSV")

        # Step 3.2: filter OBJ files with run_stiffness_analysis.
        # Match the database retrieval path: physical base material -> absolute values.
        from tools.stiffness_analysis_tool import run_stiffness_analysis

        obj_files = []
        if os.path.exists(workshop_dir):
            for filename in os.listdir(workshop_dir):
                if filename.endswith('.obj') and custom_name in filename:
                    obj_files.append(os.path.join(workshop_dir, filename))

        print(f"  Found {len(obj_files)} OBJ files for homogenized-property filtering")

        analyzed_structures = []
        failed_count = 0
        file_details = []

        for obj_path in obj_files:
            filename = os.path.basename(obj_path)
            print(f"  Analyzing: {filename}")

            try:
                analysis = run_stiffness_analysis(
                    obj_file=obj_path,
                    resolution=64,
                    youngs_modulus=base_E,
                    poisson_ratio=base_nu,
                    device='cuda:0',
                    silent=True
                )

                if not analysis.get("success"):
                    print(f"    Analysis failed: {analysis.get('error', 'Unknown')}")
                    failed_count += 1
                    continue

                props = analysis.get("effective_properties", {})
                E_actual = props.get("E_avg", 0)
                G_actual = props.get("G_avg", 0)
                nu_actual = props.get("nu_avg", 0)

                # Compare relative errors directly because results are physical values.
                E_error = abs(E_actual - E_target) / E_target if E_target != 0 else 0
                G_error = abs(G_actual - G_target) / G_target if G_target != 0 else 0
                nu_error = abs(nu_actual - nu_target) / nu_target if nu_target != 0 else 0

                # Compute total deviation.
                total_deviation = 0.0
                dev_count = 0
                if E_target > 0:
                    total_deviation += E_error
                    dev_count += 1
                if G_target > 0:
                    total_deviation += G_error
                    dev_count += 1
                if nu_target > 0:
                    total_deviation += nu_error
                    dev_count += 1
                if dev_count > 1:
                    total_deviation /= dev_count

                print(f"    Homogenized result: E={E_actual:.2f}, G={G_actual:.2f}, nu={nu_actual:.4f}")
                print(f"    Target properties:   E={E_target:.2f}, G={G_target:.2f}, nu={nu_target:.4f}")
                print(f"    Relative errors:     E={E_error:.1%}, G={G_error:.1%}, nu={nu_error:.1%}")

                # Check whether errors are within 20 percent.
                max_error = max(E_error, nu_error)
                kept = max_error <= 0.20

                file_detail = {
                    "filename": filename,
                    "filepath": obj_path,
                    "scaled_properties": {
                        "E": round(E_actual, 6),
                        "G": round(G_actual, 6),
                        "nu": round(nu_actual, 6)
                    },
                    "target_properties": {
                        "E": round(E_target, 6),
                        "G": round(G_target, 6),
                        "nu": round(nu_target, 6)
                    },
                    "relative_errors": {
                        "E": round(E_error, 3),
                        "G": round(G_error, 3),
                        "nu": round(nu_error, 3)
                    },
                    "total_deviation": total_deviation,
                    "kept": kept
                }
                file_details.append(file_detail)

                if kept:
                    analyzed_structures.append(file_detail)
                    print(f"    Result: kept (maximum error: {max_error:.1%})")
                else:
                    # Remove every file related to this microstructure (OBJ, NPY, CSV, and so on).
                    base_name = os.path.splitext(filename)[0]
                    deleted_count = 0
                    try:
                        for f in os.listdir(workshop_dir):
                            if f.startswith(base_name + '.'):
                                file_path = os.path.join(workshop_dir, f)
                                try:
                                    os.remove(file_path)
                                    deleted_count += 1
                                except Exception as e:
                                    print(f"    Failed to remove {f}: {e}")
                    except Exception as e:
                        print(f"    Error while removing files: {e}")
                    print(f"    Result: removed (maximum error: {max_error:.1%}, removed {deleted_count} related files)")

                print("-" * 60)

                # Clear GPU memory.
                clear_gpu_memory()

            except Exception as e:
                print(f"    Exception: {str(e)}")
                failed_count += 1

        # Step 3.3: rank by deviation and keep the best five, consistent with database retrieval.
        if analyzed_structures:
            analyzed_structures.sort(key=lambda x: x["total_deviation"])
            keep_limit = min(5, len(analyzed_structures))

            print(f"\n  [Quality filtering] Keeping the top {keep_limit} of {len(analyzed_structures)} structures:")
            for i, s in enumerate(analyzed_structures[:keep_limit], 1):
                print(f"    #{i}: {s['filename']} (E={s['scaled_properties']['E']:.1f}, G={s['scaled_properties']['G']:.1f}, "
                      f"nu={s['scaled_properties']['nu']:.3f}, deviation={s['total_deviation']:.3f})")

            # Remove files for structures that were not retained.
            removed_count = 0
            for s in analyzed_structures[keep_limit:]:
                s["kept"] = False
                filepath = s.get("filepath", "")
                if filepath:
                    base_name = os.path.splitext(os.path.basename(filepath))[0]
                    for f in os.listdir(workshop_dir):
                        if f.startswith(base_name + '.'):
                            try:
                                os.remove(os.path.join(workshop_dir, f))
                                removed_count += 1
                            except Exception:
                                pass

            analyzed_structures = analyzed_structures[:keep_limit]
            print(f"  Removed {removed_count} low-quality microstructure files")
        
        print(f"  Filtering complete: kept {len(analyzed_structures)}, failed {failed_count}")

        print("=" * 50)

        # Clear temporary files from the generation directory.
        try:
            cleanup_dir = str(PETL_RESULTS_DIR)
            if os.path.exists(cleanup_dir):
                for item in os.listdir(cleanup_dir):
                    item_path = os.path.join(cleanup_dir, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"    Failed to clear {item}: {e}")
                print(f"    Cleared generation directory: {cleanup_dir}")
            else:
                print(f"    Note: generation directory does not exist: {cleanup_dir}")
        except Exception as e:
            print(f"    Error while clearing generation directory: {e}")
        print("  Step 4 - result validation:")
        print(f"    Generation completed: {workshop_dir}")
        print("    Structure files saved")

        kept_details = analyzed_structures  # Use the filtered structure list directly.
        
        # Record to tracker
        tracker = get_global_tracker()
        if tracker:
            tracker.add_node("generation_artifact", "AI Generated Structures", {
                "parameters": {"E": E, "nu": nu, "G": G, "custom_base_material": custom_base_material},
                "count": len(kept_details),
                "details": kept_details,
                "directory": workshop_dir
            })

        # Build the return result.
        generation_result = {
            "status": "success, no need to repeatedly generate microstructures",
            "message": f"AI generation completed. {len(kept_details)} structures met the criteria.", 
            "kept_structures_details": kept_details,
            "microstructure_directory": workshop_dir
        }
        
        # Update the microstructure property table with mechanical properties from AI generation.
        property_table = get_property_table()
        if property_table and kept_details:
            property_table.update_from_structure_generation(generation_result)
            print("  Step 5 - property-table update:")
            print(f"    Recorded mechanical properties for {len(kept_details)} microstructures in the property table")



        # Inject AI-generated candidates into the SAES population.
        try:
            from tools.saes_integration import is_saes_enabled, get_saes_integrator
            
            if is_saes_enabled():
                integrator = get_saes_integrator()
                integrator.inject_ai_solution(generation_result)
                print("  Step 6 - inject unverified SAES candidates:")
                print(f"    Injected {len(kept_details)} AI-generated candidates; they join Pareto selection after simulation feedback")
                
                # Compute and display the current Pareto-front status.
        except ImportError:
            pass  # SAES module is not available.
        except Exception as e:
            print(f"  [SAES] Error injecting AI-generated solutions: {e}")

        return generation_result

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        # Clear GPU memory on failure as well.
        clear_gpu_memory()
        return {"error": str(e), "output": e.output, "stderr": e.stderr}
    except Exception as e:
        # Clear GPU memory on failure as well.
        clear_gpu_memory()
        return {"error": str(e)}

# Add tool info
generate_microstructure_with_ai.tool_info = {
    "tool_name": "generate_microstructure_with_ai",
    "tool_title": "AI Microstructure Generation",
    "tool_description": "Generate microstructure candidates from physical material parameters. custom_base_material is required.",
    "tool_params": [
        {"name": "E", "description": "Target Young's modulus (MPa); must not exceed the base material's Young's modulus.", "type": "number", "required": True},
        {"name": "nu", "description": "Target Poisson's ratio (0.0-0.5).", "type": "number", "required": True},
        {"name": "G", "description": "Target shear modulus (MPa); must not exceed the base material's shear modulus.", "type": "number", "required": True},
        {"name": "custom_base_material", "description": "Required base-material dictionary containing E, G, nu, and name. Example: {'E': 68970, 'G': 25900, 'nu': 0.33, 'name': 'Al6061', 'k': 167}.", "type": "object", "required": True},
        {"name": "custom_name", "description": "Custom generation-batch name.", "type": "string", "required": False, "default": "ai_generated_structures"}
    ]
}


# ============================================================================
# Database microstructure retrieval.
# ============================================================================

# Database path constants.
DB_CSV_PATH = str(DATABASE_CSV_PATH)
DB_OBJ_DIR = str(DATABASE_OBJ_DIR)

def retrieve_microstructure_from_database(
    E: float,
    nu: float,
    G: float,
    custom_base_material,  # Required parameter: base-material information is mandatory.
    vof_min: float = 0.10,
    vof_max: float = 0.50,
    top_k: int = 15,
    custom_name: str = "db_retrieved_structures"
) -> Dict[str, Any]:
    """
    Retrieve and filter microstructures from the database using homogenized
    property calculations.

    Read the microstructure CSV database, filter candidates by target mechanical
    properties (E, G, nu) and volume-fraction bounds, copy OBJ files to the
    workshop, and run homogenization-based quality filtering.

    Args:
        E: Target Young's modulus (MPa).
        nu: Target Poisson's ratio.
        G: Target shear modulus (MPa).
        custom_base_material: Required base-material dictionary containing E, G,
            nu, and name. Example: {"E": 68970, "G": 25900, "nu": 0.33,
            "name": "Al6061", "k": 167}.
        vof_min: Lower target volume-fraction bound (default: 0.10).
        vof_max: Upper target volume-fraction bound (default: 0.50).
        top_k: Number of database candidates to retrieve (default: 15).
        custom_name: Custom retrieval-batch name.
    
    Returns:
        Dictionary containing retrieval and property-filtering results.
    """
    missing_db_resources = [
        path for path in (DB_CSV_PATH, DB_OBJ_DIR)
        if not os.path.exists(path)
    ]
    if missing_db_resources:
        return {
            "error": "Database resources are missing",
            "missing_paths": missing_db_resources,
        }

    import pandas as pd

    print(f"\n{'='*60}")
    print("[Database Retrieval] Starting microstructure retrieval")
    print(f"{'='*60}")

    # ========== 1. Validate required custom_base_material ==========
    if custom_base_material is None:
        return {
            "error": "Missing required parameter: custom_base_material.",
            "hint": "Provide base-material parameters, for example: custom_base_material={'E': 68970, 'G': 25900, 'nu': 0.33, 'name': 'Al6061', 'k': 167}."
        }

    # Parse custom_base_material; strings and dictionaries are supported.
    parsed_base_material = None
    if isinstance(custom_base_material, str):
        try:
            parsed_base_material = json.loads(custom_base_material)
        except json.JSONDecodeError:
            return {"error": f"Could not parse custom_base_material string as JSON: {custom_base_material}"}
    elif isinstance(custom_base_material, dict):
        parsed_base_material = custom_base_material
    else:
        return {"error": f"custom_base_material must be a string or dictionary; received {type(custom_base_material)}"}

    required_fields = ['E', 'G', 'nu', 'name']
    missing_fields = [f for f in required_fields if f not in parsed_base_material]
    if missing_fields:
        return {"error": f"custom_base_material is missing required fields: {missing_fields}"}

    base_E = parsed_base_material['E']
    base_G = parsed_base_material['G']
    base_nu = parsed_base_material['nu']
    base_name = parsed_base_material.get('name', 'Custom')

    print(f"  Target parameters: E={E} MPa, G={G} MPa, nu={nu}")
    print(f"  Base material: {base_name} (E={base_E}, G={base_G}, nu={base_nu})")
    print(f"  Volume-fraction range: [{vof_min}, {vof_max}]")

    # ========== 2. Read and process the CSV database ==========
    if not os.path.exists(DB_CSV_PATH):
        return {"error": f"Database CSV file does not exist: {DB_CSV_PATH}"}

    try:
        df = pd.read_csv(DB_CSV_PATH)
    except Exception as e:
        return {"error": f"Failed to read CSV: {str(e)}"}

    # Clean data.
    for col in ['E', 'G', 'nu', 'vof']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['E', 'G', 'nu', 'vof'])

    print(f"  Total database records: {len(df)}")

    # ========== 3. Scale and filter data ==========
    # E and G in the CSV are normalized to E_base=1 and must be scaled by the target base material.
    df['E_real'] = df['E'] * base_E
    df['G_real'] = df['G'] * base_E  # G is also scaled by E_base for homogenized properties.
    df['nu_real'] = df['nu']  # Poisson's ratio needs no scaling.

    # Filter by volume fraction, the primary condition.
    df_filtered = df[(df['vof'] >= vof_min) & (df['vof'] <= vof_max)].copy()
    print(f"  Records after volume-fraction filter: {len(df_filtered)}")

    if len(df_filtered) == 0:
        # Relax the volume-fraction range.
        expanded_min = max(0.0, vof_min - 0.10)
        expanded_max = min(1.0, vof_max + 0.10)
        df_filtered = df[(df['vof'] >= expanded_min) & (df['vof'] <= expanded_max)].copy()
        print(f"  Relaxed volume-fraction range to [{expanded_min}, {expanded_max}]: {len(df_filtered)} records")

    if len(df_filtered) == 0:
        return {"error": "No database microstructures satisfy the volume-fraction condition."}

    # Compute combined deviation score for multi-objective ranking.
    if E > 0:
        df_filtered['E_dev'] = ((df_filtered['E_real'] - E).abs() / E)
    else:
        df_filtered['E_dev'] = 0.0

    if G > 0:
        df_filtered['G_dev'] = ((df_filtered['G_real'] - G).abs() / G)
    else:
        df_filtered['G_dev'] = 0.0

    if nu > 0:
        df_filtered['nu_dev'] = ((df_filtered['nu_real'] - nu).abs() / nu)
    else:
        df_filtered['nu_dev'] = 0.0

    # Combined score: smaller deviation is better.
    df_filtered['total_score'] = df_filtered['E_dev'] + df_filtered['G_dev'] + df_filtered['nu_dev']

    # Sort by score and take the top_k candidates.
    df_top = df_filtered.nsmallest(top_k, 'total_score')

    print(f"  Selected the top {len(df_top)} best-matching microstructures")

    # ========== 4. Copy OBJ files to the workshop ==========
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    workshop_dir = os.path.join(project_root, "data", "workshop")
    os.makedirs(workshop_dir, exist_ok=True)

    copied_files = []
    for _, row in df_top.iterrows():
        src = os.path.join(DB_OBJ_DIR, f"{row['name']}.obj")
        dst = os.path.join(workshop_dir, f"{row['name']}.obj")
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
                copied_files.append({
                    "name": row['name'],
                    "filepath": dst,
                    "vof": round(row['vof'], 4),
                    "E_real": round(row['E_real'], 2),
                    "G_real": round(row['G_real'], 2),
                    "nu_real": round(row['nu_real'], 4),
                    "total_score": round(row['total_score'], 4)
                })
            except Exception as e:
                print(f"    Failed to copy {row['name']}: {e}")
        else:
            print(f"    OBJ file does not exist: {src}")

    print(f"  Copied {len(copied_files)} OBJ files to {workshop_dir}")

    if not copied_files:
        return {"error": "No OBJ files could be copied."}

    # ========== 5. Homogenized-property calculation and filtering ==========
    print(f"\n[Database Retrieval] Filtering {len(copied_files)} microstructures by homogenized properties...")

    from tools.stiffness_analysis_tool import run_stiffness_analysis

    analyzed_structures = []
    failed_count = 0

    for item in copied_files:
        obj_path = item["filepath"]
        filename = os.path.basename(obj_path)
        print(f"  Analyzing: {filename}")

        try:
            analysis = run_stiffness_analysis(
                obj_file=obj_path,
                resolution=64,
                youngs_modulus=base_E,
                poisson_ratio=base_nu,
                device='cuda:0',
                silent=True
            )

            if not analysis.get("success"):
                print(f"    Analysis failed: {analysis.get('error', 'Unknown')}")
                failed_count += 1
                # Remove the file after a failed analysis.
                if os.path.exists(obj_path):
                    os.remove(obj_path)
                continue

            props = analysis.get("effective_properties", {})
            E_actual = props.get("E_avg", 0)
            G_actual = props.get("G_avg", 0)
            nu_actual = props.get("nu_avg", 0)

            # Compute deviations from the target.
            total_deviation = 0.0
            dev_count = 0
            if E > 0:
                E_dev = abs(E_actual - E) / E
                total_deviation += E_dev
                dev_count += 1
            if G > 0:
                G_dev = abs(G_actual - G) / G
                total_deviation += G_dev
                dev_count += 1
            if nu > 0:
                nu_dev = abs(nu_actual - nu) / nu
                total_deviation += nu_dev
                dev_count += 1
            if dev_count > 1:
                total_deviation /= dev_count

            analyzed_structures.append({
                "filename": filename,
                "filepath": obj_path,
                "E": E_actual,
                "G": G_actual,
                "nu": nu_actual,
                "volume_fraction": item["vof"],
                "total_deviation": total_deviation,
                "scaled_properties": {
                    "E": round(E_actual, 6),
                    "G": round(G_actual, 6),
                    "nu": round(nu_actual, 6)
                },
                "target_properties": {
                    "E": round(E, 6),
                    "G": round(G, 6),
                    "nu": round(nu, 6)
                },
                "relative_errors": {
                    "E": round(abs(E_actual - E) / E if E > 0 else 0, 3),
                    "G": round(abs(G_actual - G) / G if G > 0 else 0, 3),
                    "nu": round(abs(nu_actual - nu) / nu if nu > 0 else 0, 3)
                },
                "kept": True  # Initial marker; later filtering may update it.
            })

            print(f"    E={E_actual:.1f}, G={G_actual:.1f}, nu={nu_actual:.3f}, deviation={total_deviation:.3f}")

            # Clear GPU memory.
            clear_gpu_memory()

        except Exception as e:
            print(f"    Exception: {str(e)}")
            failed_count += 1
            if os.path.exists(obj_path):
                os.remove(obj_path)

    # ========== 6. Sort by deviation and keep the best five ==========
    if analyzed_structures:
        analyzed_structures.sort(key=lambda x: x["total_deviation"])
        keep_limit = min(5, len(analyzed_structures))

        print(f"\n  [Quality filtering] Keeping the top {keep_limit} of {len(analyzed_structures)} structures:")
        for i, s in enumerate(analyzed_structures[:keep_limit], 1):
            print(f"    #{i}: {s['filename']} (E={s['E']:.1f}, G={s['G']:.1f}, nu={s['nu']:.3f}, deviation={s['total_deviation']:.3f})")

        # Remove files that were not retained.
        removed_count = 0
        for s in analyzed_structures[keep_limit:]:
            s["kept"] = False
            if os.path.exists(s["filepath"]):
                os.remove(s["filepath"])
                removed_count += 1

        kept_structures = analyzed_structures[:keep_limit]
        print(f"  Removed {removed_count} low-quality microstructures")
    else:
        kept_structures = []

    # ========== 7. Update property table and tracker ==========
    property_table = get_property_table()
    if property_table and kept_structures:
        for s in kept_structures:
            # Mirror stiffness_analysis output format.
            mock_analysis = {
                "success": True,
                "file": s["filepath"],
                "effective_properties": {
                    "E_avg": s["E"],
                    "G_avg": s["G"],
                    "nu_avg": s["nu"],
                },
                "solid_fraction": s["volume_fraction"]
            }
            property_table.update_from_stiffness_analysis(mock_analysis)
        print(f"  Recorded mechanical properties for {len(kept_structures)} microstructures in the property table")

    tracker = get_global_tracker()
    if tracker:
        tracker.add_node("db_retrieval_artifact", "Database Retrieved Structures", {
            "parameters": {"E": E, "nu": nu, "G": G, "vof_range": [vof_min, vof_max]},
            "count": len(kept_structures),
            "details": kept_structures,
            "directory": workshop_dir
        })

    # Inject database-retrieved solutions into the SAES population.
    if kept_structures:
        try:
            from tools.saes_integration import is_saes_enabled, get_saes_integrator
            if is_saes_enabled():
                integrator = get_saes_integrator()
                integrator.inject_db_solution(kept_structures)
                print(f"  [SAES] Injected {len(kept_structures)} database solutions as unverified population candidates")
        except ImportError:
            pass
        except Exception as e:
            print(f"  [SAES] Error injecting database solutions: {e}")

    print(f"\n{'='*60}")
    print(f"[Database Retrieval] Complete: retrieved {len(copied_files)} -> analyzed {len(analyzed_structures)} -> kept {len(kept_structures)}")
    print(f"{'='*60}\n")

    return {
        "status": f"success: database retrieval complete; kept {len(kept_structures)} microstructures",
        "message": f"Database retrieval and filtering complete. {len(kept_structures)} microstructures passed property filtering.",
        "source": "database",
        "total_retrieved": len(copied_files),
        "total_analyzed": len(analyzed_structures),
        "total_kept": len(kept_structures),
        "kept_structures_details": [s for s in kept_structures],
        "failed_count": failed_count,
        "microstructure_directory": workshop_dir
    }


retrieve_microstructure_from_database.tool_info = {
    "tool_name": "retrieve_microstructure_from_database",
    "tool_title": "Database Microstructure Retrieval",
    "tool_description": "Retrieve microstructures matching target mechanical properties, then calculate and filter homogenized properties. Complements AI generation with existing database candidates.",
    "tool_params": [
        {"name": "E", "description": "Target Young's modulus (MPa).", "type": "number", "required": True},
        {"name": "nu", "description": "Target Poisson's ratio (0.0-0.5).", "type": "number", "required": True},
        {"name": "G", "description": "Target shear modulus (MPa).", "type": "number", "required": True},
        {"name": "custom_base_material", "description": "Required base-material dictionary containing E, G, nu, and name. Example: {'E': 68970, 'G': 25900, 'nu': 0.33, 'name': 'Al6061', 'k': 167}.", "type": "object", "required": True},
        {"name": "vof_min", "description": "Lower volume-fraction bound; default 0.10.", "type": "number", "required": False, "default": 0.10},
        {"name": "vof_max", "description": "Upper volume-fraction bound; default 0.50.", "type": "number", "required": False, "default": 0.50},
        {"name": "top_k", "description": "Number of candidates to retrieve from the database; default 15.", "type": "number", "required": False, "default": 15},
        {"name": "custom_name", "description": "Custom retrieval-batch name.", "type": "string", "required": False, "default": "db_retrieved_structures"}
    ]
}


# ============================================================================
# Unified microstructure acquisition: AI generation plus database retrieval.
# ============================================================================

def acquire_microstructures(
    E: float,
    nu: float,
    G: float,
    custom_base_material,  # Required parameter: base-material information is mandatory.
    vof_min: float = 0.10,
    vof_max: float = 0.50,
    top_k: int = 15,
    custom_name: str = "microstructures"
) -> Dict[str, Any]:
    """
    Unified microstructure acquisition: run AI generation and database retrieval, then merge their results.

    The tool invokes two sources in parallel:
    1. AI diffusion-model generation of new microstructures.
    2. Database retrieval of existing high-quality microstructures.

    Both sources run internal homogenized-property filtering. The returned
    result merges every retained microstructure candidate.

    Args:
        E: Target Young's modulus (MPa).
        nu: Target Poisson's ratio.
        G: Target shear modulus (MPa).
        custom_base_material: Required base-material dictionary containing E, G,
            nu, and name. Example: {"E": 68970, "G": 25900, "nu": 0.33,
            "name": "Al6061", "k": 167}.
        vof_min: Lower database-retrieval volume-fraction bound (default: 0.10).
        vof_max: Upper database-retrieval volume-fraction bound (default: 0.50).
        top_k: Number of database retrieval candidates (default: 15).
        custom_name: Custom batch name.

    Returns:
        Dictionary containing merged AI-generation and database-retrieval results.
    """
    print(f"\n{'='*70}")
    print("[Unified Microstructure Acquisition] Starting AI generation plus database retrieval")
    print(f"  Target parameters: E={E} MPa, G={G} MPa, nu={nu}")
    print(f"  Volume-fraction range (database retrieval): [{vof_min}, {vof_max}]")
    print(f"{'='*70}\n")

    # In wo_dual_source ablation mode, execute only one source.
    ablation_dual_source = os.environ.get("CHATMS_ABLATION_DUAL_SOURCE", "0") == "1"
    run_ai = True
    run_db = True
    if ablation_dual_source:
        # Choose the source from the environment in ablation mode.
        selected_branch = os.environ.get("CHATMS_ABLATION_BRANCH", "ai_generation")
        if selected_branch == "database":
            run_ai = False
            print("  [Ablation mode] Executing database retrieval only; skipping AI generation")
        else:
            run_db = False
            print("  [Ablation mode] Executing AI generation only; skipping database retrieval")

    ai_result = None
    db_result = None
    ai_error = None
    db_error = None

    # ========== 1. AI microstructure generation ==========
    if run_ai:
        print("[Source 1/2] Starting AI diffusion-model microstructure generation...")
        print("-" * 50)
        try:
            ai_result = generate_microstructure_with_ai(
                E=E, nu=nu, G=G,
                custom_base_material=custom_base_material,
                custom_name=f"{custom_name}_ai"
            )
            if ai_result and ai_result.get("error"):
                ai_error = ai_result["error"]
                print(f"[Source 1/2] AI generation returned an error: {ai_error}")
            else:
                ai_kept = len(ai_result.get("kept_structures_details", [])) if ai_result else 0
                print(f"[Source 1/2] AI generation complete; kept {ai_kept} microstructures")
        except Exception as e:
            ai_error = str(e)
            print(f"[Source 1/2] AI generation exception: {ai_error}")
    else:
        print("[Source 1/2] AI generation skipped in ablation mode")

    print()

    # ========== 2. Database microstructure retrieval ==========
    if run_db:
        print("[Source 2/2] Starting database microstructure retrieval...")
        print("-" * 50)
        try:
            db_result = retrieve_microstructure_from_database(
                E=E, nu=nu, G=G,
                custom_base_material=custom_base_material,
                vof_min=vof_min,
                vof_max=vof_max,
                top_k=top_k,
                custom_name=f"{custom_name}_db"
            )
            if db_result and db_result.get("error"):
                db_error = db_result["error"]
                print(f"[Source 2/2] Database retrieval returned an error: {db_error}")
            else:
                db_kept = len(db_result.get("kept_structures_details", [])) if db_result else 0
                print(f"[Source 2/2] Database retrieval complete; kept {db_kept} microstructures")
        except Exception as e:
            db_error = str(e)
            print(f"[Source 2/2] Database retrieval exception: {db_error}")
    else:
        print("[Source 2/2] Database retrieval skipped in ablation mode")

    # ========== 3. Merge results ==========
    print(f"\n{'='*70}")
    print("[Unified Microstructure Acquisition] Merging dual-source results")
    print(f"{'='*70}")

    all_kept = []
    ai_kept_count = 0
    db_kept_count = 0

    # Collect AI-generated microstructures.
    if ai_result and not ai_result.get("error"):
        ai_structures = ai_result.get("kept_structures_details", [])
        for s in ai_structures:
            s["source"] = "ai_generation"
        all_kept.extend(ai_structures)
        ai_kept_count = len(ai_structures)

    # Collect database-retrieved microstructures.
    if db_result and not db_result.get("error"):
        db_structures = db_result.get("kept_structures_details", [])
        for s in db_structures:
            s["source"] = "database_retrieval"
        all_kept.extend(db_structures)
        db_kept_count = len(db_structures)

    total_kept = len(all_kept)

    print(f"  AI generation kept: {ai_kept_count}")
    print(f"  Database retrieval kept: {db_kept_count}")
    print(f"  Total retained: {total_kept} microstructures")

    # Sort by total_deviation when available.
    all_kept.sort(key=lambda x: x.get("total_deviation", float('inf')))

    # Print merged ranking.
    if all_kept:
        print("\n  Merged ranking (smallest deviation first):")
        for i, s in enumerate(all_kept, 1):
            src_tag = "AI" if s.get("source") == "ai_generation" else "DB"
            dev = s.get("total_deviation", -1)
            props = s.get("scaled_properties", {})
            print(f"    #{i} {src_tag} {s.get('filename', 'unknown')} "
                  f"E={props.get('E', 0):.1f}, G={props.get('G', 0):.1f}, "
                  f"nu={props.get('nu', 0):.3f}, deviation={dev:.3f}")

    # Determine the workshop directory.
    workshop_dir = None
    if ai_result and not ai_result.get("error"):
        workshop_dir = ai_result.get("microstructure_directory")
    if not workshop_dir and db_result and not db_result.get("error"):
        workshop_dir = db_result.get("microstructure_directory")
    if not workshop_dir:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        workshop_dir = os.path.join(project_root, "data", "workshop")

    # Determine overall status.
    if total_kept > 0:
        status = "success"
        message = (f"Dual-source microstructure acquisition complete. AI generation kept {ai_kept_count} and database retrieval kept {db_kept_count}; "
                   f"{total_kept} microstructures passed property filtering.")
    elif ai_error and db_error:
        status = "failed"
        message = f"Both sources failed. AI error: {ai_error}; database error: {db_error}"
    elif ai_error:
        status = "partial_failure"
        message = f"AI generation failed ({ai_error}); database retrieval returned {db_kept_count} microstructures."
    elif db_error:
        status = "partial_failure"
        message = f"Database retrieval failed ({db_error}); AI generation returned {ai_kept_count} microstructures."
    else:
        status = "no_results"
        message = "Neither source returned a qualifying microstructure; consider adjusting the target parameters."

    print(f"\n  Status: {status}")
    print(f"  {message}")
    print(f"{'='*70}\n")

    # Record the result in the tracker.
    tracker = get_global_tracker()
    if tracker:
        tracker.add_node("dual_source_acquisition", "Dual-Source Microstructure Acquisition", {
            "parameters": {"E": E, "nu": nu, "G": G},
            "ai_count": ai_kept_count,
            "db_count": db_kept_count,
            "total_count": total_kept,
            "directory": workshop_dir
        })

    return {
        "status": status,
        "message": message,
        "total_kept": total_kept,
        "ai_generation": {
            "kept_count": ai_kept_count,
            "error": ai_error,
            "details": ai_result.get("kept_structures_details", []) if ai_result and not ai_result.get("error") else []
        },
        "database_retrieval": {
            "kept_count": db_kept_count,
            "error": db_error,
            "details": db_result.get("kept_structures_details", []) if db_result and not db_result.get("error") else []
        },
        "kept_structures_details": all_kept,
        "microstructure_directory": workshop_dir
    }


acquire_microstructures.tool_info = {
    "tool_name": "acquire_microstructures",
    "tool_title": "Unified Microstructure Acquisition (AI Generation + Database Retrieval)",
    "tool_description": "Run AI diffusion-model generation and database retrieval, then return all microstructure candidates satisfying the target mechanical properties. One call returns both sources.",
    "tool_params": [
        {"name": "E", "description": "Target Young's modulus (MPa); must not exceed the base material's Young's modulus.", "type": "number", "required": True},
        {"name": "nu", "description": "Target Poisson's ratio (0.0-0.5).", "type": "number", "required": True},
        {"name": "G", "description": "Target shear modulus (MPa); must not exceed the base material's shear modulus.", "type": "number", "required": True},
        {"name": "custom_base_material", "description": "Required base-material dictionary containing E, G, nu, and name. Example: {'E': 68970, 'G': 25900, 'nu': 0.33, 'name': 'Al6061', 'k': 167}.", "type": "object", "required": True},
        {"name": "vof_min", "description": "Lower database-retrieval volume-fraction bound; default 0.10.", "type": "number", "required": False, "default": 0.10},
        {"name": "vof_max", "description": "Upper database-retrieval volume-fraction bound; default 0.50.", "type": "number", "required": False, "default": 0.50},
        {"name": "top_k", "description": "Number of database-retrieval candidates; default 15.", "type": "number", "required": False, "default": 15},
        {"name": "custom_name", "description": "Custom batch name.", "type": "string", "required": False, "default": "microstructures"}
    ]
}
