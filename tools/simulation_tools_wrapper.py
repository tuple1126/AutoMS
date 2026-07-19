from typing import Dict, Any, Optional, Set
import csv
import json
import os
import glob
import subprocess
import sys
import time
from pathlib import Path
from tools.tree_tracker import get_global_tracker, get_tool_call_tracker, get_property_table
from tools.project_paths import PETL_RESULTS_DIR

# Name of the agent currently invoking a tool (set by LightAgent).
_current_agent_name = "Unknown"

# Tracks simulated files by simulation type to prevent duplicate simulations.
# Structure: {filename: {"heat": bool, "electrical": bool, "stiffness": bool, "plasticity": bool}}
_simulated_files: Dict[str, Dict[str, bool]] = {}

def reset_simulated_files_tracker():
    """Reset the simulated-file tracker."""
    global _simulated_files
    _simulated_files = {}

def get_simulated_files_tracker() -> Dict[str, Dict[str, bool]]:
    """Return the simulated-file tracker."""
    return _simulated_files

def record_simulated_file(file_path: str, simulation_type: str = "unknown"):
    """
    Record a simulated file and its simulation type.
    
    Args:
        file_path: File path.
        simulation_type: Simulation type (heat, electrical, stiffness, or plasticity).
    """
    global _simulated_files
    file_basename = os.path.basename(file_path)
    
    # Initialize the file record.
    if file_basename not in _simulated_files:
        _simulated_files[file_basename] = {
            "heat": False,
            "electrical": False,
            "stiffness": False,
            "plasticity": False
        }
    
    # Mark the simulation type as complete.
    if simulation_type in _simulated_files[file_basename]:
        _simulated_files[file_basename][simulation_type] = True

def record_simulated_files_from_results(results: Dict[str, Any], simulation_type: str = "unknown"):
    """
    Record simulated files from simulation results.
    
    Args:
        results: Simulation results.
        simulation_type: Simulation type (heat, electrical, stiffness, or plasticity).
    """
    results_list = results.get("results", [])
    for result in results_list:
        # Read a filename, which may be a full path or a name with or without an extension.
        file_path = result.get("file") or result.get("input_file") or result.get("obj_file") or result.get("filename", "")
        if file_path:
            # Extract the basename and ensure it has an .obj extension.
            file_basename = os.path.basename(file_path)
            if not file_basename.endswith(".obj"):
                file_basename = file_basename + ".obj"
            record_simulated_file(file_basename, simulation_type)
            print(f"   Recorded simulation: {file_basename} (type: {simulation_type})")

def filter_unsimulated_files(input_path: str, simulation_type: str) -> list:
    """
    Filter files that have not been simulated for a requested simulation type.
    
    Args:
        input_path: Input file or directory.
        simulation_type: Simulation type (heat, electrical, stiffness, or plasticity).
    
    Returns:
        Files not yet simulated for the requested type.
    """
    global _simulated_files
    
    # Collect all .obj files.
    if os.path.isfile(input_path) and input_path.endswith('.obj'):
        obj_files = [input_path]
    elif os.path.isdir(input_path):
        obj_files = glob.glob(os.path.join(input_path, "*.obj"))
    else:
        return []
    
    # Filter out files that have already been simulated.
    unsimulated_files = []
    for obj_file in obj_files:
        file_basename = os.path.basename(obj_file)
        
        # Include files that have not been tracked or have not completed this type.
        if file_basename not in _simulated_files:
            unsimulated_files.append(obj_file)
        elif not _simulated_files[file_basename].get(simulation_type, False):
            unsimulated_files.append(obj_file)
    
    if len(obj_files) > len(unsimulated_files):
        skipped = len(obj_files) - len(unsimulated_files)
        print(f"   [Duplicate prevention] Found {len(obj_files)} files; skipping {skipped} files with completed {simulation_type} simulations")
        print(f"   Running {simulation_type} simulation for {len(unsimulated_files)} files")
    
    return unsimulated_files

def hide_simulated_files_for_iteration(workshop_path: str = None) -> int:
    """
    At the start of an iteration, rename previously simulated files to .objk
    and clear the PETL results cache directory.
    
    Args:
        workshop_path: Workshop directory path. Defaults to ``data/workshop``.
    
    Returns:
        Number of hidden files.
    """
    global _simulated_files
    
    if workshop_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workshop_path = os.path.join(os.path.dirname(current_dir), 'data', 'workshop')
    
    if not os.path.isdir(workshop_path):
        print(f"   Warning: workshop directory does not exist: {workshop_path}")
        return 0
    
    # Clear the PETL results cache directory.
    petl_results_dir = str(PETL_RESULTS_DIR)
    if os.path.isdir(petl_results_dir):
        try:
            import shutil
            # Remove every file and subdirectory in the directory.
            for item in os.listdir(petl_results_dir):
                item_path = os.path.join(petl_results_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"   Cleared PETL results cache directory: {petl_results_dir}")
        except Exception as e:
            print(f"   Warning: failed to clear PETL cache directory: {e}")
    
    # Diagnostic information: show the current tracker state.
    print(f"   Simulated-file tracker: {len(_simulated_files)} files")
    
    # The current mechanism uses ``filter_unsimulated_files()`` rather than hiding files.
    print("   [Current mechanism] Uses filtering instead of physical file hiding; supports multiple simulation types")
    
    return 0
    
    hidden_count = 0
    
    # Collect all .obj files in the workshop.
    obj_files = glob.glob(os.path.join(workshop_path, "*.obj"))
    print(f"   Found {len(obj_files)} .obj files in the workshop")
    
    for obj_file in obj_files:
        file_basename = os.path.basename(obj_file)
        if file_basename in _simulated_files:
            # Rename to .objk.
            new_name = obj_file + "k"
            try:
                os.rename(obj_file, new_name)
                hidden_count += 1
                print(f"   Hid simulated file: {file_basename}")
            except Exception as e:
                print(f"   Warning: failed to hide {file_basename}: {e}")
        else:
            print(f"   Skipping unsimulated file: {file_basename}")
    
    if hidden_count > 0:
        print(f"\n[Duplicate prevention] Hid {hidden_count} microstructure files simulated in the previous iteration")
    else:
        print("   No simulated files need to be hidden")
    
    return hidden_count

def restore_hidden_files(input_path: str) -> int:
    """
    Restore all .objk files to .obj.
    
    Args:
        input_path: Input directory.
    
    Returns:
        Number of restored files.
    """
    if not os.path.isdir(input_path):
        return 0
    
    restored_count = 0
    for objk_file in glob.glob(os.path.join(input_path, "*.objk")):
        # Restore the .obj suffix.
        original_name = objk_file[:-1]  # Remove the trailing 'k'.
        try:
            os.rename(objk_file, original_name)
            restored_count += 1
        except Exception as e:
            print(f"   Warning: failed to restore {os.path.basename(objk_file)}: {e}")
    
    if restored_count > 0:
        print(f"   Restored {restored_count} hidden files")
    
    return restored_count

def restore_all_hidden_files_globally(workshop_path: str = None) -> int:
    """
    Restore all hidden .objk files after a conversation ends.
    
    Args:
        workshop_path: Workshop directory path. Defaults to ``data/workshop``.
    
    Returns:
        Total number of restored files.
    """
    if workshop_path is None:
        # Default path.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workshop_path = os.path.join(os.path.dirname(current_dir), 'data', 'workshop')
    
    if not os.path.isdir(workshop_path):
        return 0
    
    restored_count = restore_hidden_files(workshop_path)
    
    if restored_count > 0:
        print(f"\n[Session complete] Restored all {restored_count} hidden microstructure files")
    
    return restored_count

def set_current_agent(agent_name: str):
    """Set the name of the agent invoking a tool."""
    global _current_agent_name
    _current_agent_name = agent_name

def get_current_agent() -> str:
    """Return the name of the agent invoking a tool."""
    return _current_agent_name

def get_obj_files(input_path: str = "data/workshop") -> list:
    """Get OBJ files to analyze in the workshop directory."""
    if os.path.isfile(input_path) and input_path.endswith(".obj"):
        return [input_path]
    if not os.path.isdir(input_path):
        return []
    return sorted(glob.glob(os.path.join(input_path, "*.obj")))

get_obj_files.tool_info = {
    "tool_name": "get_obj_files",
    "tool_title": "Get OBJ Files",
    "tool_description": "List OBJ microstructure files awaiting simulation; scans data/workshop by default.",
    "tool_params": [
        {"name": "input_path", "description": "Input directory or OBJ file path", "type": "string", "required": False, "default": "data/workshop"}
    ]
}

def _inject_to_saes(simulation_results: Dict[str, Any], analysis_type: str):
    """
    Inject simulation results into the SAES population.
    
    This is the integration point between SAES and simulation calculations.
    """
    try:
        from tools.saes_integration import is_saes_enabled, get_saes_integrator
        
        if is_saes_enabled():
            integrator = get_saes_integrator()
            results_list = simulation_results.get("results", [])
            integrator.inject_simulation_results({
                "analysis_type": analysis_type,
                "results": results_list
            })
            # Detailed log output.
            print("\n[SAES] Simulation feedback injection complete")
            print(f"   Simulation type: {analysis_type}")
            print(f"   Result count: {len(results_list)}")
            pareto_size = len(integrator.saes.pareto_front)
            print(f"   Current Pareto front: {pareto_size} nondominated solutions")
        else:
            print("\n[SAES] Disabled; simulation results were not injected into the population")
    except ImportError:
        pass  # The SAES module is not loaded.
    except Exception as e:
        print(f"[SAES] Error while injecting simulation results: {e}")

# Wrapper for heat analysis
def run_heat_analysis_wrapper(
    input_path: str,
    output_dir: str = None,
    resolution: int = 64,
    mode: str = 'solid',
    base_heat: float = 1.0,
    use_gpu: bool = True,
    enable_visualization: bool = False,
    gpu_device: str = 'cuda:0',
    solver_tol: float = 1e-6,
    solver_max_iter: int = 5000,
    analysis_type: str = 'auto',
    enable_comparison: bool = True,
    save_detailed_results: bool = True
) -> Dict[str, Any]:
    """Run integrated heat analysis."""
    from tools.heat_analysis_tool import run_integrated_heat_analysis
    
    # Select only files not yet simulated for thermal analysis.
    filtered_files = filter_unsimulated_files(input_path, "heat")
    if not filtered_files:
        print("   All files have completed thermal simulation; skipping")
        return {
            "success": True,
            "message": "All files already simulated for heat analysis",
            "skipped": True,
            "results": []
        }
    
    # For a directory input, create a temporary directory containing only unsimulated files.
    temp_dir_created = False
    original_input_path = input_path
    if os.path.isdir(input_path) and len(filtered_files) < len(glob.glob(os.path.join(input_path, "*.obj"))):
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="heat_sim_")
        temp_dir_created = True
        for f in filtered_files:
            basename = os.path.basename(f)
            link_path = os.path.join(temp_dir, basename)
            try:
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(f, link_path)
                else:  # Unix
                    os.symlink(f, link_path)
            except:
                import shutil
                shutil.copy2(f, link_path)
        input_path = temp_dir
        print(f"   Created temporary directory: {temp_dir} ({len(filtered_files)} files awaiting simulation)")
    
    params = {
        "input_path": original_input_path,
        "output_dir": output_dir,
        "resolution": resolution,
        "mode": mode,
        "base_heat": base_heat,
        "use_gpu": use_gpu
    }
    
    try:
        result = run_integrated_heat_analysis(
            input_path=input_path,
            output_dir=output_dir,
            resolution=resolution,
            mode=mode,
            base_heat=base_heat,
            use_gpu=use_gpu,
            enable_visualization=enable_visualization,
            gpu_device=gpu_device,
            solver_tol=solver_tol,
            solver_max_iter=solver_max_iter,
            analysis_type=analysis_type,
            enable_comparison=enable_comparison,
            save_detailed_results=save_detailed_results
        )
        
        success = result.get("success", False)
        
        # Record files from this run as having completed thermal analysis.
        if success:
            record_simulated_files_from_results(result, "heat")
        
        # Remove the temporary directory.
        if temp_dir_created:
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"   Removed temporary directory: {temp_dir}")
            except:
                pass
        
        # Record tool call to tracker
        tool_tracker = get_tool_call_tracker()
        tool_tracker.record_tool_call(
            agent_name=get_current_agent(),
            tool_name="run_integrated_heat_analysis",
            params=params,
            result=result,
            success=success
        )
        
        # Update microstructure property table
        prop_table = get_property_table()
        prop_table.update_from_heat_analysis(result)
        
        # SAES: inject simulation results into the population.
        _inject_to_saes(result, "heat_analysis")
        
        # Record to tree tracker
        tracker = get_global_tracker()
        if tracker and success:
            tracker.record_simulation_result("Heat Analysis", result)
            
        return result
        
    except Exception as e:
        # Record failed tool call
        tool_tracker = get_tool_call_tracker()
        tool_tracker.record_tool_call(
            agent_name=get_current_agent(),
            tool_name="run_integrated_heat_analysis",
            params=params,
            result={"error": str(e)},
            success=False
        )
        raise

run_heat_analysis_wrapper.tool_info = {
    "tool_name": "run_integrated_heat_analysis",
    "tool_title": "Thermal Conductivity Analysis",
    "tool_description": "Simulate the thermal conductivity of a microstructure.",
    "tool_params": [
        {"name": "input_path", "description": "Input file path", "type": "string", "required": True},
        {"name": "output_dir", "description": "Output directory", "type": "string", "required": False},
        {"name": "base_heat", "description": "Base-material thermal conductivity", "type": "number", "required": False, "default": 1.0},
        {"name": "use_gpu", "description": "Whether to use a GPU", "type": "boolean", "required": False, "default": True}
    ]
}

# Wrapper for stiffness analysis
def run_stiffness_analysis_wrapper(
    input_path: str,
    output_dir: str = None,
    youngs_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
    voxel_resolution: int = 64
) -> Dict[str, Any]:
    """Run stiffness analysis."""
    from tools.stiffness_analysis_tool import run_stiffness_analysis
    
    # Select only files not yet simulated for stiffness analysis.
    filtered_files = filter_unsimulated_files(input_path, "stiffness")
    if not filtered_files:
        print("   All files have completed stiffness simulation; skipping")
        return {
            "success": True,
            "message": "All files already simulated for stiffness analysis",
            "skipped": True,
            "results": []
        }
    
    # For a directory input, create a temporary directory containing only unsimulated files.
    temp_dir_created = False
    original_input_path = input_path
    if os.path.isdir(input_path) and len(filtered_files) < len(glob.glob(os.path.join(input_path, "*.obj"))):
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="stiff_sim_")
        temp_dir_created = True
        for f in filtered_files:
            basename = os.path.basename(f)
            link_path = os.path.join(temp_dir, basename)
            try:
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(f, link_path)
                else:  # Unix
                    os.symlink(f, link_path)
            except:
                import shutil
                shutil.copy2(f, link_path)
        input_path = temp_dir
        print(f"   Created temporary directory: {temp_dir} ({len(filtered_files)} files awaiting simulation)")
    
    params = {
        "input_path": original_input_path,
        "output_dir": output_dir,
        "youngs_modulus": youngs_modulus,
        "poisson_ratio": poisson_ratio,
        "voxel_resolution": voxel_resolution
    }
    
    try:
        result = run_stiffness_analysis(
            obj_file=input_path,
            output_dir=output_dir,
            youngs_modulus=youngs_modulus,
            poisson_ratio=poisson_ratio,
            resolution=voxel_resolution
        )
        
        success = result.get("success", False)
        
        # Record files from this run as having completed stiffness analysis.
        if success:
            record_simulated_files_from_results(result, "stiffness")
        
        # Remove the temporary directory.
        if temp_dir_created:
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"   Removed temporary directory: {temp_dir}")
            except:
                pass
        
        # Record tool call to tracker
        tool_tracker = get_tool_call_tracker()
        tool_tracker.record_tool_call(
            agent_name=get_current_agent(),
            tool_name="run_stiffness_analysis",
            params=params,
            result=result,
            success=success
        )
        
        # Update microstructure property table
        prop_table = get_property_table()
        prop_table.update_from_stiffness_analysis(result)
        
        # SAES: inject simulation results into the population.
        _inject_to_saes(result, "stiffness_analysis")
        
        # Record to tree tracker
        tracker = get_global_tracker()
        if tracker and success:
            tracker.record_simulation_result("Stiffness Analysis", result)
            
        return result
        
    except Exception as e:
        # Record failed tool call
        tool_tracker = get_tool_call_tracker()
        tool_tracker.record_tool_call(
            agent_name=get_current_agent(),
            tool_name="run_stiffness_analysis",
            params=params,
            result={"error": str(e)},
            success=False
        )
        raise

run_stiffness_analysis_wrapper.tool_info = {
    "tool_name": "run_stiffness_analysis",
    "tool_title": "Stiffness Analysis",
    "tool_description": "Simulate the stiffness properties of a microstructure.",
    "tool_params": [
        {"name": "input_path", "description": "Input file path", "type": "string", "required": True},
        {"name": "youngs_modulus", "description": "Base-material Young's modulus", "type": "number", "required": False, "default": 1.0},
        {"name": "poisson_ratio", "description": "Base-material Poisson's ratio", "type": "number", "required": False, "default": 0.3}
    ]
}

def batch_stiffness_analysis_wrapper(
    input_path: str,
    output_dir: str = None,
    youngs_modulus: float = 1.0,
    poisson_ratio: float = 0.3,
    voxel_resolution: int = 64
) -> Dict[str, Any]:
    """Compatibility wrapper matching the Simulator prompt's batch tool name."""
    return run_stiffness_analysis_wrapper(
        input_path=input_path,
        output_dir=output_dir,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        voxel_resolution=voxel_resolution
    )

batch_stiffness_analysis_wrapper.tool_info = {
    "tool_name": "batch_stiffness_analysis",
    "tool_title": "Batch Stiffness Analysis",
    "tool_description": "Run batch stiffness and homogenization analysis for microstructures in a directory.",
    "tool_params": [
        {"name": "input_path", "description": "Input directory or OBJ file path", "type": "string", "required": True},
        {"name": "output_dir", "description": "Output directory", "type": "string", "required": False},
        {"name": "youngs_modulus", "description": "Base-material Young's modulus", "type": "number", "required": False, "default": 1.0},
        {"name": "poisson_ratio", "description": "Base-material Poisson's ratio", "type": "number", "required": False, "default": 0.3},
        {"name": "voxel_resolution", "description": "Voxel resolution", "type": "integer", "required": False, "default": 64}
    ]
}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _resolve_existing_path(path_value: str, base_dir: str = None) -> str:
    if not path_value:
        return ""
    path_text = str(path_value)
    if os.path.exists(path_text):
        return path_text
    if base_dir:
        candidate = os.path.join(base_dir, path_text)
        if os.path.exists(candidate):
            return candidate
    return path_text


def _collect_plasticity_mesh_files(input_path: str, target_files: list = None) -> list:
    """Collect ``.msh`` files consumed by the plasticity solver."""
    base_dir = input_path if input_path and os.path.isdir(input_path) else None
    mesh_files = []

    for item in _as_list(target_files):
        resolved = _resolve_existing_path(str(item), base_dir)
        if os.path.isdir(resolved):
            mesh_files.extend(sorted(glob.glob(os.path.join(resolved, "*.msh"))))
        elif resolved.endswith(".msh"):
            mesh_files.append(resolved)

    if not mesh_files and input_path:
        if os.path.isdir(input_path):
            mesh_files.extend(sorted(glob.glob(os.path.join(input_path, "*.msh"))))
        elif input_path.endswith(".msh"):
            mesh_files.append(input_path)

    return sorted(dict.fromkeys(mesh_files))


def _infer_experiment_files(mesh_file: str, experiment_files: list = None) -> list:
    explicit = [
        _resolve_existing_path(str(item), os.path.dirname(mesh_file))
        for item in _as_list(experiment_files)
    ]
    explicit = [item for item in explicit if item and os.path.exists(item)]
    if explicit:
        return explicit

    candidates = [
        f"{mesh_file}.txt",
        f"{os.path.splitext(mesh_file)[0]}.txt",
    ]
    return [path for path in candidates if os.path.exists(path)]


def _obj_name_from_mesh(mesh_file: str) -> str:
    name = os.path.basename(mesh_file)
    if name.endswith(".obj_.msh"):
        return name[:-len("_.msh")]
    if name.endswith(".msh"):
        return name[:-len(".msh")] + ".obj"
    return name


def _sanitize_case_name(text: str) -> str:
    safe = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in "._-") else "_"
        for ch in os.path.basename(str(text))
    )
    return safe.strip("_") or "case"


def _read_json_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_plasticity_curve(run_root: str):
    corrected_path = os.path.join(run_root, "simulation_corrected.csv")
    raw_path = os.path.join(run_root, "simulation.csv")
    strains, stresses = [], []

    if os.path.exists(corrected_path):
        with open(corrected_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    strains.append(float(row.get("corrected_strain", "")))
                    stresses.append(float(row.get("corrected_stress_MPa", "")))
                except (TypeError, ValueError):
                    continue
    elif os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    strains.append(float(row.get("strain", "")))
                    stresses.append(float(row.get("apparent_stress_MPa", "")))
                except (TypeError, ValueError):
                    continue

    return strains, stresses


def _interp_at(xs, ys, target_x: float):
    if not xs or not ys:
        return None
    points = sorted(zip(xs, ys), key=lambda pair: pair[0])
    if target_x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= target_x <= x1 and x1 != x0:
            t = (target_x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


def _integrate_curve(xs, ys):
    if len(xs) < 2:
        return None
    points = sorted(zip(xs, ys), key=lambda pair: pair[0])
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += 0.5 * (y0 + y1) * max(x1 - x0, 0.0)
    return area


def _estimate_plasticity_metrics(run_root: str) -> Dict[str, Any]:
    strains, stresses = _read_plasticity_curve(run_root)
    if not strains or not stresses:
        return {}

    max_strain = max(strains)
    max_stress = max(stresses)
    yield_strength = _interp_at(strains, stresses, 0.002)
    initial_pairs = [
        (x, y)
        for x, y in zip(strains, stresses)
        if 0.0 < x <= max(0.005, max_strain * 0.15)
    ]
    if initial_pairs:
        denom = sum(x * x for x, _ in initial_pairs)
        slope = sum(x * y for x, y in initial_pairs) / denom if denom > 0 else 0.0
    else:
        slope = 0.0
    plastic_strain = max(max_strain - (max_stress / slope), 0.0) if slope > 0 else max_strain
    energy_absorption = _integrate_curve(strains, stresses)

    return {
        "yield_strength": yield_strength,
        "yield_strength_method": "0.2% strain proof stress estimated from corrected simulation curve",
        "ultimate_strength": max_stress,
        "max_stress": max_stress,
        "max_strain": max_strain,
        "plastic_strain": plastic_strain,
        "specific_energy": energy_absorption,
        "energy_absorption": energy_absorption,
        "energy_unit": "MJ/m^3",
    }


# Wrapper for mesh-based plasticity simulation.
def run_plasticity_simulation_wrapper(
    input_path: str,
    target_files: list = None,
    experiment_files: list = None,
    output_dir: str = None,
    strain_limit: float = 0.05,
    steps: int = 100,
    gpu_device_id: int = 0,
    custom_E: float = None,
    custom_nu: float = None,
    custom_sig0: float = None,
    custom_H1: float = None,
    custom_Q_inf: float = 18.0,
    custom_b: float = 8.0,
    custom_eta: float = 1.0,
    specimen_height_mm: float = 40.0,
    specimen_width_mm: float = 40.0,
    specimen_depth_mm: float = 40.0,
    boundary_mode: str = "rough_platen",
    real_dimension_mm: float = None,
    no_mesh_scaling: bool = True,
    enable_multi_gpu: bool = False,
) -> Dict[str, Any]:
    """Run the mesh-based J2 plasticity solver in ``plasticity_simulation.py``."""
    if not input_path:
        raise ValueError("The input_path argument is required and must be a .msh file or a directory containing .msh meshes")

    if not os.path.exists(input_path):
        raise ValueError(f"input_path does not exist: {input_path}")

    if custom_E is None or custom_nu is None or custom_sig0 is None or custom_H1 is None:
        raise ValueError("custom_E, custom_nu, custom_sig0, and custom_H1 are required material parameters for plasticity simulation")

    script_path = Path(__file__).with_name("plasticity_simulation.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Plasticity simulation script does not exist: {script_path}")

    output_root = os.path.abspath(output_dir or os.path.join("data", "plasticity_results"))
    os.makedirs(output_root, exist_ok=True)

    mesh_files = _collect_plasticity_mesh_files(input_path, target_files)
    if not mesh_files:
        raise FileNotFoundError("No .msh mesh files were found.")

    material_params = {
        "E": float(custom_E),
        "nu": float(custom_nu),
        "sig0": float(custom_sig0),
        "H1": float(custom_H1),
        "Q_inf": float(custom_Q_inf),
        "b": float(custom_b),
        "eta": float(custom_eta),
    }
    params = {
        "input_path": input_path,
        "target_files": target_files,
        "experiment_files": experiment_files,
        "output_dir": output_dir,
        "strain_limit": strain_limit,
        "steps": steps,
        "gpu_device_id": gpu_device_id,
        "material_params": material_params,
        "specimen_height_mm": specimen_height_mm,
        "specimen_width_mm": specimen_width_mm,
        "specimen_depth_mm": specimen_depth_mm,
        "boundary_mode": boundary_mode,
        "real_dimension_mm": real_dimension_mm,
        "no_mesh_scaling": no_mesh_scaling,
    }

    results = []
    errors = []
    for mesh_file in mesh_files:
        mesh_file = os.path.abspath(mesh_file)
        mesh_experiments = _infer_experiment_files(mesh_file, experiment_files)
        obj_name = _obj_name_from_mesh(mesh_file)
        case_id = _sanitize_case_name(os.path.splitext(os.path.basename(mesh_file))[0])
        run_tag = _sanitize_case_name(f"{case_id}_{time.strftime('%Y%m%d_%H%M%S')}_{len(results)}")
        run_root = os.path.join(output_root, case_id, "runs", run_tag)

        if not mesh_experiments:
            error = {
                "filename": obj_name,
                "file": obj_name,
                "mesh_file": mesh_file,
                "success": False,
                "error": "No experimental curve file was found. Provide experiment_files or place a matching .msh.txt file next to the mesh.",
            }
            results.append(error)
            errors.append(error)
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--mesh-file",
            mesh_file,
            "--experiment-files",
            *mesh_experiments,
            "--case-id",
            case_id,
            "--output-root",
            output_root,
            "--run-tag",
            run_tag,
            "--specimen-height-mm",
            str(specimen_height_mm),
            "--specimen-width-mm",
            str(specimen_width_mm),
            "--specimen-depth-mm",
            str(specimen_depth_mm),
            "--boundary-mode",
            boundary_mode,
            "--target-num-steps",
            str(int(steps)),
            "--max-engineering-strain",
            str(float(strain_limit)),
            "--params-json",
            json.dumps(material_params, ensure_ascii=False),
        ]
        if real_dimension_mm is not None:
            cmd.extend(["--real-dimension-mm", str(real_dimension_mm)])
        if no_mesh_scaling:
            cmd.append("--no-mesh-scaling")

        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_device_id))
        if not enable_multi_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_device_id)

        completed = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            text=True,
            capture_output=True,
        )

        manifest = _read_json_file(os.path.join(run_root, "run_manifest.json"))
        score = _read_json_file(os.path.join(run_root, "score.json")) or manifest.get("score", {})
        metrics = _estimate_plasticity_metrics(run_root)
        item = {
            "filename": obj_name,
            "file": obj_name,
            "mesh_file": mesh_file,
            "experiment_files": mesh_experiments,
            "run_root": run_root,
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "score": score.get("active_score", score),
            "raw_score": score.get("raw_score", manifest.get("raw_score")),
            "corrected_score": score.get("corrected_score", manifest.get("corrected_score")),
            "material_params": manifest.get("material_params", material_params),
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
            **metrics,
        }
        if completed.returncode != 0:
            item["error"] = completed.stderr[-2000:] or completed.stdout[-2000:] or "plasticity_simulation.py failed"
            errors.append(item)
        else:
            record_simulated_file(obj_name, "plasticity")
        results.append(item)

    result = {
        "success": bool(results) and not errors,
        "analysis_type": "plasticity_simulation",
        "tool": "tools/plasticity_simulation.py",
        "params": params,
        "results": results,
        "errors": errors,
    }

    tool_tracker = get_tool_call_tracker()
    tool_tracker.record_tool_call(
        agent_name=get_current_agent(),
        tool_name="run_plasticity_simulation",
        params=params,
        result=result,
        success=result["success"],
    )

    prop_table = get_property_table()
    prop_table.update_from_plasticity_simulation(result)
    if result["success"]:
        _inject_to_saes(result, "plasticity_simulation")

    tracker = get_global_tracker()
    if tracker and result["success"]:
        tracker.record_simulation_result("Plasticity Simulation", result)

    return result

run_plasticity_simulation_wrapper.tool_info = {
    "tool_name": "run_plasticity_simulation",
    "tool_title": "Plasticity Simulation",
    "tool_description": "Run J2 plastic compression simulation with tools/plasticity_simulation.py. Requires a .msh mesh and experimental curve TXT file.",
    "tool_params": [
        {"name": "input_path", "description": "[Required] Directory containing .msh meshes, or path to one .msh file", "type": "string", "required": True},
        {"name": "target_files", "description": "Optional target .msh file list; takes precedence over automatic input_path discovery", "type": "array", "items": {"type": "string"}, "required": False},
        {"name": "experiment_files", "description": "Experimental curve TXT files; automatically looks for mesh_file + '.txt' when omitted", "type": "array", "items": {"type": "string"}, "required": False},
        {"name": "output_dir", "description": "Root directory for plasticity simulation output", "type": "string", "required": False, "default": "data/plasticity_results"},
        {"name": "strain_limit", "description": "Maximum engineering strain", "type": "number", "required": False, "default": 0.05},
        {"name": "steps", "description": "Number of loading steps", "type": "integer", "required": False, "default": 100},
        {"name": "gpu_device_id", "description": "Primary GPU device ID", "type": "integer", "required": False, "default": 0},
        {"name": "custom_E", "description": "[Required] Base-material Young's modulus (MPa); see the material table", "type": "number", "required": True},
        {"name": "custom_nu", "description": "[Required] Base-material Poisson's ratio; see the material table", "type": "number", "required": True},
        {"name": "custom_sig0", "description": "[Required] Base-material initial yield strength (MPa); see the material table", "type": "number", "required": True},
        {"name": "custom_H1", "description": "[Required] Base-material hardening modulus (MPa); see the material table", "type": "number", "required": True},
        {"name": "custom_Q_inf", "description": "Saturation hardening parameter Q_inf", "type": "number", "required": False, "default": 18.0},
        {"name": "custom_b", "description": "Saturation hardening parameter b", "type": "number", "required": False, "default": 8.0},
        {"name": "custom_eta", "description": "Viscous regularization parameter eta", "type": "number", "required": False, "default": 1.0},
        {"name": "specimen_height_mm", "description": "Specimen height (mm)", "type": "number", "required": False, "default": 40.0},
        {"name": "specimen_width_mm", "description": "Specimen width (mm)", "type": "number", "required": False, "default": 40.0},
        {"name": "specimen_depth_mm", "description": "Specimen depth (mm)", "type": "number", "required": False, "default": 40.0},
        {"name": "boundary_mode", "description": "Boundary mode: rough_platen, smooth_platen, or frictionless_platen", "type": "string", "required": False, "default": "rough_platen"},
        {"name": "real_dimension_mm", "description": "Optional target mesh-scaling dimension (mm)", "type": "number", "required": False},
        {"name": "no_mesh_scaling", "description": "Whether to disable mesh scaling", "type": "boolean", "required": False, "default": True},
        {"name": "enable_multi_gpu", "description": "Whether to enable multi-GPU parallelism (disabled by default)", "type": "boolean", "required": False, "default": False}
    ]
}

def get_heat_analysis_files_wrapper() -> list:
    """Get list of files for heat analysis."""
    # Assuming this function exists in the original tool or we implement a simple version
    # The original agent imported it.
    from tools.heat_analysis_tool import get_heat_analysis_files
    return get_heat_analysis_files()

get_heat_analysis_files_wrapper.tool_info = {
    "tool_name": "get_heat_analysis_files",
    "tool_title": "Get Thermal Analysis Files",
    "tool_description": "List microstructure files available for thermal conductivity analysis.",
    "tool_params": []
}

# Wrapper for electrical conductivity analysis
def run_electrical_analysis_wrapper(
    input_path: str,
    output_dir: str = None,
    resolution: int = 64,
    base_conductivity: float = 1.0,
    use_gpu: bool = False,
    enable_visualization: bool = False,
    gpu_device: str = 'cuda:0',
    solver_tol: float = 1e-6,
    solver_max_iter: int = 5000,
    analysis_type: str = 'auto',
    enable_comparison: bool = True,
    save_detailed_results: bool = True
) -> Dict[str, Any]:
    """Run integrated electrical conductivity analysis."""
    from tools.electrical_conductivity_tool import (
        run_integrated_electrical_conductivity_analysis,
    )
    
    # Select only files not yet simulated for electrical conductivity analysis.
    filtered_files = filter_unsimulated_files(input_path, "electrical")
    if not filtered_files:
        print("   All files have completed electrical conductivity simulation; skipping")
        return {
            "success": True,
            "message": "All files already simulated for electrical analysis",
            "skipped": True,
            "results": []
        }
    
    # For a directory input, create a temporary directory containing only unsimulated files.
    temp_dir_created = False
    original_input_path = input_path
    if os.path.isdir(input_path) and len(filtered_files) < len(glob.glob(os.path.join(input_path, "*.obj"))):
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="elec_sim_")
        temp_dir_created = True
        for f in filtered_files:
            basename = os.path.basename(f)
            link_path = os.path.join(temp_dir, basename)
            try:
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(f, link_path)
                else:  # Unix
                    os.symlink(f, link_path)
            except:
                import shutil
                shutil.copy2(f, link_path)
        input_path = temp_dir
        print(f"   Created temporary directory: {temp_dir} ({len(filtered_files)} files awaiting simulation)")
    
    params = {
        "input_path": original_input_path,
        "output_dir": output_dir,
        "resolution": resolution,
        "base_conductivity": base_conductivity,
        "use_gpu": use_gpu
    }
    
    try:
        result = run_integrated_electrical_conductivity_analysis(
            input_path=input_path,
            output_dir=output_dir,
            resolution=resolution,
            base_conductivity=base_conductivity,
            use_gpu=use_gpu,
            enable_visualization=enable_visualization,
            gpu_device=gpu_device,
            solver_tol=solver_tol,
            solver_max_iter=solver_max_iter,
            analysis_type=analysis_type,
            enable_comparison=enable_comparison,
            save_detailed_results=save_detailed_results
        )
        
        success = result.get("success", False)
        
        # Record files from this run as having completed electrical analysis.
        if success:
            record_simulated_files_from_results(result, "electrical")
        
        # Remove the temporary directory.
        if temp_dir_created:
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"   Removed temporary directory: {temp_dir}")
            except:
                pass
        
        # Record tool call to tracker
        tool_tracker = get_tool_call_tracker()
        tool_tracker.record_tool_call(
            agent_name=get_current_agent(),
            tool_name="run_integrated_electrical_conductivity_analysis",
            params=params,
            result=result,
            success=success
        )
        
        # Update microstructure property table
        prop_table = get_property_table()
        prop_table.update_from_electrical_analysis(result)
        
        # SAES: inject simulation results into the population.
        _inject_to_saes(result, "electrical_analysis")
        
        # Record to tree tracker
        tracker = get_global_tracker()
        if tracker and success:
            tracker.record_simulation_result("Electrical Conductivity Analysis", result)
            
        return result
        
    except Exception as e:
        # Record failed tool call
        tool_tracker = get_tool_call_tracker()
        tool_tracker.record_tool_call(
            agent_name=get_current_agent(),
            tool_name="run_integrated_electrical_conductivity_analysis",
            params=params,
            result={"error": str(e)},
            success=False
        )
        raise

run_electrical_analysis_wrapper.tool_info = {
    "tool_name": "run_integrated_electrical_conductivity_analysis",
    "tool_title": "Electrical Conductivity Analysis",
    "tool_description": "Simulate the electrical conductivity of a microstructure.",
    "tool_params": [
        {"name": "input_path", "description": "Input file path", "type": "string", "required": True},
        {"name": "output_dir", "description": "Output directory", "type": "string", "required": False},
        {"name": "base_conductivity", "description": "Base-material electrical conductivity", "type": "number", "required": False, "default": 1.0},
        {"name": "use_gpu", "description": "Whether to use a GPU", "type": "boolean", "required": False, "default": False}
    ]
}

def get_electrical_analysis_files_wrapper() -> list:
    """Get list of files for electrical analysis."""
    from tools.electrical_conductivity_tool import get_electrical_analysis_files

    return get_electrical_analysis_files()

get_electrical_analysis_files_wrapper.tool_info = {
    "tool_name": "get_electrical_analysis_files",
    "tool_title": "Get Electrical Analysis Files",
    "tool_description": "List microstructure files available for electrical conductivity analysis.",
    "tool_params": []
}
