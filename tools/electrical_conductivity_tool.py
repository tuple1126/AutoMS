"""
Electrical conductivity analysis tool.
Used by the optimization-planning agent to invoke
integrated_electrical_conductivity_analysis.py for electrical analysis.
"""

import os
import sys
import json
import numpy as np
import subprocess
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path.
current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(current_dir)
sys.path.insert(0, main_dir)

# Import through the package first so the tool works from the project root.
try:
    from tools.integrated_electrical_conductivity_analysis import (
        ElectricalConductivityAnalyzer,
        Config as ElectricalConfig,
        VoxelGenerator,
        NumericalHomogenization
    )
    ELECTRICAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    try:
        from integrated_electrical_conductivity_analysis import (
            ElectricalConductivityAnalyzer,
            Config as ElectricalConfig,
            VoxelGenerator,
            NumericalHomogenization
        )
        ELECTRICAL_ANALYSIS_AVAILABLE = True
    except ImportError:
        print(f"Warning: unable to import the electrical analysis module: {e}")
        ELECTRICAL_ANALYSIS_AVAILABLE = False


def run_integrated_electrical_conductivity_analysis(
    input_path: str,
    output_dir: str = None,
    resolution: int = 64,
    mode: str = 'solid',
    base_conductivity: float = 1.0,
    material_electrical_conductivity: float = 1.0,
    use_gpu: bool = False,
    enable_visualization: bool = False,
    gpu_device: str = 'cuda:0',
    solver_tol: float = 1e-5,
    solver_max_iter: int = 5000,
    analysis_type: str = 'auto',
    enable_comparison: bool = True,
    save_detailed_results: bool = True
) -> Dict[str, Any]:
    """
    Integrated electrical conductivity analysis tool.

    Args:
        input_path: Input file or directory.
        output_dir: Result directory. Defaults to ``electrical_results`` in the
            current working directory.
        resolution: Voxel resolution. Defaults to 128.
        mode: Voxelization mode, either ``surface`` or ``solid``. Defaults to
            ``solid``.
        base_conductivity: Base conductivity used by the solver. Defaults to 1.0.
        material_electrical_conductivity: Physical base-material conductivity;
            results are scaled by this value. Defaults to 1.0.
        use_gpu: Whether to use a GPU. Defaults to True.
        enable_visualization: Whether to generate visualization files. Defaults
            to False.
        gpu_device: GPU device. Defaults to ``cuda:0``.
        solver_tol: Solver tolerance. Defaults to 1e-5.
        solver_max_iter: Maximum solver iterations. Defaults to 5000.
        analysis_type: ``single`` (one file), ``batch`` (directory), or ``auto``
            (detect automatically).
        enable_comparison: Whether to compare multiple results. Defaults to True.
        save_detailed_results: Whether to save detailed results. Defaults to True.

    Returns:
        Dict: Complete analysis results, including paths, conductivity data, and
            comparison information.
    """

    if not ELECTRICAL_ANALYSIS_AVAILABLE:
        return {
            "success": False,
            "error": "The electrical analysis module is not installed correctly.",
            "results": []
        }

    try:
        # Detect the analysis type automatically.
        if analysis_type == 'auto':
            if os.path.isfile(input_path):
                analysis_type = 'single'
            elif os.path.isdir(input_path):
                analysis_type = 'batch'
            else:
                return {
                    "success": False,
                    "error": f"Invalid input path: {input_path}",
                    "results": []
                }

        # Configure the output directory.
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), 'electrical_results')

        # Update the analysis configuration.
        ElectricalConfig.OUTPUT_DIR = output_dir
        ElectricalConfig.VOXEL_RESOLUTION = resolution
        ElectricalConfig.VOXEL_MODE = mode
        ElectricalConfig.BASE_CONDUCTIVITY = base_conductivity
        actual_use_gpu = bool(use_gpu)
        if actual_use_gpu:
            try:
                import torch

                actual_use_gpu = torch.cuda.is_available()
            except ImportError:
                actual_use_gpu = False
        ElectricalConfig.USE_GPU = actual_use_gpu
        ElectricalConfig.ENABLE_VISUALIZATION = enable_visualization
        ElectricalConfig.GPU_DEVICE = gpu_device if actual_use_gpu else 'cpu'
        ElectricalConfig.SOLVER_TOL = solver_tol
        ElectricalConfig.SOLVER_MAX_ITER = solver_max_iter

        # Create the analyzer.
        analyzer = ElectricalConductivityAnalyzer(ElectricalConfig)

        results = []
        failed_files = []
        total_files = 0

        if analysis_type == 'single':
            # Analyze one file.
            if not os.path.exists(input_path):
                return {
                    "success": False,
                    "error": f"File does not exist: {input_path}",
                    "results": []
                }

            total_files = 1
            result = analyzer.process_single_file(input_path)
            if result:
                results.append(result)
            else:
                failed_files.append(input_path)

        elif analysis_type == 'batch':
            # Analyze a batch of files.
            if not os.path.exists(input_path):
                return {
                    "success": False,
                    "error": f"Directory does not exist: {input_path}",
                    "results": []
                }

            # Find all OBJ files.
            obj_files = []
            for ext in ['*.obj', '*.OBJ']:
                obj_files.extend(Path(input_path).glob(ext))

            if not obj_files:
                return {
                    "success": False,
                    "error": f"No OBJ files were found in {input_path}",
                    "results": []
                }

            total_files = len(obj_files)

            # Process all files.
            for obj_file in obj_files:
                try:
                    result = analyzer.process_single_file(str(obj_file))
                    if result:
                        results.append(result)
                    else:
                        failed_files.append(str(obj_file))
                except Exception as e:
                    failed_files.append(f"{obj_file}: {str(e)}")

        # Save detailed results.
        if results and save_detailed_results:
            analyzer._save_results(results)

        # Compare results.
        comparison_result = None
        if results and enable_comparison and len(results) > 1:
            comparison_result = _perform_comparison_analysis(results)

        # Generate summary statistics.
        statistics = _generate_statistics(results) if results else None

        return {
            "success": True,
            "analysis_type": analysis_type,
            "input_path": input_path,
            "output_directory": output_dir,
            "configuration": {
                "resolution": resolution,
                "mode": mode,
                "base_conductivity": base_conductivity,
                "material_electrical_conductivity": material_electrical_conductivity,
                "use_gpu": use_gpu,
                "solver_tol": solver_tol,
                "solver_max_iter": solver_max_iter
            },
            "processing_summary": {
                "total_files": total_files,
                "processed_files": len(results),
                "failed_files": len(failed_files),
                "success_rate": len(results) / total_files if total_files > 0 else 0
            },
            "failed_files": failed_files,
            "results": results,
            "comparison_analysis": comparison_result,
            "statistics": statistics,
            "output_files": {
                "csv_summary": os.path.join(output_dir, 'electrical_conductivity_summary.csv') if results else None,
                "detailed_results": os.path.join(output_dir, 'detailed_results.npy') if results else None
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Electrical conductivity analysis failed: {str(e)}",
            "results": []
        }


def _perform_comparison_analysis(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform a simplified comparison analysis."""
    try:
        if not results:
            return None
        
        # Sort by electrical conductivity.
        sorted_results = sorted(results, key=lambda x: x.get('electrical_conductivity', 0), reverse=True)
        
        best = sorted_results[0]
        worst = sorted_results[-1]
        
        return {
            "best_structure": {
                'filename': best['filename'],
                'electrical_conductivity': best['electrical_conductivity'],
                'volume_fraction': best['volume_fraction']
            },
            "worst_structure": {
                'filename': worst['filename'],
                'electrical_conductivity': worst['electrical_conductivity'],
                'volume_fraction': worst['volume_fraction']
            }
        }
        
    except Exception as e:
        return {"error": f"Comparison analysis failed: {str(e)}"}


def _generate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate simplified summary statistics."""
    try:
        if not results:
            return None
            
        conductivity_values = [r['electrical_conductivity'] for r in results if 'electrical_conductivity' in r]
        volume_fraction_values = [r['volume_fraction'] for r in results if 'volume_fraction' in r]
        
        return {
            "sample_count": len(results),
            "electrical_conductivity": {
                "mean": round(float(np.mean(conductivity_values)), 3),
                "min": round(float(np.min(conductivity_values)), 3),
                "max": round(float(np.max(conductivity_values)), 3)
            },
            "volume_fraction": {
                "mean": round(float(np.mean(volume_fraction_values)), 4),
                "min": round(float(np.min(volume_fraction_values)), 4),
                "max": round(float(np.max(volume_fraction_values)), 4)
            }
        }
        
    except Exception as e:
        return {"error": f"Statistical analysis failed: {str(e)}"}


# Preserve the original function name for backward compatibility.
run_electrical_conductivity_analysis = run_integrated_electrical_conductivity_analysis


def get_electrical_analysis_files(directory: str = None) -> List[str]:
    """
    Return OBJ files available for electrical conductivity analysis.

    Args:
        directory: Directory to search. Defaults to ``data/workshop``.

    Returns:
        List[str]: OBJ file paths.
    """
    if directory is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        directory = os.path.join(os.path.dirname(current_dir), 'data', 'workshop')

    if not os.path.exists(directory):
        return []

    obj_files = []
    for ext in ['*.obj', '*.OBJ']:
        obj_files.extend([str(p) for p in Path(directory).glob(ext)])

    return obj_files
