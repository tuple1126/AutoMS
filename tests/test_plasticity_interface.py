"""Static regression checks for the restored original plasticity interface."""

import ast
import inspect
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOLVER_PATH = ROOT / "tools" / "plasticity_simulation.py"
WRAPPER_PATH = ROOT / "tools" / "simulation_tools_wrapper.py"

EXPECTED_PARAMETERS = [
    "input_path",
    "target_files",
    "experiment_files",
    "output_dir",
    "strain_limit",
    "steps",
    "gpu_device_id",
    "custom_E",
    "custom_nu",
    "custom_sig0",
    "custom_H1",
    "custom_Q_inf",
    "custom_b",
    "custom_eta",
    "specimen_height_mm",
    "specimen_width_mm",
    "specimen_depth_mm",
    "boundary_mode",
    "real_dimension_mm",
    "no_mesh_scaling",
    "enable_multi_gpu",
]


def find_function(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function not found: {name}")


def find_tool_info(module: ast.Module) -> dict:
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "run_plasticity_simulation_wrapper"
                and target.attr == "tool_info"
            ):
                return ast.literal_eval(node.value)
    raise AssertionError("run_plasticity_simulation_wrapper.tool_info not found")


class PlasticityInterfaceTests(unittest.TestCase):
    def test_solver_keeps_the_original_mesh_cli(self):
        source = SOLVER_PATH.read_text(encoding="utf-8")
        ast.parse(source, filename=str(SOLVER_PATH))

        self.assertIn('parser.add_argument("--mesh-file", required=True)', source)
        self.assertIn('parser.add_argument("--experiment-files", nargs="+", required=True)', source)
        self.assertIn("from jax_fem.problem import Problem", source)
        self.assertNotIn("--voxel-file", source)
        self.assertNotIn("PAPER_RESOLUTION", source)

    def test_agent_tool_schema_matches_the_original_mesh_contract(self):
        source = WRAPPER_PATH.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(WRAPPER_PATH))
        function = find_function(module, "run_plasticity_simulation_wrapper")
        parameters = [argument.arg for argument in function.args.args]
        tool_info = find_tool_info(module)
        declared_parameters = [item["name"] for item in tool_info["tool_params"]]

        self.assertEqual(parameters, EXPECTED_PARAMETERS)
        self.assertEqual(declared_parameters, EXPECTED_PARAMETERS)
        self.assertEqual(tool_info["tool_name"], "run_plasticity_simulation")
        self.assertIn("plasticity_simulation.py", tool_info["tool_description"])
        self.assertIn(".msh", tool_info["tool_description"])
        self.assertIn('"--mesh-file"', source)
        self.assertIn('"--experiment-files"', source)
        self.assertNotIn('"--voxel-file"', source)
        self.assertNotIn("custom_epsilon0", source)

    def test_wrapper_loads_without_unrelated_simulation_dependencies(self):
        from tools.simulation_tools_wrapper import run_plasticity_simulation_wrapper

        parameters = list(inspect.signature(run_plasticity_simulation_wrapper).parameters)
        self.assertEqual(parameters, EXPECTED_PARAMETERS)
        with self.assertRaises(ValueError):
            run_plasticity_simulation_wrapper(
                "",
                custom_E=1.0,
                custom_nu=0.3,
                custom_sig0=1.0,
                custom_H1=1.0,
            )


if __name__ == "__main__":
    unittest.main()
