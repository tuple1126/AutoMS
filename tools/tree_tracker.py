import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from tools.project_paths import WORKSHOP_DIR

# Global tracker instance
_global_tracker = None

def get_global_tracker():
    return _global_tracker


class ToolCallTracker:
    """
    Track and validate tool calls to prevent agents from inventing data without
    invoking a tool. This is a key component of the ablation experiment.
    """
    
    def __init__(self):
        self.tool_calls: List[Dict[str, Any]] = []
        self.agent_tool_calls: Dict[str, List[Dict[str, Any]]] = {}  # agent_name -> tool calls
        self.pending_verifications: List[str] = []  # Agents awaiting verification.
        self._last_check_index: int = 0  # Tool-call count at the last checkpoint.
        
    def record_tool_call(self, agent_name: str, tool_name: str, params: Dict[str, Any], 
                         result: Any, success: bool):
        """Record a tool call."""
        call_record = {
            "agent": agent_name,
            "tool": tool_name,
            "params": params,
            "result_summary": self._summarize_result(result),
            "result_data": result,  # Retain the structured result for cross-checking.
            "success": success,
            "timestamp": datetime.now().isoformat()
        }
        self.tool_calls.append(call_record)
        
        if agent_name not in self.agent_tool_calls:
            self.agent_tool_calls[agent_name] = []
        self.agent_tool_calls[agent_name].append(call_record)
        
    def _summarize_result(self, result: Any) -> str:
        """Summarize a tool result without making it excessively long."""
        if result is None:
            return "None"
        result_str = str(result)
        if len(result_str) > 500:
            return result_str[:500] + "...[truncated]"
        return result_str
    
    def get_agent_tool_calls(self, agent_name: str) -> List[Dict[str, Any]]:
        """Return all tool-call records for an agent."""
        return self.agent_tool_calls.get(agent_name, [])
    
    def has_agent_called_tools(self, agent_name: str) -> bool:
        """Check whether an agent has called a tool."""
        return len(self.agent_tool_calls.get(agent_name, [])) > 0
    
    def get_simulation_tool_calls(self, agent_name: str = None) -> List[Dict[str, Any]]:
        """Return simulation-related tool calls for critical verification."""
        simulation_tools = [
            'run_integrated_heat_analysis',
            'run_integrated_electrical_conductivity_analysis', 
            'run_stiffness_analysis',
            'batch_stiffness_analysis',
            'run_plasticity_simulation',
        ]
        
        calls = self.tool_calls if agent_name is None else self.agent_tool_calls.get(agent_name, [])
        return [c for c in calls if c['tool'] in simulation_tools]
    
    def mark_check_point(self):
        """Mark a checkpoint for detecting new calls in the current turn."""
        self._last_check_index = len(self.tool_calls)
    
    def has_new_calls_since_checkpoint(self, agent_name: str = None) -> bool:
        """Check whether new tool calls occurred since the last checkpoint."""
        if agent_name is None:
            return len(self.tool_calls) > self._last_check_index
        else:
            # Check whether this agent made calls after the checkpoint.
            agent_calls = self.agent_tool_calls.get(agent_name, [])
            new_calls = [c for c in agent_calls if self.tool_calls.index(c) >= self._last_check_index 
                        if c in self.tool_calls]
            return len(new_calls) > 0
    
    def get_new_calls_since_checkpoint(self, agent_name: str = None) -> List[Dict[str, Any]]:
        """Return tool calls made since the last checkpoint."""
        new_calls = self.tool_calls[self._last_check_index:]
        if agent_name:
            new_calls = [c for c in new_calls if c['agent'] == agent_name]
        return new_calls
    
    def verify_simulator_integrity(self) -> Dict[str, Any]:
        """
        Verify that Simulator actually ran a simulation.
        This is the core validation that prevents fabricated data.
        
        Core mechanism:
        1. Check for simulation-tool-call records.
        2. Cross-check output values against tool return values.
        
        Returns:
            {
                "valid": bool,           # Whether validation passed.
                "tool_calls_count": int, # Number of tool calls.
                "simulation_calls": [],  # Simulation-tool-call records.
                "verified_values": {},   # Verified values.
                "warning": str,          # Warning message, when applicable.
                "recommendation": str    # Recommended action.
            }
        """
        planner_calls = self.get_agent_tool_calls("Simulator")
        sim_calls = self.get_simulation_tool_calls("Simulator")
        
        result = {
            "valid": False,
            "tool_calls_count": len(planner_calls),
            "simulation_calls": sim_calls,
            "verified_values": {},
            "warning": "",
            "recommendation": ""
        }
        
        if len(planner_calls) == 0:
            result["warning"] = "Simulator did not call any tools. The data may be fabricated."
            result["recommendation"] = "Run Simulator again and ensure it invokes a simulation tool."
        elif len(sim_calls) == 0:
            result["warning"] = "Simulator did not call a simulation tool. The simulation results may be fabricated."
            result["recommendation"] = "Require Simulator to call a simulation tool such as run_integrated_heat_analysis."
        else:
            # Check for successful calls.
            successful_sim_calls = [c for c in sim_calls if c['success']]
            if len(successful_sim_calls) == 0:
                result["warning"] = "All simulation tool calls failed. No valid simulation results are available."
                result["recommendation"] = "Check simulation parameters and input files, then rerun the simulation."
            else:
                # Extract values returned by tools for cross-checking.
                result["verified_values"] = self._extract_verified_values(successful_sim_calls)
                result["valid"] = True
                result["warning"] = ""
                result["recommendation"] = f"Verified {len(successful_sim_calls)} successful simulation calls."
        
        return result
    
    def _extract_verified_values(self, sim_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract verified values from simulation-tool-call results.
        These values are used to cross-check agent output.
        
        Prefer structured result_data, with a regular-expression fallback to
        result_summary.
        """
        verified = {}
        
        for call in sim_calls:
            tool_name = call.get('tool', '')
            result_data = call.get('result_data', {})
            
            # Prefer values from structured data.
            if isinstance(result_data, dict):
                results_list = result_data.get('results', [])
                
                # Extract thermal conductivity.
                if 'heat' in tool_name.lower() or 'thermal' in tool_name.lower():
                    thermal_values = []
                    for r in results_list:
                        if isinstance(r, dict) and 'thermal_conductivity' in r:
                            thermal_values.append(r['thermal_conductivity'])
                    if thermal_values:
                        # Save all values and their summary statistics.
                        verified['thermal_conductivity'] = {
                            'values': thermal_values,
                            'min': min(thermal_values),
                            'max': max(thermal_values),
                            'mean': sum(thermal_values) / len(thermal_values)
                        }
                
                # Extract electrical conductivity.
                if 'electrical' in tool_name.lower():
                    elec_values = []
                    for r in results_list:
                        if isinstance(r, dict) and 'electrical_conductivity' in r:
                            elec_values.append(r['electrical_conductivity'])
                    if elec_values:
                        verified['electrical_conductivity'] = {
                            'values': elec_values,
                            'min': min(elec_values),
                            'max': max(elec_values),
                            'mean': sum(elec_values) / len(elec_values)
                        }
                
                # Extract stiffness/Young's modulus.
                if 'stiffness' in tool_name.lower():
                    modulus_values = []
                    for r in results_list:
                        if isinstance(r, dict):
                            # Try several possible key names.
                            for key in ['youngs_modulus', 'E_eff', 'elastic_modulus']:
                                if key in r:
                                    modulus_values.append(r[key])
                                    break
                    if modulus_values:
                        verified['youngs_modulus'] = {
                            'values': modulus_values,
                            'min': min(modulus_values),
                            'max': max(modulus_values),
                            'mean': sum(modulus_values) / len(modulus_values)
                        }
        
        return verified
    
    def cross_reference_output(self, agent_output: str) -> Dict[str, Any]:
        """
        Cross-check values claimed in agent output against tool-call results.
        
        Validation logic:
        1. Extract all verified value ranges from tool-call results.
        2. Extract claimed values from agent output, excluding base-material inputs.
        3. Check whether claimed values are within verified ranges, allowing 5% tolerance.
        
        Args:
            agent_output: Text produced by an agent.
            
        Returns:
            {
                "matched": bool,           # Whether values match.
                "claimed_values": {},      # Values claimed by the agent.
                "verified_values": {},     # Values returned by tools.
                "discrepancies": [],       # Mismatched items.
                "tolerance": 0.05          # Permitted relative error (5%).
            }
        """
        import re
        
        result = {
            "matched": True,
            "claimed_values": {},
            "verified_values": {},
            "discrepancies": [],
            "tolerance": 0.05
        }
        
        # Get verified values.
        verification = self.verify_simulator_integrity()
        result["verified_values"] = verification.get("verified_values", {})
        
        if not result["verified_values"]:
            # Avoid a false negative when calls succeeded but expose no values.
            if verification.get("valid", False):
                result["matched"] = True
                return result
            result["matched"] = False
            result["discrepancies"].append("Unable to obtain verified values for comparison")
            return result
        
        # Common base-material values are inputs, not simulation outputs.
        # Al6061: 167 W/(m*K), Copper: 386-401 W/(m*K), Steel: 45-65 W/(m*K)
        BASE_MATERIAL_THERMAL_CONDUCTIVITY = {
            167, 167.0,      # Al6061 aluminum substrate.
            386, 401,        # Pure copper.
            205, 237,        # Aluminum alloys.
            45, 50, 65,      # Steel.
            16, 16.3,        # Stainless steel.
            156, 156.0,      # Silicon.
        }
        
        # Extract values claimed in agent output.
        # Thermal conductivity: match values shown in tables.
        thermal_values = []
        thermal_patterns = [
            r'(\d+\.?\d*)\s*W/\(m[\*.]K\)',
            r'thermal conductivity:\s*(\d+\.?\d*)',
            r'thermal[_\s]?conductivity[:\s]*(\d+\.?\d*)',
        ]
        for pattern in thermal_patterns:
            matches = re.findall(pattern, agent_output, re.IGNORECASE)
            for m in matches:
                val = float(m)
                # Ignore common base-material values, which are inputs.
                if val not in BASE_MATERIAL_THERMAL_CONDUCTIVITY:
                    thermal_values.append(val)
        if thermal_values:
            result["claimed_values"]["thermal_conductivity"] = thermal_values
        
        # Electrical conductivity: match scientific notation and plain values.
        elec_values = []
        elec_patterns = [
            r'(\d+[,\d]*\.?\d*)\s*S/m',
            r'electrical conductivity:\s*(\d+[,\d]*\.?\d*)',
            r'(\d+\.?\d*)[eE][+-]?\d+',  # Scientific notation.
        ]
        for pattern in elec_patterns:
            matches = re.findall(pattern, agent_output, re.IGNORECASE)
            for m in matches:
                try:
                    # Parse values that include thousands separators.
                    val = float(m.replace(',', ''))
                    elec_values.append(val)
                except:
                    pass
        if elec_values:
            result["claimed_values"]["electrical_conductivity"] = elec_values
        
        # Cross-check whether claimed values are within verified ranges.
        for prop_name, claimed_values in result["claimed_values"].items():
            if prop_name in result["verified_values"]:
                verified_data = result["verified_values"][prop_name]
                
                if isinstance(verified_data, dict) and 'values' in verified_data:
                    verified_values = verified_data['values']
                    verified_min = verified_data.get('min', min(verified_values))
                    verified_max = verified_data.get('max', max(verified_values))
                    
                    # Check whether each claimed value matches a verified value.
                    for claimed in claimed_values:
                        found_match = False
                        for verified in verified_values:
                            if verified != 0:
                                relative_error = abs(claimed - verified) / abs(verified)
                                if relative_error <= result["tolerance"]:
                                    found_match = True
                                    break
                        
                        # If there is no exact match, check the expanded range.
                        if not found_match:
                            # Allow a range from min - 10% to max + 10%.
                            range_min = verified_min * 0.9
                            range_max = verified_max * 1.1
                            if range_min <= claimed <= range_max:
                                found_match = True
                        
                        # Report an error only when the value is clearly implausible.
                        if not found_match and claimed > 0:
                            # Avoid false matches against unrelated table values.
                            if claimed < verified_min * 0.5 or claimed > verified_max * 2:
                                result["matched"] = False
                                result["discrepancies"].append(
                                    f"{prop_name}: claimed value={claimed:.2f} is outside the verified range "
                                    f"[{verified_min:.2f}, {verified_max:.2f}]"
                                )
        
        return result
    
    def get_verification_summary(self) -> str:
        """Generate a tool-call verification summary for context injection."""
        lines = []
        lines.append("\n[Tool Call Verification]:")
        
        # Overall statistics.
        total_calls = len(self.tool_calls)
        lines.append(f"- Total tool calls in session: {total_calls}")
        
        # Simulator-specific checks.
        verification = self.verify_simulator_integrity()
        if not verification["valid"]:
            lines.append(f"- VERIFICATION FAILED: {verification['warning']}")
            lines.append(f"- Recommendation: {verification['recommendation']}")
            lines.append("- Status: SIMULATION DATA NOT VERIFIED - DO NOT TRUST REPORTED NUMBERS")
        else:
            lines.append(f"- Simulator tool calls verified: {verification['tool_calls_count']}")
            lines.append(f"- Simulation tool calls: {len(verification['simulation_calls'])}")
            for call in verification['simulation_calls'][-3:]:  # Three most recent calls.
                lines.append(f"    - {call['tool']} @ {call['timestamp']}: success={call['success']}")
            lines.append("- Status: SIMULATION DATA VERIFIED")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear records for a new session."""
        self.tool_calls.clear()
        self.agent_tool_calls.clear()


class MicrostructurePropertyTable:
    """
    A consolidated microstructure-property table.

    It associates results computed by simulators with microstructure files to
    form complete property records. Thermal conductivity, electrical
    conductivity, stiffness, plasticity, and other properties are supported.
    """
    
    def __init__(self):
        # Store all properties under the microstructure filename.
        self.properties: Dict[str, Dict[str, Any]] = {}
        self.update_history: List[Dict[str, Any]] = []
        
    def _normalize_filename(self, filename: str) -> str:
        """Normalize a filename so multiple formats map to one microstructure."""
        import os
        # Remove the path.
        basename = os.path.basename(filename)
        # Remove common extensions, including generation-stage .csv and .npy files.
        for ext in ['.obj', '.OBJ', '.msh', '.MSH', '.vtk', '.VTK', '.csv', '.CSV', '.npy', '.NPY']:
            if basename.endswith(ext):
                basename = basename[:-len(ext)]
                break
        return basename
    
    def update_from_heat_analysis(self, result: Dict[str, Any]):
        """Update the property table from heat-conduction results."""
        if not result.get("success"):
            return
            
        results_list = result.get("results", [])
        for item in results_list:
            filename = self._normalize_filename(item.get("filename", ""))
            if not filename:
                continue
                
            if filename not in self.properties:
                self.properties[filename] = {"filename": filename}
            
            self.properties[filename].update({
                "thermal_conductivity": item.get("thermal_conductivity"),
                "thermal_volume_fraction": item.get("volume_fraction"),
            })
            
        self._record_update("heat_analysis", result)
        # [GPAC - IFM] Update the Pareto front after heat-conduction simulation.
        self._trigger_incremental_pareto_update("heat_analysis", len(results_list))
    
    def update_from_electrical_analysis(self, result: Dict[str, Any]):
        """Update the property table from electrical-conduction results."""
        if not result.get("success"):
            return
            
        results_list = result.get("results", [])
        for item in results_list:
            filename = self._normalize_filename(item.get("filename", ""))
            if not filename:
                continue
                
            if filename not in self.properties:
                self.properties[filename] = {"filename": filename}
            
            self.properties[filename].update({
                "electrical_conductivity": item.get("electrical_conductivity"),
                "electrical_volume_fraction": item.get("volume_fraction"),
            })
            
        self._record_update("electrical_analysis", result)
        # [GPAC - IFM] Update the Pareto front after electrical-conduction simulation.
        self._trigger_incremental_pareto_update("electrical_analysis", len(results_list))
    
    def update_from_stiffness_analysis(self, result: Dict[str, Any]):
        """Update the property table from stiffness-analysis results."""
        if not result.get("success"):
            return
        
        # Single-file analysis result.
        if "effective_properties" in result:
            filename = self._normalize_filename(result.get("file", result.get("obj_file", "")))
            if filename:
                if filename not in self.properties:
                    self.properties[filename] = {"filename": filename}
                
                props = result.get("effective_properties", {})
                # Note: stiffness_analysis_tool returns E_avg, G_avg, and nu_avg.
                self.properties[filename].update({
                    "youngs_modulus": props.get("E_avg"),  # MPa
                    "shear_modulus": props.get("G_avg"),   # MPa
                    "poisson_ratio": props.get("nu_avg"),
                    "bulk_modulus": props.get("K"),    # MPa
                    "stiffness_volume_fraction": result.get("solid_fraction") or props.get("volume_fraction"),
                })
        
        # Batch analysis results.
        results_list = result.get("results", [])
        for item in results_list:
            filename = self._normalize_filename(item.get("filename", item.get("file", "")))
            if not filename:
                continue
                
            if filename not in self.properties:
                self.properties[filename] = {"filename": filename}
            
            props = item.get("effective_properties", {})
            # Note: stiffness_analysis_tool returns E_avg, G_avg, and nu_avg.
            self.properties[filename].update({
                "youngs_modulus": props.get("E_avg"),
                "shear_modulus": props.get("G_avg"),
                "poisson_ratio": props.get("nu_avg"),
                "bulk_modulus": props.get("K"),
                "stiffness_volume_fraction": item.get("solid_fraction", item.get("volume_fraction")),
            })
            
        self._record_update("stiffness_analysis", result)
        # [GPAC - IFM] Update the Pareto front after stiffness simulation.
        self._trigger_incremental_pareto_update("stiffness_analysis", len(results_list) or 1)
    
    def update_from_plasticity_simulation(self, result: Dict[str, Any]):
        """Update the property table from plasticity-simulation results."""
        results_list = result.get("results", [])
        if not result.get("success") and not any(item.get("success") for item in results_list):
            return

        for item in results_list:
            if item.get("success") is False:
                continue
            filename = self._normalize_filename(item.get("filename", item.get("file", "")))
            if not filename:
                continue
                
            if filename not in self.properties:
                self.properties[filename] = {"filename": filename}
            
            self.properties[filename].update({
                "yield_strength": item.get("yield_strength"),
                "ultimate_strength": item.get("ultimate_strength"),
                "plastic_strain": item.get("plastic_strain"),
                "specific_energy": item.get("specific_energy"),
                "energy_absorption": item.get("energy_absorption") or item.get("specific_energy"),
                "hardening_modulus": item.get("hardening_modulus"),
                "max_stress": item.get("max_stress"),
                "max_strain": item.get("max_strain"),
            })
            
        self._record_update("plasticity_simulation", result)
        # [GPAC - IFM] Update the Pareto front after plasticity simulation.
        self._trigger_incremental_pareto_update("plasticity_simulation", len(results_list))
    
    def update_from_structure_generation(self, result: Dict[str, Any]):
        """
        Update the property table from AI microstructure-generation results.
        This records mechanical properties (E, G, nu) calculated during
        generation, preventing empty values before subsequent simulations.
        
        [GPAC integration] This is an entry point for Incremental Front
        Maintenance (IFM). Each time StructureGenerator produces new
        structures, it:
        1. Records mechanical properties in the property table.
        2. Triggers an incremental SAES Guidance Pareto-front update.
        3. Supplies data for the next Dominance-Guided Generation (DGG) cycle.
        
        Args:
            result: Result returned by generate_microstructure_with_ai, containing:
                - kept_structures_details: Details for retained structures.
                - microstructure_directory: Microstructure storage directory.
        """
        if "error" in result:
            return
            
        kept_details = result.get("kept_structures_details", [])
        for item in kept_details:
            filename = self._normalize_filename(item.get("filename", ""))
            if not filename:
                continue
                
            if filename not in self.properties:
                self.properties[filename] = {"filename": filename}
            
            # Read mechanical properties from scaled_properties.
            scaled_props = item.get("scaled_properties", {})
            target_props = item.get("target_properties", {})
            relative_errors = item.get("relative_errors", {})
            
            self.properties[filename].update({
                # Calculated properties from homogenization.
                "youngs_modulus": scaled_props.get("E"),     # MPa
                "shear_modulus": scaled_props.get("G"),      # MPa  
                "poisson_ratio": scaled_props.get("nu"),
                # User-specified target properties.
                "target_E": target_props.get("E"),
                "target_G": target_props.get("G"),
                "target_nu": target_props.get("nu"),
                # Relative errors.
                "E_error": relative_errors.get("E"),
                "G_error": relative_errors.get("G"),
                "nu_error": relative_errors.get("nu"),
                # Source marker.
                "source": "ai_generation",
            })
        
        if kept_details:
            self._record_update("structure_generation", {
                "success": True,
                "results": kept_details,
                "directory": result.get("microstructure_directory")
            })
            
            # [GPAC - IFM] Trigger incremental Pareto-front maintenance.
            self._trigger_incremental_pareto_update("structure_generation", len(kept_details))
    
    def _record_update(self, analysis_type: str, result: Dict[str, Any]):
        """Record update history."""
        self.update_history.append({
            "type": analysis_type,
            "timestamp": datetime.now().isoformat(),
            "files_updated": len(result.get("results", [])) or 1,
            "success": result.get("success", False)
        })
    
    def _trigger_incremental_pareto_update(self, source: str, count: int):
        """
        [GPAC feature 4 - Incremental Front Maintenance (IFM)]
        
        Trigger an incremental Pareto-front update whenever new simulation or
        generation results arrive. This GPAC component supports:
        1. Real-time Pareto-front updates without waiting for all simulations.
        2. Dynamic objective-weight adjustment (AWA - Adaptive Weight Adjustment).
        3. Guidance for the next iteration (DGG - Dominance-Guided Generation).
        4. Convergence detection to decide whether another iteration is needed.
        
        Args:
            source: Data source: heat_analysis, electrical_analysis,
                stiffness_analysis, plasticity_simulation, or structure_generation.
            count: Number of updated records.
        """
        try:
            from tools.saes_guidance import get_saes_guidance, is_multi_objective_scenario
            
            # Check whether this is a multi-objective scenario.
            if not is_multi_objective_scenario():
                return
            
            saes_guidance = get_saes_guidance()
            if not saes_guidance.optimizer.objectives:
                return
            
            # 1. Synchronize the current property table with the Pareto optimizer.
            property_list = self.get_property_table()
            saes_guidance.optimizer.add_solutions_from_property_table(property_list)
            
            # 2. Recompute Pareto fronts (non-dominated sorting plus crowding distance).
            saes_guidance.optimizer.compute_pareto_fronts()
            
            # 3. Record this update for iteration-history analysis.
            saes_guidance.record_iteration_update(source, count, len(property_list))
            
            # 4. Optionally check convergence for early termination.
            convergence_info = saes_guidance.check_convergence()
            if convergence_info.get("converged"):
                print(f"[GPAC-IFM] Pareto-front convergence check: {convergence_info.get('reason')}")
            
        except ImportError:
            # Silently ignore an unavailable SAES Guidance module.
            pass
        except Exception as e:
            # Record errors without interrupting the main workflow.
            print(f"[GPAC-IFM Warning] Pareto update failed: {str(e)}")
    
    def get_property_table(self) -> List[Dict[str, Any]]:
        """Return the complete property table as a list."""
        return list(self.properties.values())
    
    def get_property_summary(self) -> str:
        """Generate a property-table summary for context injection."""
        if not self.properties:
            return "\n[Microstructure Property Table]: Empty - No simulation results yet."
        
        lines = []
        lines.append("\n[Microstructure Property Table (Verified Simulation Results)]:")
        lines.append(f"Total microstructures: {len(self.properties)}")
        lines.append("-" * 80)
        
        # Table headers.
        headers = ["Filename", "Thermal(W/(m*K))", "Electrical(S/m)", "E(MPa)", "G(MPa)", "nu", "Vol.Frac"]
        lines.append(f"| {' | '.join(h.center(12) for h in headers)} |")
        lines.append("|" + "-" * (14 * len(headers) + len(headers) - 1) + "|")
        
        # Data rows.
        for filename, props in self.properties.items():
            row = [
                filename[:12] if len(filename) <= 12 else filename[:10] + "..",
                self._format_value(props.get("thermal_conductivity"), "{:.2f}"),
                self._format_value(props.get("electrical_conductivity"), "{:.2e}"),
                self._format_value(props.get("youngs_modulus"), "{:.1f}"),
                self._format_value(props.get("shear_modulus"), "{:.1f}"),
                self._format_value(props.get("poisson_ratio"), "{:.3f}"),
                self._format_value(props.get("thermal_volume_fraction") or props.get("stiffness_volume_fraction"), "{:.3f}"),
            ]
            lines.append(f"| {' | '.join(v.center(12) for v in row)} |")
        
        lines.append("-" * 80)
        
        # Update history.
        if self.update_history:
            lines.append(f"Last updated: {self.update_history[-1]['timestamp']}")
            lines.append(f"Update history: {[h['type'] for h in self.update_history[-5:]]}")
        
        # Add SAES Guidance Pareto analysis for multi-objective scenarios.
        pareto_summary = self._get_pareto_analysis_summary()
        if pareto_summary:
            lines.append(pareto_summary)
        
        return "\n".join(lines)
    
    def _get_pareto_analysis_summary(self) -> str:
        """Return a Pareto-analysis summary from SAES Guidance."""
        try:
            from tools.saes_guidance import get_saes_guidance, is_multi_objective_scenario
            
            saes_guidance = get_saes_guidance()
            if not saes_guidance.optimizer.objectives:
                return ""
            
            # Synchronize the current property table with the Pareto optimizer.
            saes_guidance.optimizer.add_solutions_from_property_table(self.get_property_table())
            saes_guidance.optimizer.compute_pareto_fronts()
            
            return saes_guidance.get_context_injection()
        except ImportError:
            return ""
        except Exception as e:
            return f"\n[SAES Guidance Warning]: {str(e)}"
    
    def _format_value(self, value: Any, fmt: str = "{}") -> str:
        """Format a value."""
        if value is None:
            return "-"
        try:
            return fmt.format(value)
        except:
            return str(value)[:12]
    
    def export_to_csv(self, filepath: str):
        """Export the table to a CSV file."""
        import csv
        
        if not self.properties:
            return False
        
        # Collect all possible columns.
        all_keys = set()
        for props in self.properties.values():
            all_keys.update(props.keys())
        
        columns = ["filename"] + sorted([k for k in all_keys if k != "filename"])
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for props in self.properties.values():
                writer.writerow({k: props.get(k, "") for k in columns})
        
        return True
    
    def export_to_json(self, filepath: str):
        """Export the table to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "properties": self.get_property_table(),
                "update_history": self.update_history,
                "exported_at": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        return True
    
    def clear(self):
        """Clear the property table."""
        self.properties.clear()
        self.update_history.clear()


# Global property table instance
_property_table = None

def get_property_table() -> MicrostructurePropertyTable:
    global _property_table
    if _property_table is None:
        _property_table = MicrostructurePropertyTable()
    return _property_table

def reset_property_table():
    global _property_table
    _property_table = MicrostructurePropertyTable()


# Global tool call tracker instance
_tool_call_tracker = None

def get_tool_call_tracker() -> ToolCallTracker:
    global _tool_call_tracker
    if _tool_call_tracker is None:
        _tool_call_tracker = ToolCallTracker()
    return _tool_call_tracker

def reset_tool_call_tracker():
    global _tool_call_tracker
    _tool_call_tracker = ToolCallTracker()

class TreeNode:
    def __init__(self, node_type: str, name: str, content: Any = None, parent_id: str = None):
        self.id = str(uuid.uuid4())
        self.type = node_type  # e.g., "root", "phase", "agent_message", "tool_call", "artifact"
        self.name = name
        self.content = content or {}
        self.parent_id = parent_id
        self.children: List[TreeNode] = []
        self.timestamp = datetime.now().isoformat()
        self.status = "completed" # pending, in_progress, completed, failed

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "content": self.content,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "children": [child.to_dict() for child in self.children]
        }

    @classmethod
    def from_dict(cls, data):
        node = cls(data["type"], data["name"], data["content"], data["parent_id"])
        node.id = data["id"]
        node.timestamp = data["timestamp"]
        node.status = data.get("status", "completed")
        node.children = [cls.from_dict(child) for child in data["children"]]
        return node

class TreePlanTracker:
    def __init__(self, session_id: str, base_dir: str = "data/case_library_tree"):
        global _global_tracker
        _global_tracker = self
        
        self.session_id = session_id
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        self.root = TreeNode("root", f"Session {session_id}", {"session_id": session_id})
        self.node_map = {self.root.id: self.root}
        self.current_node_id = self.root.id
        
        # Iteration counters used by Manager to determine whether to stop.
        self.iteration_count = 0
        self.max_iterations = 40  # Maximum agent iterations.
        self.design_iteration_count = 0  # Simulator-to-StructureGenerator cycles.
        self.max_design_iterations = 10  # Maximum design iterations.

    def increment_iteration(self):
        """Increment the agent-iteration count."""
        self.iteration_count += 1
        
    def increment_design_iteration(self):
        """Increment the design-iteration count when a new cycle begins."""
        self.design_iteration_count += 1
        
    def get_iteration_status(self) -> dict:
        """Return iteration status."""
        return {
            "current_iteration": self.iteration_count,
            "max_iterations": self.max_iterations,
            "design_iteration": self.design_iteration_count,
            "max_design_iterations": self.max_design_iterations,
            "should_terminate": self.iteration_count >= self.max_iterations or self.design_iteration_count >= self.max_design_iterations
        }

    def add_node(self, node_type: str, name: str, content: Any = None, parent_id: str = None) -> str:
        """
        Add a new node to the tree.
        If parent_id is not provided, adds to the current_node_id.
        Returns the new node's ID.
        """
        if parent_id is None:
            parent_id = self.current_node_id
        
        parent_node = self.node_map.get(parent_id)
        if not parent_node:
            # Fallback to root if parent not found (shouldn't happen usually)
            parent_node = self.root
            parent_id = self.root.id

        new_node = TreeNode(node_type, name, content, parent_id)
        parent_node.children.append(new_node)
        self.node_map[new_node.id] = new_node
        return new_node.id

    def update_node(self, node_id: str, content_update: Dict = None, status: str = None):
        """Update an existing node."""
        node = self.node_map.get(node_id)
        if node:
            if content_update:
                node.content.update(content_update)
            if status:
                node.status = status

    def set_current_node(self, node_id: str):
        """Set the pointer for where the next node will be added by default."""
        if node_id in self.node_map:
            self.current_node_id = node_id

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        return self.node_map.get(node_id)

    def save_tree(self):
        """Save the entire tree to a JSON file."""
        file_path = os.path.join(self.base_dir, f"{self.session_id}_tree.json")
        
        # Custom encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                    np.int16, np.int32, np.int64, np.uint8,
                    np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, 
                    np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.root.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        return file_path

    def load_tree(self, file_path: str):
        """Load a tree from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.root = TreeNode.from_dict(data)
        # Rebuild node_map
        self.node_map = {}
        self._traverse_and_map(self.root)
        self.current_node_id = self.root.id

    def _traverse_and_map(self, node: TreeNode):
        self.node_map[node.id] = node
        for child in node.children:
            self._traverse_and_map(child)

    def scan_workspace_artifacts(self, workspace_dirs: List[str] = None) -> List[str]:
        """
        Scan specific directories for files to verify existence of artifacts.
        Returns a list of found file paths (relative to workspace root if possible).
        """
        if workspace_dirs is None:
            default_workshop = str(WORKSHOP_DIR)
            if os.path.exists(default_workshop):
                workspace_dirs = [default_workshop]
            else:
                workspace_dirs = ["data/workshop"]
        
        found_files = []
        for dir_name in workspace_dirs:
            if os.path.exists(dir_name):
                for root, _, files in os.walk(dir_name):
                    for file in files:
                        # Skip hidden files or pycache
                        if file.startswith('.') or '__pycache__' in root:
                            continue
                        full_path = os.path.join(root, file)
                        found_files.append(full_path)
        return found_files

    def get_context_summary(self) -> str:
        """
        Generate a summary of the current session context to prevent forgetting and hallucination.
        """
        summary = []
        summary.append("=== Current Session Context ===")
        summary.append(f"Session ID: {self.session_id}")
        summary.append(f"Timestamp: {datetime.now().isoformat()}")
        
        # Iteration status is critical for Manager decisions.
        iter_status = self.get_iteration_status()
        summary.append("\n[Iteration Status]:")
        summary.append(f"- Agent Turns: {iter_status['current_iteration']}/{iter_status['max_iterations']}")
        summary.append(f"- Design Iterations: {iter_status['design_iteration']}/{iter_status['max_design_iterations']}")
        if iter_status['should_terminate']:
            summary.append("- ITERATION LIMIT REACHED - MUST TERMINATE")
            summary.append(f"- Action: Select SummaryReporter to generate final report, then TERMINATE")
        elif iter_status['design_iteration'] >= iter_status['max_design_iterations'] - 1:
            summary.append(f"- APPROACHING DESIGN ITERATION LIMIT ({iter_status['design_iteration']}/{iter_status['max_design_iterations']})")
            summary.append(f"- Action: Consider wrapping up if no significant improvement")
        
        # 1. User Requirement (Root)
        if self.root.content and "text" in self.root.content:
            summary.append(f"\n[User Requirement]: {self.root.content['text']}")
        
        # 2. Recent Agent Interactions (Last 3)
        # We traverse the tree to find agent_response nodes
        agent_responses = []
        def collect_responses(node):
            if node.type == "agent_response":
                agent_responses.append(node)
            for child in node.children:
                collect_responses(child)
        collect_responses(self.root)
        
        if agent_responses:
            summary.append("\n[Recent Activity]:")
            for node in agent_responses[-3:]:
                # Truncate long responses
                resp_text = str(node.content.get('response', ''))
                if len(resp_text) > 200:
                    resp_text = resp_text[:200] + "..."
                summary.append(f"- {node.name}: {resp_text}")

        # 3. Verified Artifacts (Grounding) - disabled to keep prompts concise.
        # found_files = self.scan_workspace_artifacts()
        # if found_files:
        #     summary.append("\n[Verified Existing Files (Grounding Truth)]:")
        #     for f in found_files:
        #         summary.append(f"- {f}")
        # else:
        #     summary.append("\n[Verified Existing Files]: None found yet.")

        # 4. Simulation Results Summary
        sim_results = []
        def collect_sim_results(node):
            if node.type == "simulation_artifact":
                sim_results.append(node)
            for child in node.children:
                collect_sim_results(child)
        collect_sim_results(self.root)

        if sim_results:
            summary.append("\n[Simulation Results (Grounding Truth)]:")
            for node in sim_results:
                summary.append(f"- {node.name}: {json.dumps(node.content.get('summary', {}), ensure_ascii=False)}")
        
        # 5. Performance Verification Status (Critical for Manager Decision)
        summary.append("\n[Performance Verification Status]:")
        if not sim_results:
            summary.append("- NO SIMULATION RESULTS RECORDED YET")
            summary.append("- Status: NOT READY TO TERMINATE")
            summary.append("- Action Required: Simulator must complete simulations first.")
        else:
            # Check if we have thermal, electrical, and stiffness results
            sim_types = [node.name.lower() for node in sim_results]
            has_thermal = any('thermal' in t or 'heat' in t for t in sim_types)
            has_electrical = any('electrical' in t for t in sim_types)
            has_stiffness = any('stiffness' in t for t in sim_types)
            
            missing = []
            if not has_thermal:
                missing.append("Thermal Conductivity")
            if not has_electrical:
                missing.append("Electrical Conductivity")
            if not has_stiffness:
                missing.append("Stiffness Analysis")
            
            if missing:
                summary.append(f"- MISSING SIMULATION RESULTS: {', '.join(missing)}")
                summary.append("- Status: NOT READY TO TERMINATE")
                summary.append("- Action Required: Simulator must run missing simulations.")
            else:
                summary.append("- All required simulations completed.")
                summary.append("- Status: READY FOR FINAL EVALUATION")
                summary.append("- Action: SummaryReporter can now generate final report.")

        # 6. Microstructure Property Table (Consolidated Results)
        prop_table = get_property_table()
        summary.append(prop_table.get_property_summary())

        # 7. Tool Call Verification
        tool_tracker = get_tool_call_tracker()
        summary.append(tool_tracker.get_verification_summary())

        # 8. Role Boundary Reminder (Anti-Role-Impersonation - CRITICAL)
        summary.append("\n[Role Boundary Reminder (CRITICAL)]:")
        summary.append("- Each agent has a SPECIFIC role. Do NOT simulate or impersonate other agents.")
        summary.append("- FORBIDDEN: Including '[OtherAgentName]: ...' in your output")
        summary.append("- CORRECT: Only output YOUR role's content, then hand off to the next agent")
        summary.append("- Role Responsibilities:")
        summary.append("  * StructureGenerator: Acquire or generate candidates ONLY")
        summary.append("  * Simulator: Call simulation tools and evaluate ONLY")
        summary.append("  * SummaryReporter: Generate final report ONLY")

        summary.append("\n===============================")
        return "\n".join(summary)

    def record_simulation_result(self, simulation_type: str, results: Dict[str, Any]):
        """
        Specialized method to record simulation results.
        """
        # Create a summary node
        node_content = {
            "type": simulation_type,
            "timestamp": datetime.now().isoformat(),
            "summary": results.get("statistics", {}),
            "details": []
        }
        
        # Extract per-file results if available
        if "results" in results and isinstance(results["results"], list):
            for res in results["results"]:
                # Simplify result for context
                simple_res = {k: v for k, v in res.items() if k in ['filename', 'file', 'volume_fraction', 'thermal_conductivity', 'electrical_conductivity', 'effective_properties', 'solid_fraction', 'stiffness_matrix']}
                node_content["details"].append(simple_res)
        
        self.add_node("simulation_artifact", f"{simulation_type} Results", node_content)
