"""
SAES guidance helpers.

This module contains the Pareto analysis, adaptive objective weighting, and
reporting helpers used inside the SAES loop. They are implementation details of
SAES rather than a separate public algorithm.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from tools.pareto_optimizer import (
    ParetoOptimizer, 
    Objective, 
    ObjectiveType,
    Solution,
    get_pareto_optimizer
)


@dataclass
class IterationRecord:
    """Record for one optimization iteration."""
    iteration_id: int
    timestamp: str
    source: str  # "database" or "ai_generated" 
    solutions_added: int
    pareto_improvement: bool  # Whether the Pareto front improved.
    hypervolume_delta: float  # Hypervolume change.
    guidance_used: Dict[str, Any]
    # Per-objective improvement details: {obj_name: {"improved": bool, "delta": float, "best_before": float, "best_after": float}}.
    objective_improvements: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.objective_improvements is None:
            self.objective_improvements = {}
    

class SAESGuidance:
    """
    Multi-objective guidance strategy.

    Core responsibilities:
    1. Parse user requirements and identify multi-objective scenarios.
    2. Provide focused parameter recommendations to agents.
    3. Track iteration history and adapt the strategy.
    4. Generate a final Pareto-optimal-solution report.
    """
    
    def __init__(self):
        self.optimizer = get_pareto_optimizer()
        self.iteration_history: List[IterationRecord] = []
        self.current_iteration = 0
        self.max_iterations = 10
        self.convergence_threshold = 0.02
        self.stagnation_window = 3
        self.min_iterations_before_convergence = 3
        self._weight_history: List[Dict[str, float]] = []
        
    def parse_multi_objective_requirement(self, parsed_req: Dict[str, Any]) -> bool:
        """
        Parse multi-objective requirements from RequirementParser output.

        Returns whether this is a multi-objective optimization scenario.
        """
        target_properties = parsed_req.get("target_properties", {})
        
        # Identify a multi-objective scenario.
        is_multi_objective = len(target_properties) >= 2
        
        if is_multi_objective:
            print(f"[SAES Guidance] Multi-objective scenario detected; objective count: {len(target_properties)}")
            self.optimizer.register_objectives_from_requirement(parsed_req)
            
            # Analyze potential conflicts between objectives.
            conflicts = self._analyze_objective_conflicts(target_properties)
            if conflicts:
                print(f"[SAES Guidance] Potential objective conflicts detected: {conflicts}")
        
        return is_multi_objective
    
    def _analyze_objective_conflicts(self, properties: Dict) -> List[Tuple[str, str]]:
        """Analyze potential conflicts between objectives."""
        conflicts = []
        
        # Known conflict pairs.
        conflict_pairs = [
            ("thermal_conductivity", "volume_fraction"),  # High thermal conductivity versus low mass.
            ("youngs_modulus", "volume_fraction"),        # High stiffness versus low mass.
            ("electrical_conductivity", "thermal_conductivity"),  # In some applications.
        ]
        
        prop_names = set(properties.keys())
        for p1, p2 in conflict_pairs:
            # Check alternate property names.
            p1_exists = p1 in prop_names or p1.replace("_", "") in prop_names
            p2_exists = p2 in prop_names or p2.replace("_", "") in prop_names or "vof" in prop_names
            
            if p1_exists and p2_exists:
                conflicts.append((p1, p2))
        
        return conflicts
    
    def generate_structure_generator_params(self) -> Dict[str, Any]:
        """
        Generate parameter suggestions for StructureGenerator.

        Suggestions adapt dynamically to the current Pareto-front state.
        """
        guidance = self.optimizer.generate_guidance_for_structure_generator()
        
        # Apply iteration-aware adaptation.
        if self.current_iteration > 0:
            guidance = self._apply_adaptive_adjustment(guidance, "structure_generator")
        
        # Convert to a format StructureGenerator can use directly.
        params = {
            "E": None,
            "G": None, 
            "nu": None,
            "exploration_mode": "balanced",  # "exploitation", "exploration", "balanced"
            "diversity_weight": 0.3,
        }
        
        suggested = guidance.get("suggested_params", {})
        exploration = guidance.get("exploration_directions", {})
        
        # Map parameter names.
        param_mapping = {
            "youngs_modulus": "E",
            "E": "E",
            "shear_modulus": "G",
            "G": "G",
            "poisson_ratio": "nu",
            "nu": "nu"
        }
        
        for src_name, target_name in param_mapping.items():
            if src_name in suggested:
                params[target_name] = suggested[src_name]
            elif src_name in exploration:
                params[target_name] = exploration[src_name].get("suggested_target")
        
        # Select the exploration mode.
        pareto_size = guidance.get("current_pareto_size", 0)
        if pareto_size < 3:
            params["exploration_mode"] = "exploration"  # Early phase: broad exploration.
            params["diversity_weight"] = 0.5
        elif pareto_size < 8:
            params["exploration_mode"] = "balanced"
            params["diversity_weight"] = 0.3
        else:
            params["exploration_mode"] = "exploitation"  # Convergence phase.
            params["diversity_weight"] = 0.1
        
        params["_guidance"] = guidance
        params["_iteration"] = self.current_iteration
        
        return params
    
    def generate_retrieval_filter(self) -> Dict[str, Any]:
        """
        Generate database filtering conditions for Generator's internal retrieval branch.

        Under-covered Pareto-front regions produce focused retrieval conditions.
        """
        guidance = self.optimizer.generate_guidance_for_retrieval()
        
        # Apply iteration-aware adaptation.
        if self.current_iteration > 0:
            guidance = self._apply_adaptive_adjustment(guidance, "retrieval")
        
        # Generate a Python filtering-code snippet.
        filter_conditions = guidance.get("filter_conditions", {})
        vof_range = guidance.get("volume_fraction_range", (0.15, 0.35))
        
        code_snippet = self._generate_filter_code_snippet(filter_conditions, vof_range)
        
        return {
            "filter_conditions": filter_conditions,
            "vof_range": vof_range,
            "code_snippet": code_snippet,
            "recommendation": guidance.get("recommendation", ""),
            "priority_fields": self._get_priority_filter_fields(),
            "_guidance": guidance,
            "_iteration": self.current_iteration
        }
    
    def _generate_filter_code_snippet(self, conditions: Dict, vof_range: Tuple) -> str:
        """Generate a Python snippet for database filtering."""
        lines = [
            "# Multi-objective filtering conditions generated by SAES Guidance",
            f"# Iteration: {self.current_iteration}",
            "",
            "# Volume-fraction constraint (primary filter)",
            f"df_filtered = df[(df['vof'] >= {vof_range[0]}) & (df['vof'] <= {vof_range[1]})]",
            ""
        ]
        
        # Add additional conditions.
        for prop_name, cond in conditions.items():
            if "min" in cond and "max" in cond:
                col_name = self._get_csv_column_name(prop_name)
                if col_name:
                    lines.append(f"# {prop_name} constraint")
                    lines.append(f"df_filtered = df_filtered[(df_filtered['{col_name}'] >= {cond['min']:.4f}) & (df_filtered['{col_name}'] <= {cond['max']:.4f})]")
        
        lines.append("")
        lines.append("# Sort by weighted multi-objective score")
        lines.append("# df_filtered['score'] = ...")
        
        return "\n".join(lines)
    
    def _get_csv_column_name(self, prop_name: str) -> Optional[str]:
        """Map a property name to a CSV column name."""
        mapping = {
            "youngs_modulus": "E",
            "E": "E",
            "shear_modulus": "G", 
            "G": "G",
            "poisson_ratio": "nu",
            "nu": "nu",
            "volume_fraction": "vof",
            "vof": "vof"
        }
        return mapping.get(prop_name)
    
    def _get_priority_filter_fields(self) -> List[str]:
        """Return fields to filter first."""
        # Order by objective priority.
        sorted_objectives = sorted(
            self.optimizer.objectives.items(),
            key=lambda x: x[1].priority
        )
        return [obj_name for obj_name, _ in sorted_objectives[:3]]
    
    def _apply_adaptive_adjustment(self, guidance: Dict, agent_type: str) -> Dict:
        """Apply adaptive adjustments."""
        if not self.iteration_history:
            return guidance
        
        # Analyze recent iteration outcomes.
        recent = self.iteration_history[-3:]
        improvements = [r.pareto_improvement for r in recent]
        
        if agent_type == "structure_generator":
            if sum(improvements) == 0:
                # Repeated lack of improvement: increase exploration.
                if "suggested_params" in guidance:
                    for key in guidance["suggested_params"]:
                        # Increase perturbation.
                        guidance["suggested_params"][key] *= (1 + np.random.uniform(-0.15, 0.15))
                guidance["exploration_boost"] = True
                
        elif agent_type == "retrieval":
            if sum(improvements) == 0:
                # Repeated lack of improvement: relax filters.
                if "filter_conditions" in guidance:
                    for key in guidance["filter_conditions"]:
                        cond = guidance["filter_conditions"][key]
                        if "min" in cond and "max" in cond:
                            range_width = cond["max"] - cond["min"]
                            cond["min"] -= range_width * 0.2
                            cond["max"] += range_width * 0.2
                guidance["relaxed_filters"] = True
        
        return guidance
    
    def record_iteration(self, source: str, solutions_added: int, 
                        previous_pareto_size: int, guidance_used: Dict) -> None:
        """Record one iteration."""
        self.current_iteration += 1
        
        # Compute Pareto improvement.
        self.optimizer.compute_pareto_fronts()
        current_pareto_size = len(self.optimizer.pareto_fronts[0]) if self.optimizer.pareto_fronts else 0
        pareto_improvement = current_pareto_size > previous_pareto_size
        
        # Compute hypervolume change (simplified estimate).
        hv_delta = self._estimate_hypervolume_delta()
        
        record = IterationRecord(
            iteration_id=self.current_iteration,
            timestamp=datetime.now().isoformat(),
            source=source,
            solutions_added=solutions_added,
            pareto_improvement=pareto_improvement,
            hypervolume_delta=hv_delta,
            guidance_used=guidance_used
        )
        
        self.iteration_history.append(record)
        
        print(f"[SAES Guidance] Iteration {self.current_iteration}: source={source}, added={solutions_added}, "
              f"pareto_improved={pareto_improvement}")
    
    def _estimate_hypervolume_delta(self) -> float:
        """Estimate hypervolume change with a simplified calculation."""
        if len(self.iteration_history) < 2:
            return 0.0
        
        # Use Pareto-front size as a proxy metric.
        if self.optimizer.pareto_fronts:
            return len(self.optimizer.pareto_fronts[0]) * 0.1
        return 0.0
    
    def _compute_hypervolume(self, pareto_front: List[str], reference_point: List[float] = None) -> float:
        """
        Compute the Pareto-front hypervolume.

        Hypervolume is a standard multi-objective metric that measures the
        dominated-space volume between the Pareto front and a reference point.
        
        Args:
            pareto_front: Solution IDs in the Pareto front.
            reference_point: Reference point containing each objective's worst value.
            
        Returns:
            Hypervolume value, normalized to the [0, 1] range.
        """
        if not pareto_front:
            return 0.0
        
        # Collect objective values for Pareto-front solutions.
        pareto_solutions = []
        for sid in pareto_front:
            if sid in self.optimizer.solutions:
                sol = self.optimizer.solutions[sid]
                obj_values = []
                for obj_name, obj in self.optimizer.objectives.items():
                    if obj_name in sol.objectives:
                        # ParetoOptimizer normalizes maximize, minimize, and
                        # target_match objectives so larger values are always better.
                        obj_values.append(sol.objectives[obj_name])
                if len(obj_values) == len(self.optimizer.objectives):
                    pareto_solutions.append(obj_values)
        
        if not pareto_solutions:
            return 0.0
        
        n_objectives = len(self.optimizer.objectives)
        
        # Determine a reference point when one was not supplied.
        if reference_point is None:
            # Use each objective's worst value across all solutions.
            reference_point = []
            for i in range(n_objectives):
                all_values = [sol[i] for sol in pareto_solutions]
                # The reference point must be worse than all solutions.
                min_val = min(all_values)
                reference_point.append(min_val - abs(min_val) * 0.1 - 1e-6)
        
        # Compute hypervolume.
        if n_objectives == 2:
            # Use an exact calculation in two dimensions.
            hv = self._compute_2d_hypervolume(pareto_solutions, reference_point)
        else:
            # Use a Monte Carlo estimate in higher dimensions.
            hv = self._compute_monte_carlo_hypervolume(pareto_solutions, reference_point)
        
        return hv
    
    def _compute_2d_hypervolume(self, pareto_solutions: List[List[float]], 
                                 reference_point: List[float]) -> float:
        """
        Compute two-dimensional hypervolume with an exact algorithm.

        For two objectives, sort by the first objective and accumulate rectangle
        areas one at a time.
        """
        if not pareto_solutions:
            return 0.0
        
        # Sort descending by the first objective.
        sorted_solutions = sorted(pareto_solutions, key=lambda x: x[0], reverse=True)
        
        hv = 0.0
        prev_y = reference_point[1]
        
        for sol in sorted_solutions:
            x, y = sol[0], sol[1]
            if y > prev_y:
                # Compute the rectangle area.
                width = x - reference_point[0]
                height = y - prev_y
                hv += width * height
                prev_y = y
        
        return hv
    
    def _compute_monte_carlo_hypervolume(self, pareto_solutions: List[List[float]],
                                          reference_point: List[float],
                                          n_samples: int = 10000) -> float:
        """
        Estimate higher-dimensional hypervolume with Monte Carlo sampling.

        Sample points between the reference and ideal points and count the
        fraction dominated by the Pareto front.
        """
        import random
        
        n_objectives = len(reference_point)
        
        # Determine the ideal point from objective-wise best values.
        ideal_point = []
        for i in range(n_objectives):
            all_values = [sol[i] for sol in pareto_solutions]
            ideal_point.append(max(all_values))
        
        # Monte Carlo sampling.
        dominated_count = 0
        
        for _ in range(n_samples):
            # Randomly sample a point between the reference and ideal points.
            sample = [
                random.uniform(reference_point[i], ideal_point[i])
                for i in range(n_objectives)
            ]
            
            # Check whether a Pareto-front solution dominates this point.
            for sol in pareto_solutions:
                if all(sol[i] >= sample[i] for i in range(n_objectives)):
                    dominated_count += 1
                    break
        
        # Compute hypervolume.
        box_volume = 1.0
        for i in range(n_objectives):
            box_volume *= (ideal_point[i] - reference_point[i])
        
        hv = box_volume * (dominated_count / n_samples)
        
        return hv
    
    def compute_hypervolume_delta(self) -> float:
        """
        Compute hypervolume change for convergence detection.

        Compare the current Pareto front with the previous iteration. The
        optimization is considered converged if hypervolume does not improve
        significantly for k consecutive iterations.
        """
        if not self.optimizer.pareto_fronts:
            return 0.0
        
        current_hv = self._compute_hypervolume(self.optimizer.pareto_fronts[0])
        
        # Retrieve the previous hypervolume from history.
        if hasattr(self, '_last_hypervolume'):
            delta = current_hv - self._last_hypervolume
        else:
            delta = current_hv
        
        self._last_hypervolume = current_hv
        
        return delta
    
    def record_iteration_update(self, source: str, count: int, total_solutions: int):
        """
        [GPAC feature 4 - IFM] Record an incremental update.

        Record iteration data after each simulation-tool result to:
        1. Track Pareto-front evolution.
        2. Compute hypervolume changes for convergence detection.
        3. Adjust objective weights for the next iteration (AWA).
        
        Args:
            source: Update source, such as heat_analysis or structure_generation.
            count: Number of records in this update.
            total_solutions: Current total number of solutions.
        """
        # Determine whether the Pareto front improved.
        pareto_improved = False
        hv_delta = 0.0
        
        # Compute per-objective improvements precisely.
        objective_improvements = self._compute_objective_improvements()
        
        if self.optimizer.pareto_fronts:
            # solutions_added is an integer, not a list.
            old_pareto_size = self.iteration_history[-1].solutions_added if self.iteration_history else 0
            new_pareto_size = len(self.optimizer.pareto_fronts[0])
            pareto_improved = new_pareto_size > old_pareto_size
            
            # When front size is unchanged, check for objective-value improvement.
            if not pareto_improved and objective_improvements:
                pareto_improved = any(
                    info.get("improved", False) 
                    for info in objective_improvements.values()
                )
            
            # Simplified hypervolume-change estimate.
            hv_delta = (new_pareto_size - old_pareto_size) / max(total_solutions, 1)
        
        record = IterationRecord(
            iteration_id=self.current_iteration,
            timestamp=datetime.now().isoformat(),
            source=source,
            solutions_added=count,
            pareto_improvement=pareto_improved,
            hypervolume_delta=hv_delta,
            guidance_used={},
            objective_improvements=objective_improvements
        )
        self.iteration_history.append(record)
        
        # Trigger AWA only for sources with simulation results. structure_generation
        # records mechanical properties (E, G, nu), not complete simulation data;
        # adapting weights from it would make decisions from incomplete evidence.
        simulation_sources = {"heat_analysis", "electrical_analysis", "stiffness_analysis", "plasticity_simulation"}
        if source in simulation_sources:
            self._update_adaptive_weights(source)
    
    def _compute_objective_improvements(self) -> Dict[str, Dict[str, Any]]:
        """
        [AWA enhancement] Compute per-objective improvements precisely.

        Compare each objective's best Pareto-front value to determine which
        objectives improved and which stagnated.
        
        Returns:
            {
                "thermal_conductivity": {
                    "improved": True,
                    "delta": 0.5,  # Improvement magnitude.
                    "best_before": 23.0,
                    "best_after": 23.5,
                    "improvement_rate": 0.0217  # 2.17%
                },
                ...
            }
        """
        improvements = {}
        
        if not self.optimizer.objectives:
            return improvements
        
        # Retrieve best values from the previous iteration record, when present.
        previous_best = {}
        if self.iteration_history:
            last_record = self.iteration_history[-1]
            if last_record.objective_improvements:
                for obj_name, info in last_record.objective_improvements.items():
                    if "best_after" in info:
                        previous_best[obj_name] = info["best_after"]
        
        # Compute each objective's best current Pareto-front value.
        current_best = self._get_current_best_values()
        
        # Compare improvements.
        for obj_name, obj in self.optimizer.objectives.items():
            info = {
                "improved": False,
                "delta": 0.0,
                "best_before": previous_best.get(obj_name),
                "best_after": current_best.get(obj_name),
                "improvement_rate": 0.0
            }
            
            current_val = current_best.get(obj_name)
            prev_val = previous_best.get(obj_name)
            
            # Ensure current_val exists and is not None.
            if current_val is not None:
                # Ensure prev_val exists and is not None.
                if prev_val is not None:
                    # Determine improvement from the objective type.
                    if obj.obj_type == ObjectiveType.TARGET_MATCH and obj.target_value is not None:
                        prev_error = abs(prev_val - obj.target_value)
                        current_error = abs(current_val - obj.target_value)
                        delta = prev_error - current_error
                        info["improved"] = delta > 1e-6
                    elif obj.obj_type == ObjectiveType.MAXIMIZE:
                        # Maximization: current value greater than prior value improves.
                        delta = current_val - prev_val
                        info["improved"] = delta > 1e-6  # Avoid floating-point noise with a small threshold.
                    else:
                        # Minimization: current value less than prior value improves.
                        delta = prev_val - current_val
                        info["improved"] = delta > 1e-6
                    
                    info["delta"] = delta
                    
                    # Compute improvement rate relative to the prior value.
                    if abs(prev_val) > 1e-10:
                        info["improvement_rate"] = delta / abs(prev_val)
                else:
                    # Treat the first record (prev_val is None) as an improvement.
                    info["improved"] = True
                    info["delta"] = current_val if current_val is not None else 0.0
            
            improvements[obj_name] = info
        
        return improvements
    
    def _get_current_best_values(self) -> Dict[str, float]:
        """
        Return each objective's best value from the current Pareto front.
        
        Returns:
            {obj_name: best_value}
        """
        best_values = {}
        
        if not self.optimizer.pareto_fronts or not self.optimizer.pareto_fronts[0]:
            # Without a Pareto front, search all solutions.
            solutions = list(self.optimizer.solutions.values())
        else:
            # Search the Pareto front.
            solutions = [
                self.optimizer.solutions[sid] 
                for sid in self.optimizer.pareto_fronts[0]
                if sid in self.optimizer.solutions
            ]
        
        if not solutions:
            return best_values
        
        for obj_name, obj in self.optimizer.objectives.items():
            values = []
            for sol in solutions:
                if obj_name in sol.properties:
                    val = sol.properties[obj_name]
                    # Ignore None values.
                    if val is not None:
                        values.append(val)
            
            if values:
                if obj.obj_type == ObjectiveType.TARGET_MATCH and obj.target_value is not None:
                    best_values[obj_name] = min(values, key=lambda value: abs(value - obj.target_value))
                elif obj.obj_type == ObjectiveType.MAXIMIZE:
                    best_values[obj_name] = max(values)
                else:
                    best_values[obj_name] = min(values)
        
        return best_values

    def _update_adaptive_weights(self, successful_source: str):
        """
        [GPAC feature 1 - AWA] Dynamically adjust objective weights.

        Core mechanism:
        1. Precisely track each objective's improvement over the latest k iterations.
        2. Increase the weight of objectives that repeatedly fail to improve.
        3. Apply the paper's multiplicative update and clip weights to [0.1, 2.0].
        4. Emit detailed improvement-analysis logs.

        In ablation mode, CHATMS_ABLATION_ADAPTIVE_WEIGHT=1 keeps every
        objective weight fixed at 1.0 and disables dynamic adjustment.
        """
        # === Ablation check ===
        if not is_adaptive_weight_enabled():
            # Ablation mode keeps weights fixed to isolate AWA's contribution.
            return
        
        # Record the current weights before adjustment.
        current_weights = {}
        for obj_name, obj in self.optimizer.objectives.items():
            current_weights[obj_name] = obj.weight
        self._weight_history.append(current_weights.copy())
        
        # Analyze each objective's improvement over recent iterations.
        k = self.stagnation_window
        if len(self.iteration_history) < 2:
            # At least two iterations are required for comparison.
            return
        
        recent_records = self.iteration_history[-min(k, len(self.iteration_history)):]
        
        # Count improvements per objective precisely.
        objective_improvement_count = {obj_name: 0 for obj_name in self.optimizer.objectives}
        objective_improvement_details = {obj_name: [] for obj_name in self.optimizer.objectives}
        
        for record in recent_records:
            if record.objective_improvements:
                for obj_name, info in record.objective_improvements.items():
                    if obj_name in objective_improvement_count:
                        if info.get("improved", False):
                            objective_improvement_count[obj_name] += 1
                            objective_improvement_details[obj_name].append({
                                "iteration": record.iteration_id,
                                "delta": info.get("delta", 0),
                                "rate": info.get("improvement_rate", 0)
                            })
        
        # Compute weight adjustments. Objectives without improvement receive
        # higher weights to strengthen exploration in that direction.
        weight_adjustments = {}
        adjustment_reasons = {}
        window_size = len(recent_records)
        
        for obj_name, improvement_count in objective_improvement_count.items():
            improvement_ratio = improvement_count / window_size if window_size > 0 else 0
            
            if improvement_count == 0:
                # Three consecutive iterations without improvement: w_i <- w_i * 1.25.
                adjustment = 0.25
                reason = f"stagnated ({improvement_count}/{window_size} improvements)"
            elif improvement_ratio > 0.6:
                # Rapidly converging objective: w_i <- w_i * 0.90.
                adjustment = -0.1
                reason = f"rapid convergence ({improvement_count}/{window_size} improvements)"
            else:
                adjustment = 0.0
                reason = f"weight unchanged ({improvement_count}/{window_size} improvements)"
            
            weight_adjustments[obj_name] = adjustment
            adjustment_reasons[obj_name] = reason
        
        # Apply weight changes without normalization; clip to the paper's bounds.
        for obj_name, obj in self.optimizer.objectives.items():
            old_weight = obj.weight
            adjustment = weight_adjustments.get(obj_name, 0)
            
            # Apply the change and constrain the result to [0.1, 2.0].
            new_weight = max(0.1, min(2.0, old_weight * (1 + adjustment)))
            obj.weight = new_weight
        
        for obj in self.optimizer.objectives.values():
            obj.weight = round(obj.weight, 4)
        
        # Record the changed weights.
        new_weights = {obj_name: round(obj.weight, 4) for obj_name, obj in self.optimizer.objectives.items()}
        current_weights_rounded = {k: round(v, 4) for k, v in current_weights.items()}
        
        # Emit detailed AWA analysis logs.
        if current_weights_rounded != new_weights:
            print(f"\n[SAES Guidance-AWA] Adaptive weight adjustment (observation window: {window_size} iterations)")
            print("-" * 50)
            for obj_name in self.optimizer.objectives:
                old_w = current_weights_rounded.get(obj_name, 1.0)
                new_w = new_weights.get(obj_name, 1.0)
                reason = adjustment_reasons.get(obj_name, "")
                change = "up" if new_w > old_w else ("down" if new_w < old_w else "unchanged")
                print(f"   {obj_name}: {old_w:.3f} {change} {new_w:.3f} | {reason}")
            print("-" * 50)
    
    def check_convergence(self) -> Dict[str, Any]:
        """
        [GPAC - convergence detection]

        Determine whether optimization has converged and may terminate early.
        
        Returns:
            {
                "converged": bool,
                "reason": str,
                "confidence": float  # 0-1
            }
        """
        result = {
            "converged": False,
            "reason": "",
            "confidence": 0.0
        }
        
        # Condition 1: maximum iteration count reached.
        if self.current_iteration >= self.max_iterations:
            result["converged"] = True
            result["reason"] = "maximum iteration count reached"
            result["confidence"] = 1.0
            return result
        
        # Minimum-iteration guard.
        if self.current_iteration < self.min_iterations_before_convergence:
            result["reason"] = f"minimum-iteration guard active ({self.current_iteration}/{self.min_iterations_before_convergence})"
            return result
        
        # Condition 2: the paper's three-iteration stagnation window.
        if len(self.iteration_history) >= self.stagnation_window:
            recent = self.iteration_history[-self.stagnation_window:]
            if all(not r.pareto_improvement for r in recent):
                result["converged"] = True
                result["reason"] = f"no Pareto improvement for {self.stagnation_window} consecutive iterations"
                result["confidence"] = 0.8
                return result
        
        # Condition 3: all objectives are satisfied.
        if self._check_all_objectives_satisfied():
            result["converged"] = True
            result["reason"] = "all objectives are satisfied"
            result["confidence"] = 0.95
            return result
        
        # Condition 4: hypervolume change has flattened over the same window.
        if len(self.iteration_history) >= self.stagnation_window:
            recent_hv = [r.hypervolume_delta for r in self.iteration_history[-self.stagnation_window:]]
            avg_hv_change = sum(abs(h) for h in recent_hv) / self.stagnation_window
            if avg_hv_change < self.convergence_threshold:
                result["converged"] = True
                result["reason"] = f"hypervolume change ({avg_hv_change:.4f}) is below the threshold"
                result["confidence"] = 0.7
                return result
        
        return result
    
    def should_continue_iteration(self) -> Tuple[bool, str]:
        """Return whether the optimization should continue."""
        if self.current_iteration >= self.max_iterations:
            return False, "maximum iteration count reached"
        
        # Minimum-iteration guard: force continuation for the first N iterations.
        if self.current_iteration < self.min_iterations_before_convergence:
            return True, f"minimum-iteration guard active ({self.current_iteration}/{self.min_iterations_before_convergence})"
        
        if len(self.iteration_history) >= self.stagnation_window:
            recent = self.iteration_history[-self.stagnation_window:]
            if all(not r.pareto_improvement for r in recent):
                return False, f"no improvement for {self.stagnation_window} consecutive iterations; converged"
        
        # Check whether all objectives are within acceptable bounds.
        if self._check_all_objectives_satisfied():
            return False, "all objectives are satisfied"
        
        return True, "continue optimization"
    
    def _check_all_objectives_satisfied(self) -> bool:
        """Return whether all objectives are satisfied."""
        if not self.optimizer.pareto_fronts:
            return False
        
        # Without configured objectives, the run cannot be considered satisfied.
        if not self.optimizer.objectives:
            return False
        
        # Gather objectives that have target values.
        objectives_with_target = [
            obj for obj in self.optimizer.objectives.values() 
            if obj.target_value is not None
        ]
        
        # With only maximize/minimize objectives and no target value, automatic
        # termination is not appropriate; the user or other logic must decide.
        if not objectives_with_target:
            return False
        
        pareto_solutions = [self.optimizer.solutions[sid] for sid in self.optimizer.pareto_fronts[0]]
        
        for sol in pareto_solutions:
            all_satisfied = True
            for obj_name, obj in self.optimizer.objectives.items():
                if obj.target_value is not None and obj_name in sol.properties:
                    value = sol.properties[obj_name]
                    tolerance = obj.target_value * 0.1  # Ten-percent tolerance.
                    
                    if obj.obj_type == ObjectiveType.TARGET_MATCH and obj.target_value is not None:
                        if abs(value - obj.target_value) > tolerance:
                            all_satisfied = False
                            break
                    elif obj.obj_type == ObjectiveType.MAXIMIZE:
                        if value < obj.target_value - tolerance:
                            all_satisfied = False
                            break
                    else:
                        if value > obj.target_value + tolerance:
                            all_satisfied = False
                            break
                            
            if all_satisfied:
                return True
        
        return False
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate the final Pareto optimization report."""
        self.optimizer.compute_pareto_fronts()
        pareto_solutions = self.optimizer.get_pareto_optimal_solutions(10)
        
        report = {
            "summary": {
                "total_iterations": self.current_iteration,
                "total_solutions_evaluated": len(self.optimizer.solutions),
                "pareto_optimal_count": len(pareto_solutions),
                "objectives": [
                    {
                        "name": obj.name,
                        "display_name": obj.display_name,
                        "type": obj.obj_type.value,
                        "target": obj.target_value
                    }
                    for obj in self.optimizer.objectives.values()
                ]
            },
            "pareto_optimal_solutions": [
                {
                    "rank": i + 1,
                    "id": sol.id,
                    "source": sol.source,
                    "properties": sol.properties,
                    "crowding_distance": sol.crowding_distance
                }
                for i, sol in enumerate(pareto_solutions)
            ],
            "iteration_history": [
                {
                    "iteration": r.iteration_id,
                    "source": r.source,
                    "solutions_added": r.solutions_added,
                    "improved": r.pareto_improvement
                }
                for r in self.iteration_history
            ],
            "recommendation": self._generate_final_recommendation(pareto_solutions)
        }
        
        return report
    
    def _generate_final_recommendation(self, solutions: List[Solution]) -> str:
        """Generate the final recommendation."""
        if not solutions:
            return "No Pareto-optimal solution satisfies the constraints; consider relaxing the target constraints."
        
        lines = ["## Pareto-Optimal Solution Recommendations\n"]
        
        # Find the best solution for each objective.
        for obj_name, obj in self.optimizer.objectives.items():
            best_sol = None
            best_val = None
            
            for sol in solutions:
                if obj_name in sol.properties:
                    val = sol.properties[obj_name]
                    if best_val is None:
                        best_val = val
                        best_sol = sol
                    elif obj.obj_type == ObjectiveType.TARGET_MATCH and obj.target_value is not None:
                        if abs(val - obj.target_value) < abs(best_val - obj.target_value):
                            best_val = val
                            best_sol = sol
                    elif obj.obj_type == ObjectiveType.MAXIMIZE and val > best_val:
                        best_val = val
                        best_sol = sol
                    elif obj.obj_type == ObjectiveType.MINIMIZE and val < best_val:
                        best_val = val
                        best_sol = sol
            
            if best_sol:
                lines.append(f"- **Best {obj.display_name}**: {best_sol.id} ({best_val:.4g} {obj.unit})")
        
        # Find a balanced solution with the largest crowding distance.
        balanced = max(solutions, key=lambda s: s.crowding_distance if s.crowding_distance < float('inf') else 0)
        lines.append(f"\n- **Balanced recommendation**: {balanced.id}")
        
        return "\n".join(lines)
    
    def get_context_injection(self) -> str:
        """
        Generate Pareto information to inject into agent context.

        This is SAES Guidance's primary interface to the agent system.
        """
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("[SAES Guidance Status]")
        lines.append("=" * 70)
        
        lines.append(f"Current iteration: {self.current_iteration}/{self.max_iterations}")
        lines.append(f"Evaluated solutions: {len(self.optimizer.solutions)}")
        
        if self.optimizer.objectives:
            lines.append("\nOptimization objectives:")
            for obj in self.optimizer.objectives.values():
                target_str = f", target={obj.target_value}" if obj.target_value else ""
                lines.append(f"  - {obj.display_name} ({obj.obj_type.value}){target_str}")
        
        # Pareto-front status.
        if self.optimizer.pareto_fronts:
            pareto_size = len(self.optimizer.pareto_fronts[0])
            lines.append(f"\nPareto-optimal solutions: {pareto_size}")
            
            if pareto_size > 0:
                pareto_solutions = self.optimizer.get_pareto_optimal_solutions(3)
                lines.append("Top 3 Pareto solutions:")
                for i, sol in enumerate(pareto_solutions, 1):
                    props = ", ".join([f"{k}={v:.3g}" for k, v in list(sol.properties.items())[:4]])
                    lines.append(f"  {i}. [{sol.source}] {sol.id}: {props}")
        
        # Iteration recommendation.
        should_continue, reason = self.should_continue_iteration()
        lines.append(f"\nIteration status: {'continue' if should_continue else 'stop'} - {reason}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# Global guidance instance.
_global_saes_guidance: Optional[SAESGuidance] = None


def is_saes_guidance_enabled() -> bool:
    """Return whether SAES Guidance is enabled (ablation support)."""
    import os
    return os.environ.get("CHATMS_ABLATION_SAES_GUIDANCE", "0") != "1"


def is_adaptive_weight_enabled() -> bool:
    """
    Return whether adaptive weighting is enabled (ablation support).

    When CHATMS_ABLATION_ADAPTIVE_WEIGHT=1, keep all objective weights fixed
    at 1.0 and disable dynamic adjustment to isolate AWA's contribution.
    
    Returns:
        True: Use dynamic adaptive weights (AWA).
        False: Keep fixed weights (ablation mode).
    """
    import os
    return os.environ.get("CHATMS_ABLATION_ADAPTIVE_WEIGHT", "0") != "1"


def get_saes_guidance() -> SAESGuidance:
    """Return the global SAES Guidance instance."""
    global _global_saes_guidance
    if _global_saes_guidance is None:
        _global_saes_guidance = SAESGuidance()
    return _global_saes_guidance


def reset_saes_guidance() -> None:
    """Reset SAES Guidance for a new session."""
    global _global_saes_guidance
    from tools.pareto_optimizer import reset_pareto_optimizer
    reset_pareto_optimizer()
    _global_saes_guidance = SAESGuidance()


def is_multi_objective_scenario(parsed_requirement: Dict[str, Any] = None) -> bool:
    """
    Quickly determine whether this is a multi-objective scenario.
    
    Args:
        parsed_requirement: Optional parsed-requirement dictionary. When absent,
            inspect objectives registered with SAES Guidance.
    """
    # Ablation check.
    if not is_saes_guidance_enabled():
        return False
    
    if parsed_requirement is not None:
        target_properties = parsed_requirement.get("target_properties", {})
        return len(target_properties) >= 2
    else:
        # Check whether global SAES Guidance has registered objectives.
        global _global_saes_guidance
        if _global_saes_guidance is not None and _global_saes_guidance.optimizer.objectives:
            return len(_global_saes_guidance.optimizer.objectives) >= 2
        return False
