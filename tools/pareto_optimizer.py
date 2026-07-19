"""
Gradient-Free Pareto-Guided Agent Coordination (GPAC)
Multi-objective optimization guidance module: a Pareto optimizer that works
with the agent system.

Key features:
1. Dynamic objective-weight adaptation based on the current Pareto-front distribution.
2. Dominance-based candidate guidance for microstructure generation.
3. Crowding-distance-driven database retrieval filters.
4. Incremental Pareto-front maintenance for real-time updates and visualization.

Author: GPAC Module for LightAgent Multi-Agent System
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime


class ObjectiveType(Enum):
    """Objective direction: maximize, minimize, or match a target."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET_MATCH = "target_match"


@dataclass
class Objective:
    """Definition of one optimization objective."""
    name: str                          # Objective key, such as "thermal_conductivity".
    display_name: str                  # Display name, such as "Thermal conductivity".
    unit: str                          # Unit, such as "W/(m K)".
    obj_type: ObjectiveType            # Whether to maximize, minimize, or match.
    target_value: Optional[float] = None   # User-specified target value, when available.
    target_range: Optional[Tuple[float, float]] = None  # Target interval.
    weight: float = 1.0                # Initial weight.
    priority: int = 1                  # Priority, where 1 is highest.
    
    def normalize(self, value: float, global_min: float, global_max: float) -> float:
        """Normalize a value to [0, 1] for this objective type."""
        if self.obj_type == ObjectiveType.TARGET_MATCH and self.target_value is not None:
            error = abs(value - self.target_value) / (abs(self.target_value) + 1e-10)
            return 1.0 / (1.0 + error)
        if global_max == global_min:
            return 0.5
        normalized = (value - global_min) / (global_max - global_min)
        if self.obj_type == ObjectiveType.MINIMIZE:
            normalized = 1.0 - normalized
        return np.clip(normalized, 0.0, 1.0)


@dataclass 
class Solution:
    """One candidate solution (microstructure)."""
    id: str                            # Microstructure identifier.
    source: str                        # Source: "database" or "ai_generated".
    properties: Dict[str, float]       # Performance-property mapping.
    objectives: Dict[str, float] = field(default_factory=dict)  # Normalized objective values.
    crowding_distance: float = 0.0     # Crowding distance.
    rank: int = 0                      # Pareto rank (0 is the optimal front).
    dominated_count: int = 0           # Number of solutions that dominate this one.
    dominates: List[str] = field(default_factory=list)  # IDs of dominated solutions.


class ParetoOptimizer:
    """
    Core Pareto optimizer.

    Responsibilities:
    1. Manage multi-objective definitions.
    2. Maintain Pareto fronts.
    3. Compute crowding distance.
    4. Generate agent guidance strategies.
    """
    
    def __init__(self):
        self.objectives: Dict[str, Objective] = {}
        self.solutions: Dict[str, Solution] = {}
        self.pareto_fronts: List[List[str]] = []  # Solution IDs ordered by rank.
        self.iteration_history: List[Dict[str, Any]] = []
        self._global_bounds: Dict[str, Tuple[float, float]] = {}  # Global bounds by objective.
        
    def register_objective(self, objective: Objective) -> None:
        """Register one optimization objective."""
        self.objectives[objective.name] = objective
        print(f"[ParetoOptimizer] Registered objective: {objective.display_name} ({objective.obj_type.value})")
    
    def register_objectives_from_requirement(self, parsed_requirement: Dict[str, Any]) -> None:
        """
        Register objectives automatically from a parsed requirement.

        This is the interface used by RequirementParser.
        """
        target_properties = parsed_requirement.get("target_properties", {})
        
        # Common property-to-objective mappings.
        property_mapping = {
            "thermal_conductivity": ("Thermal conductivity", "W/(m K)", ObjectiveType.MAXIMIZE),
            "electrical_conductivity": ("Electrical conductivity", "S/m", ObjectiveType.MAXIMIZE),
            "youngs_modulus": ("Young's modulus", "MPa", ObjectiveType.MAXIMIZE),
            "E": ("Young's modulus", "MPa", ObjectiveType.MAXIMIZE),
            "shear_modulus": ("Shear modulus", "MPa", ObjectiveType.MAXIMIZE),
            "G": ("Shear modulus", "MPa", ObjectiveType.MAXIMIZE),
            "poisson_ratio": ("Poisson's ratio", "", ObjectiveType.MINIMIZE),  # Smaller is usually preferred.
            "nu": ("Poisson's ratio", "", ObjectiveType.MINIMIZE),
            "volume_fraction": ("Volume fraction", "", ObjectiveType.MINIMIZE),  # Supports lightweight designs.
            "vof": ("Volume fraction", "", ObjectiveType.MINIMIZE),
            "density": ("Density", "g/cm^3", ObjectiveType.MINIMIZE),
        }
        
        priority = 1
        for prop_name, prop_spec in target_properties.items():
            if prop_name in property_mapping:
                display_name, unit, default_type = property_mapping[prop_name]
                
                # Parse a target value or range.
                target_value = None
                target_range = None
                obj_type = default_type
                
                if isinstance(prop_spec, dict):
                    if "min" in prop_spec and "max" in prop_spec:
                        target_range = (prop_spec["min"], prop_spec["max"])
                        target_value = prop_spec.get("target", sum(target_range) / 2)
                        obj_type = ObjectiveType.TARGET_MATCH
                    elif "value" in prop_spec:
                        target_value = prop_spec["value"]
                        obj_type = ObjectiveType.TARGET_MATCH
                    elif "min" in prop_spec:
                        target_value = prop_spec["min"]
                        obj_type = ObjectiveType.MAXIMIZE
                    elif "max" in prop_spec:
                        target_value = prop_spec["max"]
                        obj_type = ObjectiveType.MINIMIZE
                elif isinstance(prop_spec, (int, float)):
                    target_value = float(prop_spec)
                    obj_type = ObjectiveType.TARGET_MATCH
                
                self.register_objective(Objective(
                    name=prop_name,
                    display_name=display_name,
                    unit=unit,
                    obj_type=obj_type,
                    target_value=target_value,
                    target_range=target_range,
                    priority=priority
                ))
                priority += 1
    
    def add_solution(self, solution: Solution) -> None:
        """Add one candidate solution."""
        self.solutions[solution.id] = solution
        self._update_global_bounds(solution)
    
    def add_solutions_from_property_table(self, property_table: List[Dict[str, Any]], source: str = "unknown") -> int:
        """
        Add candidate solutions from MicrostructurePropertyTable.

        This is the interface used by the simulation agent.
        """
        count = 0
        for item in property_table:
            filename = item.get("filename", "")
            if not filename:
                continue
            
            properties = {}
            # Map property names.
            mapping = {
                "thermal_conductivity": "thermal_conductivity",
                "electrical_conductivity": "electrical_conductivity", 
                "youngs_modulus": "E",
                "shear_modulus": "G",
                "poisson_ratio": "nu",
                "thermal_volume_fraction": "volume_fraction",
                "stiffness_volume_fraction": "volume_fraction",
            }
            
            for table_key, prop_key in mapping.items():
                if table_key in item and item[table_key] is not None:
                    properties[prop_key] = float(item[table_key])
            
            if properties:
                # Determine the source.
                if "ai_generated" in filename.lower():
                    actual_source = "ai_generated"
                else:
                    actual_source = source if source != "unknown" else "database"
                    
                self.add_solution(Solution(
                    id=filename,
                    source=actual_source,
                    properties=properties
                ))
                count += 1
        
        return count
    
    def _update_global_bounds(self, solution: Solution) -> None:
        """Update global objective bounds."""
        for obj_name in self.objectives:
            if obj_name in solution.properties:
                value = solution.properties[obj_name]
                if obj_name not in self._global_bounds:
                    self._global_bounds[obj_name] = (value, value)
                else:
                    min_val, max_val = self._global_bounds[obj_name]
                    self._global_bounds[obj_name] = (min(min_val, value), max(max_val, value))
    
    def _normalize_solutions(self) -> None:
        """Normalize objective values for all solutions."""
        for sol in self.solutions.values():
            sol.objectives = {}
            for obj_name, obj in self.objectives.items():
                if obj_name in sol.properties:
                    if obj_name in self._global_bounds:
                        min_val, max_val = self._global_bounds[obj_name]
                        sol.objectives[obj_name] = obj.normalize(
                            sol.properties[obj_name], min_val, max_val
                        )
    
    def _dominates(self, sol_a: Solution, sol_b: Solution) -> bool:
        """Return whether sol_a dominates sol_b across all objectives."""
        dominated_objectives = list(self.objectives.keys())
        
        at_least_one_better = False
        for obj_name in dominated_objectives:
            val_a = sol_a.objectives.get(obj_name, 0)
            val_b = sol_b.objectives.get(obj_name, 0)
            
            if val_a < val_b:  # Normalization makes larger values better.
                return False
            elif val_a > val_b:
                at_least_one_better = True
        
        return at_least_one_better
    
    def compute_pareto_fronts(self) -> List[List[str]]:
        """
        Compute Pareto-front levels using NSGA-II non-dominated sorting.

        Returns solution-ID lists ordered by rank.
        """
        self._normalize_solutions()
        
        # Initialize solution state.
        for sol in self.solutions.values():
            sol.dominated_count = 0
            sol.dominates = []
            sol.rank = -1
        
        # Compute dominance relationships.
        sol_list = list(self.solutions.values())
        for i, sol_a in enumerate(sol_list):
            for j, sol_b in enumerate(sol_list):
                if i != j:
                    if self._dominates(sol_a, sol_b):
                        sol_a.dominates.append(sol_b.id)
                        sol_b.dominated_count += 1
        
        # Assign fronts by rank.
        self.pareto_fronts = []
        current_front = [sol.id for sol in sol_list if sol.dominated_count == 0]
        rank = 0
        
        while current_front:
            for sol_id in current_front:
                self.solutions[sol_id].rank = rank
            self.pareto_fronts.append(current_front)
            
            next_front = []
            for sol_id in current_front:
                for dominated_id in self.solutions[sol_id].dominates:
                    self.solutions[dominated_id].dominated_count -= 1
                    if self.solutions[dominated_id].dominated_count == 0:
                        next_front.append(dominated_id)
            
            current_front = next_front
            rank += 1
        
        # Compute crowding distance.
        self._compute_crowding_distance()
        
        return self.pareto_fronts
    
    def _compute_crowding_distance(self) -> None:
        """Compute crowding distances to preserve diversity."""
        for front in self.pareto_fronts:
            if len(front) <= 2:
                for sol_id in front:
                    self.solutions[sol_id].crowding_distance = float('inf')
                continue
            
            # Initialize distances.
            for sol_id in front:
                self.solutions[sol_id].crowding_distance = 0
            
            # Compute distance contributions for every objective.
            for obj_name in self.objectives:
                # Sort by this objective.
                sorted_front = sorted(front, key=lambda x: self.solutions[x].objectives.get(obj_name, 0))
                
                # Boundary points receive infinite distance.
                self.solutions[sorted_front[0]].crowding_distance = float('inf')
                self.solutions[sorted_front[-1]].crowding_distance = float('inf')
                
                # Compute distances for interior points.
                obj_range = (
                    self.solutions[sorted_front[-1]].objectives.get(obj_name, 0) -
                    self.solutions[sorted_front[0]].objectives.get(obj_name, 0)
                )
                
                if obj_range > 0:
                    for i in range(1, len(sorted_front) - 1):
                        prev_val = self.solutions[sorted_front[i-1]].objectives.get(obj_name, 0)
                        next_val = self.solutions[sorted_front[i+1]].objectives.get(obj_name, 0)
                        self.solutions[sorted_front[i]].crowding_distance += (next_val - prev_val) / obj_range
    
    def get_pareto_optimal_solutions(self, max_count: int = 10) -> List[Solution]:
        """Return Pareto-optimal solutions from the first front."""
        if not self.pareto_fronts:
            self.compute_pareto_fronts()
        
        if not self.pareto_fronts:
            return []
        
        front_0 = self.pareto_fronts[0]
        # Prefer solutions from sparsely covered regions.
        sorted_front = sorted(front_0, key=lambda x: self.solutions[x].crowding_distance, reverse=True)
        
        return [self.solutions[sol_id] for sol_id in sorted_front[:max_count]]
    
    def generate_guidance_for_structure_generator(self) -> Dict[str, Any]:
        """
        Generate parameter guidance for StructureGenerator.

        Sparse Pareto-front regions guide exploration of under-covered objective space.
        """
        if not self.pareto_fronts or not self.pareto_fronts[0]:
            # No existing solution: return initial guidance.
            return self._generate_initial_guidance()
        
        pareto_solutions = [self.solutions[sid] for sid in self.pareto_fronts[0]]
        
        # 1. Analyze the coverage of the current Pareto front.
        coverage_analysis = self._analyze_pareto_coverage(pareto_solutions)
        
        # 2. Identify sparse regions near high-crowding-distance solutions.
        sparse_regions = self._identify_sparse_regions(pareto_solutions)
        
        # 3. Compute suggested target parameters.
        suggested_params = self._compute_suggested_params(sparse_regions, coverage_analysis)
        
        # 4. Select an exploration direction from the objective type.
        exploration_directions = {}
        for obj_name, obj in self.objectives.items():
            if obj_name in coverage_analysis:
                if obj.obj_type == ObjectiveType.TARGET_MATCH and obj.target_value is not None:
                    exploration_directions[obj_name] = {
                        "direction": "target",
                        "current_best": min(
                            pareto_solutions,
                            key=lambda s: abs(s.properties.get(obj_name, obj.target_value) - obj.target_value)
                        ).properties.get(obj_name, obj.target_value),
                        "suggested_target": obj.target_value
                    }
                elif obj.obj_type == ObjectiveType.MAXIMIZE:
                    # Maximization: try to exceed the current maximum.
                    exploration_directions[obj_name] = {
                        "direction": "increase",
                        "current_best": coverage_analysis[obj_name]["max"],
                        "suggested_target": coverage_analysis[obj_name]["max"] * 1.1
                    }
                else:
                    # Minimization: try to improve below the current minimum.
                    exploration_directions[obj_name] = {
                        "direction": "decrease", 
                        "current_best": coverage_analysis[obj_name]["min"],
                        "suggested_target": coverage_analysis[obj_name]["min"] * 0.9
                    }
        
        return {
            "type": "structure_generator_guidance",
            "timestamp": datetime.now().isoformat(),
            "current_pareto_size": len(pareto_solutions),
            "coverage_analysis": coverage_analysis,
            "sparse_regions": sparse_regions,
            "suggested_params": suggested_params,
            "exploration_directions": exploration_directions,
            "priority_objectives": self._get_priority_objectives(),
            "recommendation": self._generate_recommendation_text(suggested_params, exploration_directions)
        }
    
    def generate_guidance_for_retrieval(self) -> Dict[str, Any]:
        """
        Generate database-retrieval guidance for Generator's internal retrieval branch.

        Pareto analysis produces focused SQL or filtering conditions.
        """
        if not self.pareto_fronts:
            return self._generate_initial_db_guidance()
        
        pareto_solutions = [self.solutions[sid] for sid in self.pareto_fronts[0]]
        
        # Analyze regions that lack database solutions.
        db_solutions = [s for s in pareto_solutions if s.source == "database"]
        ai_solutions = [s for s in pareto_solutions if s.source == "ai_generated"]
        
        # Find regions covered by AI-generated solutions but absent from the database.
        target_regions = self._find_underrepresented_db_regions(db_solutions, ai_solutions)
        
        # Generate filter conditions.
        filter_conditions = self._generate_filter_conditions(target_regions)
        
        return {
            "type": "retrieval_guidance",
            "timestamp": datetime.now().isoformat(),
            "db_solutions_in_pareto": len(db_solutions),
            "ai_solutions_in_pareto": len(ai_solutions),
            "target_regions": target_regions,
            "filter_conditions": filter_conditions,
            "volume_fraction_range": self._suggest_vof_range(),
            "recommendation": self._generate_db_recommendation_text(filter_conditions)
        }
    
    def _generate_initial_guidance(self) -> Dict[str, Any]:
        """Generate initial guidance when no historical data is available."""
        suggested_params = {}
        for obj_name, obj in self.objectives.items():
            if obj.target_value is not None:
                suggested_params[obj_name] = obj.target_value
            elif obj.target_range is not None:
                # Start from the midpoint of the target interval.
                suggested_params[obj_name] = (obj.target_range[0] + obj.target_range[1]) / 2
        
        return {
            "type": "structure_generator_guidance",
            "timestamp": datetime.now().isoformat(),
            "current_pareto_size": 0,
            "suggested_params": suggested_params,
            "recommendation": "Initial exploration: generate microstructures from the user-specified target values."
        }
    
    def _generate_initial_db_guidance(self) -> Dict[str, Any]:
        """Generate initial database-retrieval guidance."""
        filter_conditions = {}
        for obj_name, obj in self.objectives.items():
            if obj.target_range is not None:
                filter_conditions[obj_name] = {
                    "min": obj.target_range[0],
                    "max": obj.target_range[1]
                }
            elif obj.target_value is not None:
                # Use a plus-or-minus 20 percent target interval.
                margin = obj.target_value * 0.2
                filter_conditions[obj_name] = {
                    "min": obj.target_value - margin,
                    "max": obj.target_value + margin
                }
        
        return {
            "type": "retrieval_guidance",
            "timestamp": datetime.now().isoformat(),
            "db_solutions_in_pareto": 0,
            "filter_conditions": filter_conditions,
            "recommendation": "Initial retrieval: use broad filters to collect a diverse candidate set."
        }
    
    def _analyze_pareto_coverage(self, solutions: List[Solution]) -> Dict[str, Dict[str, float]]:
        """Analyze objective-space coverage of the Pareto front."""
        coverage = {}
        for obj_name in self.objectives:
            values = [s.properties.get(obj_name) for s in solutions if obj_name in s.properties]
            if values:
                coverage[obj_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values)
                }
        return coverage
    
    def _identify_sparse_regions(self, solutions: List[Solution]) -> List[Dict[str, Any]]:
        """Identify sparse regions of the Pareto front."""
        if len(solutions) < 3:
            return []
        
        # Select high-crowding-distance solutions as sparse-region representatives.
        sparse_solutions = sorted(solutions, key=lambda s: s.crowding_distance, reverse=True)[:3]
        
        regions = []
        for sol in sparse_solutions:
            if sol.crowding_distance < float('inf'):
                region = {
                    "reference_solution": sol.id,
                    "crowding_distance": sol.crowding_distance,
                    "properties": sol.properties.copy()
                }
                regions.append(region)
        
        return regions
    
    def _compute_suggested_params(self, sparse_regions: List[Dict], coverage: Dict) -> Dict[str, float]:
        """Compute suggested objective parameters."""
        suggested = {}
        
        for obj_name, obj in self.objectives.items():
            if sparse_regions and obj_name in sparse_regions[0].get("properties", {}):
                # Base the suggestion on the sparse region.
                base_value = sparse_regions[0]["properties"][obj_name]
                if obj.obj_type == ObjectiveType.TARGET_MATCH and obj.target_value is not None:
                    suggested[obj_name] = obj.target_value
                elif obj.obj_type == ObjectiveType.MAXIMIZE:
                    suggested[obj_name] = base_value * 1.05  # Slightly increase the value.
                else:
                    suggested[obj_name] = base_value * 0.95  # Slightly reduce the value.
            elif obj_name in coverage:
                # Base the suggestion on the current range.
                if obj.obj_type == ObjectiveType.MAXIMIZE:
                    suggested[obj_name] = coverage[obj_name]["max"] * 1.1
                else:
                    suggested[obj_name] = coverage[obj_name]["min"] * 0.9
            elif obj.target_value is not None:
                suggested[obj_name] = obj.target_value
        
        return suggested
    
    def _get_priority_objectives(self) -> List[str]:
        """Return objective names ordered by priority."""
        return sorted(self.objectives.keys(), key=lambda x: self.objectives[x].priority)
    
    def _generate_recommendation_text(self, params: Dict, directions: Dict) -> str:
        """Generate a recommendation for StructureGenerator."""
        lines = ["Based on the current Pareto-front analysis, we recommend:"]
        
        for obj_name, direction_info in directions.items():
            obj = self.objectives[obj_name]
            if direction_info["direction"] == "target":
                lines.append(f"- {obj.display_name}: prioritize the target value {direction_info['suggested_target']:.2f} {obj.unit}")
            elif direction_info["direction"] == "increase":
                lines.append(f"- {obj.display_name}: try increasing to {direction_info['suggested_target']:.2f} {obj.unit}")
            else:
                lines.append(f"- {obj.display_name}: try decreasing to {direction_info['suggested_target']:.2f} {obj.unit}")
        
        return "\n".join(lines)
    
    def _find_underrepresented_db_regions(self, db_solutions: List[Solution], ai_solutions: List[Solution]) -> List[Dict]:
        """Find regions without database solutions."""
        regions = []
        
        for ai_sol in ai_solutions:
            # Check for a nearby database solution.
            has_nearby_db = False
            for db_sol in db_solutions:
                distance = self._compute_objective_distance(ai_sol, db_sol)
                if distance < 0.2:  # Distance threshold.
                    has_nearby_db = True
                    break
            
            if not has_nearby_db:
                regions.append({
                    "reference": ai_sol.id,
                    "properties": ai_sol.properties.copy()
                })
        
        return regions
    
    def _compute_objective_distance(self, sol_a: Solution, sol_b: Solution) -> float:
        """Compute distance between two solutions in normalized objective space."""
        distance = 0
        count = 0
        for obj_name in self.objectives:
            if obj_name in sol_a.objectives and obj_name in sol_b.objectives:
                distance += (sol_a.objectives[obj_name] - sol_b.objectives[obj_name]) ** 2
                count += 1
        return np.sqrt(distance / max(count, 1))
    
    def _generate_filter_conditions(self, target_regions: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Generate database filtering conditions."""
        if not target_regions:
            return {}
        
        conditions = {}
        for obj_name in self.objectives:
            values = [r["properties"].get(obj_name) for r in target_regions if obj_name in r.get("properties", {})]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values) if len(values) > 1 else mean_val * 0.1
                conditions[obj_name] = {
                    "min": mean_val - 2 * std_val,
                    "max": mean_val + 2 * std_val,
                    "target": mean_val
                }
        
        return conditions
    
    def _suggest_vof_range(self) -> Tuple[float, float]:
        """Suggest a volume-fraction interval."""
        if "volume_fraction" in self._global_bounds:
            min_vof, max_vof = self._global_bounds["volume_fraction"]
            return (max(0.1, min_vof - 0.05), min(0.5, max_vof + 0.05))
        return (0.15, 0.35)  # Default interval.
    
    def _generate_db_recommendation_text(self, conditions: Dict) -> str:
        """Generate database-retrieval recommendations."""
        lines = ["Based on Pareto analysis, adjust the retrieval conditions as follows:"]
        
        for obj_name, cond in conditions.items():
            if obj_name in self.objectives:
                obj = self.objectives[obj_name]
                lines.append(f"- {obj.display_name}: {cond['min']:.2f} ~ {cond['max']:.2f} {obj.unit}")
        
        return "\n".join(lines)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Return a summary of the current optimization state."""
        if not self.pareto_fronts:
            self.compute_pareto_fronts()
        
        pareto_optimal = self.get_pareto_optimal_solutions(5)
        
        status = {
            "total_solutions": len(self.solutions),
            "pareto_optimal_count": len(self.pareto_fronts[0]) if self.pareto_fronts else 0,
            "pareto_fronts_count": len(self.pareto_fronts),
            "objectives": list(self.objectives.keys()),
            "global_bounds": self._global_bounds.copy(),
            "best_solutions": []
        }
        
        for sol in pareto_optimal[:5]:
            status["best_solutions"].append({
                "id": sol.id,
                "source": sol.source,
                "properties": sol.properties,
                "rank": sol.rank,
                "crowding_distance": sol.crowding_distance
            })
        
        return status
    
    def get_pareto_summary_for_agents(self) -> str:
        """
        Generate a Pareto summary for agents.

        The result is injected into the conversation context.
        """
        if not self.pareto_fronts:
            self.compute_pareto_fronts()
        
        lines = ["\n[Multi-Objective Optimization Status (Pareto Analysis)]"]
        lines.append(f"Total candidates: {len(self.solutions)}")
        lines.append(f"Pareto-optimal solutions: {len(self.pareto_fronts[0]) if self.pareto_fronts else 0}")
        lines.append(f"Objectives: {', '.join(self.objectives.keys())}")
        lines.append("-" * 60)
        
        if self.pareto_fronts and self.pareto_fronts[0]:
            lines.append("Top Pareto-optimal solutions:")
            pareto_optimal = self.get_pareto_optimal_solutions(5)
            for i, sol in enumerate(pareto_optimal, 1):
                props_str = ", ".join([f"{k}={v:.3g}" for k, v in sol.properties.items()])
                lines.append(f"  {i}. [{sol.source}] {sol.id}: {props_str}")
        
        lines.append("-" * 60)
        
        # Add guidance recommendations.
        guidance = self.generate_guidance_for_structure_generator()
        if "recommendation" in guidance:
            lines.append(f"[Guidance for StructureGenerator]: {guidance['recommendation']}")
        
        return "\n".join(lines)


# Global optimizer instance.
_global_pareto_optimizer: Optional[ParetoOptimizer] = None


def get_pareto_optimizer() -> ParetoOptimizer:
    """Return the global Pareto optimizer instance."""
    global _global_pareto_optimizer
    if _global_pareto_optimizer is None:
        _global_pareto_optimizer = ParetoOptimizer()
    return _global_pareto_optimizer


def reset_pareto_optimizer() -> None:
    """Reset the Pareto optimizer for a new session."""
    global _global_pareto_optimizer
    _global_pareto_optimizer = ParetoOptimizer()


def inject_pareto_context(context_summary: str, property_table: List[Dict[str, Any]]) -> str:
    """
    Inject Pareto analysis into the context.

    Call this function from workflow_manager or main.py before building the
    augmented query.
    """
    optimizer = get_pareto_optimizer()
    
    # Update the solution set from property_table.
    optimizer.add_solutions_from_property_table(property_table)
    
    # Compute Pareto fronts.
    optimizer.compute_pareto_fronts()
    
    # Generate the summary.
    pareto_summary = optimizer.get_pareto_summary_for_agents()
    
    return context_summary + pareto_summary
