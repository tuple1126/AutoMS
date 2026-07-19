"""
Runtime integration for Simulation-Aware Evolutionary Search (SAES).

The integrator owns the paper-facing optimization loop: it receives verified
simulation feedback, updates the SAES history, maintains the Pareto frontier,
and emits the next round of guidance for the Generator and Simulator.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import copy
import math
from numbers import Real
from pathlib import Path

from tools.saes_optimizer import (
    SAES, 
    get_saes, 
    reset_saes,
    Chromosome,
    GeneticSourceType,
    SimulationFeedback
)
from tools.saes_guidance import (
    SAESGuidance,
    get_saes_guidance,
    reset_saes_guidance
)
from tools.pareto_optimizer import (
    ParetoOptimizer,
    Solution,
    Objective,
    ObjectiveType,
    get_pareto_optimizer
)


class SAESIntegrator:
    """
    Paper-facing SAES integration layer.

    The guidance helper and Pareto store are implementation details of SAES;
    they remain separate in code only to keep responsibilities small.
    """
    
    def __init__(self):
        self.saes = get_saes()
        self.saes_guidance = get_saes_guidance()
        self.pareto_optimizer = get_pareto_optimizer()
        
        # Simulation-result cache.
        self._simulation_cache: Dict[str, Dict[str, Any]] = {}
        self._simulation_provenance: Dict[str, List[Dict[str, Any]]] = {}
        
        # Parameter-to-microstructure mapping.
        self._param_structure_map: Dict[str, str] = {}
        
        # Integration state.
        self._is_initialized = False
        self._iteration_count = 0
        
        # Ablation support: single-source mode.
        self._single_source_mode = False
        self._active_source = None  # Either 'ai' or 'db', fixed in single-source mode.
        
    _PROPERTY_ALIASES = {
        "elastic_modulus": ("elastic_modulus", "youngs_modulus", "young_modulus", "e", "e_avg"),
        "shear_modulus": ("shear_modulus", "g", "g_avg"),
        "poisson_ratio": ("poisson_ratio", "nu", "nu_avg", "poisson"),
        "thermal_conductivity": ("thermal_conductivity", "thermal_conductivity_avg", "k", "k_eff"),
        "electrical_conductivity": ("electrical_conductivity", "conductivity", "sigma", "sigma_eff"),
        "volume_fraction": (
            "volume_fraction", "solid_fraction", "vof", "stiffness_volume_fraction",
            "thermal_volume_fraction", "electrical_volume_fraction",
        ),
        "eqps_rve_average": ("eqps_rve_average", "equivalent_plastic_strain"),
    }

    @staticmethod
    def _candidate_key(candidate_id: Any) -> str:
        text = str(candidate_id or "").replace("\\", "/")
        filename = text.rsplit("/", 1)[-1]
        return Path(filename).stem.casefold()

    @staticmethod
    def _finite_number(value: Any) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def _canonicalize_simulation_result(self, item: Dict[str, Any]) -> Dict[str, float]:
        """Flatten simulator output and expose property aliases under active objective keys."""
        effective_properties = item.get("effective_properties", {})
        flat = dict(effective_properties) if isinstance(effective_properties, dict) else {}
        flat.update(item)
        by_lower_name = {str(key).lower(): value for key, value in flat.items()}
        canonical: Dict[str, float] = {}

        for canonical_name, aliases in self._PROPERTY_ALIASES.items():
            for alias in aliases:
                value = self._finite_number(by_lower_name.get(alias))
                if value is not None:
                    canonical[canonical_name] = value
                    break

        for objective_name in self.saes.objectives:
            lower_objective = objective_name.lower()
            if lower_objective in by_lower_name:
                value = self._finite_number(by_lower_name[lower_objective])
                if value is not None:
                    canonical[objective_name] = value
                    continue
            for canonical_name, aliases in self._PROPERTY_ALIASES.items():
                if lower_objective == canonical_name or lower_objective in aliases:
                    if canonical_name in canonical:
                        canonical[objective_name] = canonical[canonical_name]
                    break

        return canonical

    @staticmethod
    def _candidate_provenance(source: str, candidate: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
        return {
            "candidate_id": candidate_id,
            "source": source,
            "registered_at": datetime.now().isoformat(),
            "candidate_metadata": copy.deepcopy(candidate),
            "simulation_history": [],
        }

    def set_single_source_mode(self, enabled: bool):
        """
        Configure single-source mode for an ablation experiment.

        When enabled, SAES uses only one source (AI or database), disabling the
        core dual-source coevolution mechanism to isolate its contribution.
        
        Args:
            enabled: Whether to enable single-source mode.
        """
        import random
        self._single_source_mode = enabled
        if enabled:
            # Select a source randomly to support a fair comparison.
            self._active_source = random.choice(['ai', 'db'])
            print("\n[SAES] Single-source mode enabled")
            print(f"   Active source: {'AI generation' if self._active_source == 'ai' else 'database retrieval'}")
            print("   Dual-source coevolution is disabled; using one source only")
        else:
            self._active_source = None
            print("\n[SAES] Dual-source coevolution mode enabled")
    
    def is_source_enabled(self, source_type: str) -> bool:
        """
        Return whether a source is enabled.
        
        Args:
            source_type: Either 'ai' or 'db'.
            
        Returns:
            Whether the source is enabled.
        """
        if not self._single_source_mode:
            return True  # Both sources are enabled in dual-source mode.
        return self._active_source == source_type
        
    def initialize(self, parsed_requirement: Dict[str, Any]):
        """
        Initialize the integrated optimizer.
        
        Args:
            parsed_requirement: Parsed result from RequirementParser.
        """
        # Extract target properties.
        target_properties = parsed_requirement.get("target_properties", {})
        
        # Convert to SAES objective format.
        saes_objectives = {}
        for prop_name, prop_spec in target_properties.items():
            obj_type = self._infer_objective_type(prop_name, prop_spec)
            target_value = self._extract_target_value(prop_spec)
            goal = self._infer_goal(prop_spec, target_value)
            
            saes_objectives[prop_name] = {
                "type": obj_type,
                "goal": goal,
                "target_value": target_value,
                "weight": 1.0
            }
        
        # Register with each component.
        self.saes.register_objectives(saes_objectives)
        self.saes_guidance.parse_multi_objective_requirement(parsed_requirement)
        
        self._is_initialized = True
        print("\n" + "=" * 70)
        print("[SAES] Simulation-Aware Evolutionary Search optimizer initialized")
        print("=" * 70)
        print(f"   Registered objectives: {len(saes_objectives)}")
        for name, spec in saes_objectives.items():
            print(f"   - {name}: {spec['type']}, target={spec.get('target_value', 'N/A')}")
        print("   Waiting for dual-source (AI + database) solutions before evolution begins...")
        print("=" * 70 + "\n")
    
    def _infer_objective_type(self, prop_name: str, prop_spec: Any) -> str:
        """Infer the objective type."""
        # Properties that are usually minimized.
        minimize_props = {"volume_fraction", "vof", "density", "cost"}
        
        if prop_name.lower() in minimize_props:
            return "minimize"
        return "maximize"

    def _infer_goal(self, prop_spec: Any, target_value: Optional[float]) -> str:
        """Distinguish paper target-matching tasks from true maximize/minimize tasks."""
        if isinstance(prop_spec, dict):
            if "min" in prop_spec and "max" in prop_spec:
                return "target_match"
            if "value" in prop_spec or "target" in prop_spec:
                return "target_match"
        if target_value is not None and not isinstance(prop_spec, dict):
            return "target_match"
        return "optimize"
    
    def _extract_target_value(self, prop_spec: Any) -> Optional[float]:
        """Extract a target value."""
        if isinstance(prop_spec, (int, float)):
            return float(prop_spec)
        if isinstance(prop_spec, dict):
            if "value" in prop_spec:
                return float(prop_spec["value"])
            if "target" in prop_spec:
                return float(prop_spec["target"])
            if "min" in prop_spec and "max" in prop_spec:
                return (float(prop_spec["min"]) + float(prop_spec["max"])) / 2.0
            if "min" in prop_spec:
                return float(prop_spec["min"])
            if "max" in prop_spec:
                return float(prop_spec["max"])
        return None
    
    def inject_ai_solution(self, 
                           generation_result: Dict[str, Any],
                           simulation_results: Dict[str, Any] = None):
        """
        Inject AI-generated solutions.
        
        Args:
            generation_result: Output from StructureGenerator.
            simulation_results: Optional output from Simulator.
            
        In single-source mode, this method returns silently when AI is not the
        active source.
        """
        # In single-source mode, skip when AI is not active.
        if self._single_source_mode and not self.is_source_enabled('ai'):
            print("\n[SAES] Single-source mode: AI source is inactive; skipping AI solution injection")
            return
            
        kept_structures = generation_result.get("kept_structures_details", [])
        
        for struct in kept_structures:
            filename = struct.get("filename", "")
            candidate_key = self._candidate_key(filename)
            scaled_props = struct.get("scaled_properties", {})
            
            # Extract parameters.
            params = {
                "E": scaled_props.get("E", 0),
                "G": scaled_props.get("G", 0),
                "nu": scaled_props.get("nu", 0)
            }
            
            # Find corresponding simulation results.
            sim_result = {}
            if simulation_results:
                sim_result = self._find_simulation_for_structure(filename, simulation_results)
            
            # Merge stiffness-simulation results.
            if sim_result:
                merged_result = {**scaled_props, **sim_result}
            else:
                merged_result = scaled_props
            
            # Add to the SAES population.
            self.saes.add_ai_solution(
                params=params,
                microstructure_id=filename,
                provenance=self._candidate_provenance("ai_generated", struct, candidate_key)
            )
            
            # Synchronize with ParetoOptimizer.
        
        mode_str = " (single source: AI)" if self._single_source_mode else ""
        print(f"\n[SAES] Injected {len(kept_structures)} AI-generated solutions into the population{mode_str}")
        pop_stats = self.saes.population_manager.get_population_stats()
        print(f"   Current population: AI={pop_stats['ai_subpop_size']}, DB={pop_stats['db_subpop_size']}")
    
    def inject_db_solution(self,
                           db_results: List[Dict[str, Any]],
                           simulation_results: Dict[str, Any] = None):
        """
        Inject database-retrieved solutions.
        
        Args:
            db_results: Database results returned by Generator's internal retrieval branch.
            simulation_results: Optional output from Simulator.
            
        In single-source mode, this method returns silently when the database is
        not the active source.
        """
        # In single-source mode, skip when the database is not active.
        if self._single_source_mode and not self.is_source_enabled('db'):
            print("\n[SAES] Single-source mode: database source is inactive; skipping database solution injection")
            return
            
        for item in db_results:
            filename = item.get("filename", item.get("name", ""))
            candidate_key = self._candidate_key(filename)
            
            # Extract parameters from original database properties.
            params = {
                "E": item.get("E", item.get("youngs_modulus", 0)),
                "G": item.get("G", item.get("shear_modulus", 0)),
                "nu": item.get("nu", item.get("poisson_ratio", 0.3))
            }
            
            # Find corresponding simulation results.
            sim_result = {}
            if simulation_results:
                sim_result = self._find_simulation_for_structure(filename, simulation_results)
            
            # Merge results.
            merged_result = {**item, **sim_result}
            
            # Add to the SAES population.
            self.saes.add_db_solution(
                params=params,
                microstructure_id=filename,
                provenance=self._candidate_provenance("database", item, candidate_key)
            )
            
            # Synchronize with ParetoOptimizer.
        
        mode_str = " (single source: DB)" if self._single_source_mode else ""
        print(f"\n[SAES] Injected {len(db_results)} database solutions into the population{mode_str}")
        pop_stats = self.saes.population_manager.get_population_stats()
        print(f"   Current population: AI={pop_stats['ai_subpop_size']}, DB={pop_stats['db_subpop_size']}")
    
    def inject_simulation_results(self, simulation_results: Dict[str, Any]):
        """
        Inject simulation results.

        Result format:
        {
            "analysis_type": "heat/stiffness/electrical",
            "results": [
                {"filename": "xxx.obj", "thermal_conductivity": 25.3, ...},
                ...
            ]
        }
        """
        analysis_type = simulation_results.get("analysis_type", "unknown")
        results = simulation_results.get("results", [])
        
        if isinstance(results, list):
            for item in results:
                filename = item.get("filename", item.get("file", ""))
                if not filename or item.get("success") is False:
                    continue
                candidate_key = self._candidate_key(filename)
                canonical = self._canonicalize_simulation_result(item)
                if canonical:
                    self._simulation_cache[candidate_key] = {
                        **self._simulation_cache.get(candidate_key, {}),
                        **canonical,
                    }
                self._simulation_provenance.setdefault(candidate_key, []).append({
                    "analysis_type": analysis_type,
                    "received_at": datetime.now().isoformat(),
                    "raw_result": copy.deepcopy(item),
                })
        
        # Update simulation results for matching chromosomes in the SAES population.
        self._update_population_simulation_results()
        
        print(f"\n[SAES] Simulation feedback: updated fitness for {len(results)} chromosomes")
        print(f"   Simulation type: {analysis_type}")
        
        # Compute and print Pareto status.
        self.saes.compute_pareto_front()
        pareto_size = len(self.saes.pareto_front)
        print(f"   Pareto-front update: {pareto_size} non-dominated solutions")
    
    def _find_simulation_for_structure(self, filename: str, 
                                        simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Find simulation results for a specific structure."""
        # Check the cache first.
        candidate_key = self._candidate_key(filename)
        if candidate_key in self._simulation_cache:
            return self._simulation_cache[candidate_key]
        
        # Check the result list.
        results = simulation_results.get("results", [])
        if isinstance(results, list):
            for item in results:
                item_filename = item.get("filename", item.get("file", ""))
                if item_filename and self._candidate_key(item_filename) == candidate_key:
                    return self._canonicalize_simulation_result(item)
        
        return {}
    
    def _update_population_simulation_results(self):
        """Update chromosome simulation results in the population."""
        for chr in self.saes.population_manager.combined_population:
            if not chr.microstructure_id:
                continue
            candidate_key = self._candidate_key(chr.microstructure_id)
            verified_values = self._simulation_cache.get(candidate_key)
            if not verified_values:
                continue

            recorded_feedback = self.saes.update_verified_simulation(chr, verified_values)
            history = self._simulation_provenance.get(candidate_key, [])
            if history:
                chr.provenance.setdefault("simulation_history", []).extend(history)
                self._simulation_provenance[candidate_key] = []

            if recorded_feedback:
                self._sync_to_pareto_optimizer(
                    chr.microstructure_id,
                    chr.source.value,
                    chr.simulation_results,
                )
    
    def _sync_to_pareto_optimizer(self, solution_id: str, source: str, properties: Dict[str, Any]):
        """Synchronize a solution with ParetoOptimizer."""
        self.pareto_optimizer.add_solution(Solution(
            id=solution_id,
            source=source,
            properties={k: float(v) for k, v in properties.items() if isinstance(v, (int, float))}
        ))
    
    def compute_pareto_front(self) -> Dict[str, Any]:
        """
        Compute and return the Pareto front.

        Update both SAES and ParetoOptimizer fronts.
        """
        # Compute the SAES front.
        saes_front = self.saes.compute_pareto_front()
        
        # Compute ParetoOptimizer fronts.
        pareto_fronts = self.pareto_optimizer.compute_pareto_fronts()
        
        # Combine results.
        result = {
            "saes_pareto_size": len(saes_front),
            "pareto_optimizer_fronts": len(pareto_fronts),
            "total_solutions": len(self.saes.population_manager.combined_population),
            "pareto_solutions": []
        }
        
        for chr in saes_front:
            result["pareto_solutions"].append({
                "id": chr.microstructure_id or chr.id,
                "source": chr.source.value,
                "properties": chr.simulation_results,
                "fitness": chr.fitness_values,
                "crowding_distance": chr.crowding_distance
            })
        
        return result
    
    def get_next_generation_guidance(self) -> Dict[str, Any]:
        """
        Return integrated guidance for the next generation.

        Combine SAES candidate updates and Pareto guidance.
        """
        # SAES candidate-parameter suggestions.
        saes_guidance = self.saes.get_guidance_for_structure_generator()
        
        # Strategy guidance from SAES Guidance.
        guidance = self.saes_guidance.generate_structure_generator_params()
        
        # Merge guidance.
        merged_guidance = {
            "source": "SAES Integration",
            "iteration": self._iteration_count,
            "generation": self.saes.population_manager.current_generation,
            
            # SAES parameter suggestions from local updates and retained exploration.
            "genetic_suggested_params": saes_guidance.get("suggested_params", []),
            
            # SAES Guidance parameter suggestions derived from Pareto analysis.
            "saes_guidance_suggested_params": {
                "E": guidance.get("E"),
                "G": guidance.get("G"),
                "nu": guidance.get("nu")
            },
            
            # Exploration mode.
            "exploration_mode": guidance.get("exploration_mode", "balanced"),
            
            # Evolution statistics.
            "evolution_stats": saes_guidance.get("evolution_stats", {}),
            
            # Pareto status.
            "pareto_front_size": saes_guidance.get("pareto_front_size", 0),
            
            # Convergence detection.
            "convergence": saes_guidance.get("convergence_status", (False, "")),
            
            # Combined recommendation.
            "recommendation": self._generate_merged_recommendation(
                saes_guidance, guidance
            )
        }
        
        self._iteration_count += 1
        
        return merged_guidance
    
    def _generate_merged_recommendation(self, 
                                         saes_guidance: Dict[str, Any],
                                         guidance: Dict[str, Any]) -> str:
        """Generate the merged recommendation."""
        lines = []
        lines.append("=" * 60)
        lines.append("[SAES Integrated Optimization Guidance]")
        lines.append("=" * 60)
        
        # SAES search status.
        gen = saes_guidance.get("evolution_stats", {}).get("generation", 0)
        pareto_size = saes_guidance.get("pareto_front_size", 0)
        lines.append(f"\nEvolution status: generation {gen}, Pareto front with {pareto_size} solutions")
        
        # Population composition.
        stats = saes_guidance.get("evolution_stats", {})
        ai_count = stats.get("ai_subpop_size", 0)
        db_count = stats.get("db_subpop_size", 0)
        lines.append(f"   Population: AI generated {ai_count}, database {db_count}")
        
        # Parameter suggestions.
        genetic_params = saes_guidance.get("suggested_params", [])
        if genetic_params:
            lines.append("\nSAES-suggested parameters:")
            for i, params in enumerate(genetic_params[:3], 1):
                lines.append(f"   Candidate {i}: E={params.get('E', 0):.1f}, "
                           f"G={params.get('G', 0):.1f}, nu={params.get('nu', 0):.3f}")
        
        # Exploration mode.
        mode = guidance.get("exploration_mode", "balanced")
        mode_desc = {
            "exploration": "broad exploration (early stage)",
            "balanced": "balanced mode (middle stage)",
            "exploitation": "focused exploitation (convergence stage)",
        }
        lines.append(f"\nExploration mode: {mode_desc.get(mode, mode)}")
        
        # Convergence status.
        converged, reason = saes_guidance.get("convergence_status", (False, ""))
        if converged:
            lines.append(f"\nConvergence detected: {reason}")
        else:
            lines.append(f"\nContinue optimization: {reason}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def record_iteration(self):
        """Record one iteration."""
        self.saes.record_generation()
        self.saes_guidance.current_iteration += 1
    
    def check_termination(self) -> Tuple[bool, str]:
        """Return whether optimization should terminate."""
        saes_converged, saes_reason = self.saes.check_convergence()
        saes_guidance_continue, saes_guidance_reason = self.saes_guidance.should_continue_iteration()
        
        if saes_converged:
            return True, f"SAES: {saes_reason}"
        if not saes_guidance_continue:
            return True, f"SAES Guidance: {saes_guidance_reason}"
        
        return False, "continue optimization"
    
    def get_final_report(self) -> Dict[str, Any]:
        """Return the final optimization report."""
        # SAES summary.
        saes_summary = self.saes.get_evolution_summary()
        
        # SAES Guidance report.
        saes_guidance_report = self.saes_guidance.generate_final_report()
        
        # Best solutions.
        best_solutions = self.saes.get_best_solutions(10)
        
        return {
            "optimization_method": "SAES Integration",
            "total_iterations": self._iteration_count,
            "total_generations": saes_summary.get("total_generations", 0),
            "total_evaluations": saes_summary.get("total_evaluations", 0),
            "final_pareto_size": saes_summary.get("final_pareto_size", 0),
            "convergence_status": self.check_termination(),
            "best_solutions": best_solutions,
            "saes_summary": saes_summary,
            "saes_guidance_report": saes_guidance_report,
            "evolution_history": self.saes.evolution_history
        }
    
    def get_context_injection(self) -> str:
        """
        Generate optimization-state information for agent-context injection.

        This is the core interface to the LightAgent main loop and includes
        SAES search status plus Pareto guidance status.
        """
        lines = []
        
        # ============== Part 1: SAES search status ==============
        lines.append("\n" + "=" * 70)
        lines.append("[SAES Status] Simulation-Aware Evolutionary Search")
        lines.append("=" * 70)
        
        # Evolution status.
        gen = self.saes.population_manager.current_generation
        lines.append(f"Generation: {gen}")
        
        # Population statistics.
        stats = self.saes.population_manager.get_population_stats()
        lines.append(f"Population size: AI generated {stats['ai_subpop_size']} + database {stats['db_subpop_size']}")
        lines.append(f"Evaluated: AI {stats['evaluated_ai']} + database {stats['evaluated_db']}")
        
        # Local-search landscape analysis.
        if hasattr(self.saes, 'fitness_landscape') and self.saes.fitness_landscape:
            landscape = self.saes.fitness_landscape
            if hasattr(landscape, 'get_landscape_analysis'):
                analysis = landscape.get_landscape_analysis()
                if analysis:
                    lines.append("\nLocal-search landscape analysis:")
                    lines.append(f"   - Exploration hot regions: {analysis.get('hot_regions', 'N/A')}")
                    lines.append(f"   - Local-optimum traps: {analysis.get('local_optima', 'N/A')}")
        
        # Pareto front.
        pareto_size = len(self.saes.pareto_front)
        lines.append(f"\nPareto front: {pareto_size} non-dominated solutions")
        
        if pareto_size > 0:
            lines.append("Top 3 Pareto solutions:")
            for i, chr in enumerate(self.saes.pareto_front[:3], 1):
                # Safely format a mixture of string and numeric values.
                props_list = []
                for k, v in list(chr.simulation_results.items())[:3]:
                    if isinstance(v, (int, float)):
                        props_list.append(f"{k}={v:.3g}")
                    else:
                        props_list.append(f"{k}={v}")
                props = ", ".join(props_list)
                lines.append(f"  {i}. [{chr.source.value}] {chr.microstructure_id}: {props}")
        
        # Next-generation parameter suggestions.
        next_params = self.saes.get_next_generation_params(2)
        if next_params:
            lines.append("\n[Evolutionary Suggested Parameters] Next-generation structure suggestions:")
            for i, params in enumerate(next_params, 1):
                lines.append(f"  Candidate {i}: E={params['E']:.1f} MPa, G={params['G']:.1f} MPa, nu={params['nu']:.3f}")
        
        # ============== Part 2: SAES Guidance multi-objective strategy ==============
        lines.append("\n" + "-" * 70)
        lines.append("[SAES Guidance] Multi-Objective Guidance Strategy")
        lines.append("-" * 70)
        
        # Optimization objectives.
        if self.saes_guidance.optimizer.objectives:
            lines.append("\nOptimization objective configuration:")
            for obj_name, obj in self.saes_guidance.optimizer.objectives.items():
                target_str = f", target={obj.target_value}" if obj.target_value else ""
                weight = getattr(obj, 'weight', 1.0)
                lines.append(f"   - {obj.display_name}: {obj.obj_type.value}, weight={weight:.2f}{target_str}")
        
        # Current iteration.
        lines.append(f"\nOptimization iteration: {self.saes_guidance.current_iteration}/{self.saes_guidance.max_iterations}")
        lines.append(f"   Evaluated solutions: {len(self.saes_guidance.optimizer.solutions)}")
        
        # Pareto-front analysis.
        if self.saes_guidance.optimizer.pareto_fronts:
            pareto_solutions = self.saes_guidance.optimizer.get_pareto_optimal_solutions(3)
            if pareto_solutions:
                lines.append("\nSAES Guidance Pareto-optimal recommendations:")
                for i, sol in enumerate(pareto_solutions, 1):
                    props = ", ".join([f"{k}={v:.3g}" if isinstance(v, (int, float)) else f"{k}={v}" 
                                       for k, v in list(sol.properties.items())[:4]])
                    lines.append(f"   {i}. [{sol.source}] {sol.id}: {props}")
        
        # Iteration strategy recommendation.
        should_continue, reason = self.saes_guidance.should_continue_iteration()
        if should_continue:
            lines.append(f"\nSAES Guidance recommendation: continue iterating - {reason}")
        else:
            lines.append(f"\nSAES Guidance recommendation: may terminate - {reason}")
        
        # ============== Part 3: integrated convergence status ==============
        lines.append("\n" + "=" * 70)
        lines.append("[Integrated Optimization Status]")
        lines.append("=" * 70)
        
        converged, reason = self.check_termination()
        status = "converged - recommend terminating iterations" if converged else "continue evolution - optimum not yet reached"
        lines.append(f"Optimization status: {status}")
        lines.append(f"Reason: {reason}")
        
        # When evolution continues, provide concrete next actions.
        if not converged:
            lines.append("\nRecommended next actions:")
            lines.append("   1. StructureGenerator: generate new microstructure parameters from SAES guidance")
            lines.append("   2. Simulator: run the requested simulations for the new structures")
            lines.append("   3. Feed simulation results back to SAES for the next local update")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# Global instance.
_global_integrator: Optional[SAESIntegrator] = None


def get_saes_integrator(reset: bool = False) -> SAESIntegrator:
    """Return the global integrator instance."""
    global _global_integrator
    if _global_integrator is None or reset:
        _global_integrator = SAESIntegrator()
    return _global_integrator


def reset_saes_integrator():
    """Reset the integrator."""
    global _global_integrator
    reset_saes()
    reset_saes_guidance()
    _global_integrator = None


def is_saes_enabled() -> bool:
    """Return whether SAES is enabled (ablation support)."""
    import os
    # Explicitly disable SAES when the ablation flag is set.
    if os.environ.get("CHATMS_ABLATION_SAES", "0") == "1":
        return False
    global _global_integrator
    return _global_integrator is not None and _global_integrator._is_initialized
