"""
SAES: Simulation-Aware Evolutionary Search.

This module implements the optimization loop described in the paper:
1. Local gradient approximation from simulation history with spatial and temporal weighting.
2. Gradient-guided parameter updates with retained stochastic exploration.
3. Pareto-driven selection with adaptive weighting for multi-objective search.

Retrieval- and generation-sourced candidates share one verified population so
the SAES loop can reuse all available simulation feedback.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import copy
from datetime import datetime
import random
import os


def is_history_mutation_enabled() -> bool:
    """
    Return whether History-Driven Mutation is enabled (ablation support).

    When CHATMS_ABLATION_HISTORY_MUTATION=1, use purely random mutation in
    place of gradient-guided mutation to isolate HDMO's contribution.
    
    Returns:
        True: Use full gradient-guided mutation (HDMO).
        False: Use purely random mutation (ablation mode).
    """
    return os.environ.get("CHATMS_ABLATION_HISTORY_MUTATION", "0") != "1"


class GeneticSourceType(Enum):
    """Source type for an individual."""
    AI_GENERATED = "ai_generated"
    DATABASE = "database"
    CROSSOVER = "crossover"      # Produced by crossover.
    MUTATION = "mutation"        # Produced by mutation.
    HYBRID = "hybrid"            # Produced from mixed sources.
    SAES_GUIDED = "saes_guided"  # Produced by SAES gradient guidance.


@dataclass
class Chromosome:
    """
    Chromosome encoding microstructure-generation parameters.

    Genes:
    - E: Young's modulus (MPa)
    - G: Shear modulus (MPa)
    - nu: Poisson's ratio
    - vof_hint: Optional volume-fraction hint
    """
    genes: Dict[str, float]  # {param_name: value}
    
    # Metadata.
    source: GeneticSourceType = GeneticSourceType.AI_GENERATED
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    
    # Fitness data, populated after simulation.
    fitness_values: Dict[str, float] = field(default_factory=dict)  # Multi-objective fitness.
    pareto_rank: int = -1
    crowding_distance: float = 0.0
    is_evaluated: bool = False
    
    # Associated microstructure.
    microstructure_id: Optional[str] = None
    simulation_results: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.id = f"chr_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}"
    
    def clone(self) -> 'Chromosome':
        """Return a deep copy."""
        new_chr = Chromosome(
            genes=copy.deepcopy(self.genes),
            source=self.source,
            parent_ids=self.parent_ids.copy(),
            generation=self.generation
        )
        new_chr.id = f"chr_{datetime.now().strftime('%H%M%S')}_{random.randint(1000, 9999)}"
        return new_chr


@dataclass
class SimulationFeedback:
    """
    Simulation-feedback record used to construct the fitness landscape.
    """
    chromosome_id: str
    input_params: Dict[str, float]  # E, G, nu
    output_properties: Dict[str, float]  # thermal_conductivity, youngs_modulus, etc.
    target_properties: Dict[str, float]  # User targets.
    relative_errors: Dict[str, float]  # Relative error by property.
    timestamp: str
    success: bool


class SimulationAwareFitnessLandscape:
    """
    Feature 2: Simulation-Aware Fitness Landscape (SAFL).

    Core approach:
    - Build a dynamic landscape from simulation history instead of replacing
      simulations with a surrogate model.
    - Update the landscape after every simulation for more accurate fitness estimates.
    - Support multi-objective Pareto dominance.

    Robust gradient estimation:
    - Use weighted least squares instead of simple finite differences.
    - Weight samples by distance and quality.
    - Detect and remove outliers automatically.
    - Estimate gradient confidence.
    """
    
    def __init__(self):
        self.feedback_history: List[SimulationFeedback] = []
        self.param_bounds: Dict[str, Tuple[float, float]] = {
            "E": (100, 100000),     # MPa
            "G": (50, 50000),       # MPa
            "nu": (0.1, 0.49),      # Physical Poisson's-ratio bounds.
        }
        self.target_objectives: Dict[str, Dict[str, Any]] = {}  # User targets.
        
        # Gradient-approximation cache used by HDMO.
        self._gradient_cache: Dict[str, np.ndarray] = {}
        # Gradient-confidence cache (0-1, higher is more reliable).
        self._gradient_confidence: Dict[str, float] = {}
        self._landscape_version = 0
        
        # Robust-gradient-estimation configuration.
        self._gradient_config = {
            "min_samples": 5,           # Minimum sample count.
            "neighbor_window": 5,       # SAES local-neighborhood window M.
            "max_samples": 30,          # Maximum number of samples to use.
            "outlier_threshold": 2.5,   # Outlier-detection threshold in MAD multiples.
            "temporal_decay": 0.5,      # Temporal decay factor.
            "min_confidence": 0.3,      # Minimum confidence threshold.
        }
        
    def register_objectives(self, objectives: Dict[str, Dict[str, Any]]):
        """Register optimization objectives."""
        self.target_objectives = objectives
        print(f"[SAES] Registered {len(objectives)} local-search objectives")
        
    def record_feedback(self, feedback: SimulationFeedback):
        """Record simulation feedback and update the landscape."""
        self.feedback_history.append(feedback)
        self._landscape_version += 1
        
        # Update the gradient cache.
        if len(self.feedback_history) >= 2:
            self._update_gradient_cache()
    
    def _update_gradient_cache(self):
        """
        Robust Gradient Estimation.

        Improvements:
        1. Use weighted least squares (WLS) instead of simple differences.
        2. Apply Gaussian weights by sample distance, favoring recent samples.
        3. Use median absolute deviation (MAD) for outlier detection.
        4. Compute gradient confidence for later mutation-strength adjustment.
        5. Normalize parameters to balance contributions across dimensions.

        In ablation mode, CHATMS_ABLATION_HISTORY_MUTATION=1 skips gradient
        computation.
        """
        # === Ablation check ===
        if not is_history_mutation_enabled():
            # Ablation mode skips gradients and uses purely random mutation.
            return
        
        min_samples = self._gradient_config["min_samples"]
        max_samples = self._gradient_config["max_samples"]
        
        if len(self.feedback_history) < min_samples:
            return
        
        # Retrieve recent samples.
        recent = self.feedback_history[-max_samples:]
        n_samples = len(recent)
        
        for obj_name in self.target_objectives:
            gradient, confidence = self._estimate_gradient_robust(recent, obj_name)
            
            if gradient is not None and confidence > self._gradient_config["min_confidence"]:
                self._gradient_cache[obj_name] = gradient
                self._gradient_confidence[obj_name] = confidence
            elif obj_name in self._gradient_cache:
                # With insufficient confidence, decay the existing gradient rather than discard it.
                self._gradient_confidence[obj_name] *= 0.9
    
    def _estimate_gradient_robust(self, samples: List[SimulationFeedback], 
                                   obj_name: str) -> Tuple[Optional[np.ndarray], float]:
        """
        Estimate a gradient with weighted least squares.

        Mathematical basis:
        The objective can be locally approximated as f(x) ~= f(x0) + grad(f) * (x - x0).
        WLS solves min sum(w_i * (y_i - grad(f) * delta_x_i)^2).
        
        Args:
            samples: Simulation-feedback samples.
            obj_name: Objective-property name.
            
        Returns:
            (gradient, confidence): Gradient vector and confidence.
        """
        # Extract valid data points.
        data_points = []
        for fb in samples:
            if obj_name not in fb.output_properties:
                continue
            
            # Normalize parameters to [0, 1].
            E_norm = (fb.input_params.get("E", 0) - self.param_bounds["E"][0]) / \
                     (self.param_bounds["E"][1] - self.param_bounds["E"][0])
            G_norm = (fb.input_params.get("G", 0) - self.param_bounds["G"][0]) / \
                     (self.param_bounds["G"][1] - self.param_bounds["G"][0])
            nu_norm = (fb.input_params.get("nu", 0) - self.param_bounds["nu"][0]) / \
                      (self.param_bounds["nu"][1] - self.param_bounds["nu"][0])
            
            data_points.append({
                "params": np.array([E_norm, G_norm, nu_norm]),
                "value": fb.output_properties[obj_name],
                "index": len(data_points)  # Temporal index used for time weights.
            })
        
        if len(data_points) < self._gradient_config["min_samples"]:
            return None, 0.0
        
        # Detect and remove outliers.
        data_points = self._remove_outliers(data_points)
        
        if len(data_points) < 3:
            return None, 0.0
        
        # Build the design matrix and objective vector.
        n = len(data_points)
        
        # Use the most recent point as the reference.
        ref_point = data_points[-1]
        ref_params = ref_point["params"]
        ref_value = ref_point["value"]
        
        # Build difference data.
        X = []  # Parameter-difference matrix.
        y = []  # Objective-difference vector.
        weights = []  # Weight vector.
        
        for i, dp in enumerate(data_points[:-1]):
            delta_params = dp["params"] - ref_params
            delta_value = dp["value"] - ref_value
            
            # Skip near-identical parameter values, which may be noise dominated.
            if np.linalg.norm(delta_params) < 1e-6:
                continue
            
            X.append(delta_params)
            y.append(delta_value)
            
            # Compute weights from temporal and distance decay.
            time_weight = np.exp(-self._gradient_config["temporal_decay"] * (n - 1 - i) / n)
            distance_weight = 1.0 / (1.0 + np.linalg.norm(delta_params) ** 2)
            weights.append(time_weight * distance_weight)
        
        if len(X) < 3:
            return None, 0.0
        
        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)
        
        # Weighted least-squares solve: (X^T W X)^(-1) X^T W y.
        try:
            W = np.diag(weights)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            
            # Add regularization to avoid a singular matrix.
            regularization = 1e-6 * np.eye(3)
            gradient_normalized = np.linalg.solve(XtWX + regularization, XtWy)
            
            # Convert to a direction in original parameter space; normalize magnitude.
            grad_norm = np.linalg.norm(gradient_normalized)
            if grad_norm > 1e-10:
                gradient = gradient_normalized / grad_norm
            else:
                return None, 0.0
            
            # Compute confidence.
            confidence = self._compute_gradient_confidence(X, y, weights, gradient_normalized)
            
            return gradient, confidence
            
        except np.linalg.LinAlgError:
            # Singular matrix: fall back to the simpler method.
            return self._fallback_gradient_estimation(data_points, obj_name)
    
    def _remove_outliers(self, data_points: List[Dict]) -> List[Dict]:
        """
        Detect and remove outliers with median absolute deviation (MAD).

        MAD is more robust than standard deviation and less sensitive to outliers.
        """
        if len(data_points) < 5:
            return data_points
        
        values = np.array([dp["value"] for dp in data_points])
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))
        
        if mad < 1e-10:
            # MAD of zero indicates tightly clustered data; retain all points.
            return data_points
        
        # Use MAD-based robust normalization.
        threshold = self._gradient_config["outlier_threshold"]
        filtered = []
        for dp in data_points:
            # Modified Z score: |x - median| / (1.4826 * MAD).
            # 1.4826 makes MAD consistent with standard deviation.
            z_score = abs(dp["value"] - median_val) / (1.4826 * mad)
            if z_score <= threshold:
                filtered.append(dp)
        
        return filtered
    
    def _compute_gradient_confidence(self, X: np.ndarray, y: np.ndarray, 
                                      weights: np.ndarray, gradient: np.ndarray) -> float:
        """
        Compute confidence in a gradient estimate.

        Factors:
        1. Goodness of fit (R^2).
        2. Sample count.
        3. Gradient-direction consistency.
        """
        n_samples = len(y)
        
        # 1. Compute weighted R^2.
        y_pred = X @ gradient
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
        
        if ss_tot < 1e-10:
            r_squared = 0.0
        else:
            r_squared = max(0, 1 - ss_res / ss_tot)
        
        # 2. Sample-count factor (sigmoid form).
        sample_factor = 1 - np.exp(-n_samples / 10)
        
        # 3. Gradient-direction consistency across local data points.
        direction_consistency = self._compute_direction_consistency(X, y, gradient)
        
        # Combine confidence factors.
        confidence = (0.4 * r_squared + 
                     0.3 * sample_factor + 
                     0.3 * direction_consistency)
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _compute_direction_consistency(self, X: np.ndarray, y: np.ndarray, 
                                        gradient: np.ndarray) -> float:
        """
        Compute gradient-direction consistency.

        Compare local directions at each data point with the estimated gradient.
        """
        if len(X) < 2:
            return 0.5
        
        consistent_count = 0
        total_count = 0
        
        for i in range(len(X)):
            delta_x = X[i]
            delta_y = y[i]
            
            # Predicted direction of change.
            predicted_sign = np.sign(np.dot(gradient, delta_x))
            # Observed direction of change.
            actual_sign = np.sign(delta_y)
            
            if abs(delta_y) > 1e-10 and abs(np.dot(gradient, delta_x)) > 1e-10:
                total_count += 1
                if predicted_sign == actual_sign:
                    consistent_count += 1
        
        if total_count == 0:
            return 0.5
        
        return consistent_count / total_count
    
    def _fallback_gradient_estimation(self, data_points: List[Dict], 
                                       obj_name: str) -> Tuple[Optional[np.ndarray], float]:
        """
        Simpler fallback gradient-estimation method.

        Used when the WLS method fails.
        """
        if len(data_points) < 2:
            return None, 0.0
        
        gradients = []
        
        for i in range(len(data_points) - 1):
            dp1 = data_points[i]
            dp2 = data_points[i + 1]
            
            delta_params = dp2["params"] - dp1["params"]
            delta_value = dp2["value"] - dp1["value"]
            
            param_norm = np.linalg.norm(delta_params)
            if param_norm > 1e-10 and abs(delta_value) > 1e-10:
                # Retain direction and relative magnitude information.
                grad = delta_params * delta_value / (param_norm ** 2)
                gradients.append(grad)
        
        if not gradients:
            return None, 0.0
        
        # Use a median rather than a mean for robustness.
        gradient = np.median(gradients, axis=0)
        grad_norm = np.linalg.norm(gradient)
        
        if grad_norm > 1e-10:
            gradient = gradient / grad_norm
            # The fallback method has lower confidence.
            confidence = 0.4 * len(gradients) / max(10, len(gradients))
            return gradient, confidence
        
        return None, 0.0
    
    def get_gradient_direction(self, objective_name: str, maximize: bool = True) -> np.ndarray:
        """
        Return an objective's gradient direction for HDMO.
        
        Returns:
            Normalized gradient-direction vector (3 dimensions: E, G, nu).
        """
        if objective_name not in self._gradient_cache:
            return np.zeros(3)
        
        grad = self._gradient_cache[objective_name]
        return grad if maximize else -grad
    
    def get_gradient_confidence(self, objective_name: str) -> float:
        """
        Return the confidence of a gradient estimate.

        HDMO uses it to adapt gradient-guidance strength dynamically.
        
        Returns:
            Confidence from 0 to 1; higher values indicate a more reliable estimate.
        """
        return self._gradient_confidence.get(objective_name, 0.0)
    
    def get_gradient_info(self, objective_name: str) -> Dict[str, Any]:
        """
        Return complete gradient information.
        
        Returns:
            Dictionary containing gradient direction, confidence, sample count, and related data.
        """
        return {
            "gradient": self._gradient_cache.get(objective_name, np.zeros(3)),
            "confidence": self._gradient_confidence.get(objective_name, 0.0),
            "total_samples": len(self.feedback_history),
            "landscape_version": self._landscape_version,
        }

    def estimate_local_gradient_at(self, params: Dict[str, float], obj_name: str) -> Tuple[Optional[np.ndarray], float]:
        """
        Estimate a local gradient near a point using the paper's SAES local-neighborhood definition.
        """
        valid = [fb for fb in self.feedback_history if obj_name in fb.output_properties]
        if len(valid) < self._gradient_config["min_samples"]:
            return None, 0.0

        anchor = np.array([
            (params.get("E", 0) - self.param_bounds["E"][0]) /
            (self.param_bounds["E"][1] - self.param_bounds["E"][0]),
            (params.get("G", 0) - self.param_bounds["G"][0]) /
            (self.param_bounds["G"][1] - self.param_bounds["G"][0]),
            (params.get("nu", 0) - self.param_bounds["nu"][0]) /
            (self.param_bounds["nu"][1] - self.param_bounds["nu"][0]),
        ])

        points = []
        total = len(valid)
        for idx, fb in enumerate(valid):
            point = np.array([
                (fb.input_params.get("E", 0) - self.param_bounds["E"][0]) /
                (self.param_bounds["E"][1] - self.param_bounds["E"][0]),
                (fb.input_params.get("G", 0) - self.param_bounds["G"][0]) /
                (self.param_bounds["G"][1] - self.param_bounds["G"][0]),
                (fb.input_params.get("nu", 0) - self.param_bounds["nu"][0]) /
                (self.param_bounds["nu"][1] - self.param_bounds["nu"][0]),
            ])
            points.append({
                "params": point,
                "value": fb.output_properties[obj_name],
                "time_index": idx,
                "distance": float(np.linalg.norm(point - anchor)),
            })

        neighbors = sorted(points, key=lambda item: item["distance"])[
            :self._gradient_config["neighbor_window"]
        ]
        neighbors = self._remove_outliers(neighbors)
        if len(neighbors) < 3:
            return None, 0.0

        anchor_value = min(neighbors, key=lambda item: item["distance"])["value"]
        latest_index = max(item["time_index"] for item in neighbors)
        X, y, weights = [], [], []

        for item in neighbors:
            delta_params = item["params"] - anchor
            if np.linalg.norm(delta_params) < 1e-6:
                continue
            X.append(delta_params)
            y.append(item["value"] - anchor_value)
            distance_weight = 1.0 / (1.0 + item["distance"] ** 2)
            freshness_gap = max(latest_index - item["time_index"], 0)
            time_weight = np.exp(
                -self._gradient_config["temporal_decay"] * freshness_gap / max(total, 1)
            )
            weights.append(distance_weight * time_weight)

        if len(X) < 3:
            return None, 0.0

        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)

        try:
            W = np.diag(weights)
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            gradient_raw = np.linalg.solve(XtWX + 1e-6 * np.eye(3), XtWy)
            grad_norm = np.linalg.norm(gradient_raw)
            if grad_norm <= 1e-10:
                return None, 0.0
            gradient = gradient_raw / grad_norm
            confidence = self._compute_gradient_confidence(X, y, weights, gradient_raw)
            return gradient, confidence
        except np.linalg.LinAlgError:
            return self._fallback_gradient_estimation(neighbors, obj_name)
    
    def evaluate_fitness(self, chromosome: Chromosome) -> Dict[str, float]:
        """
        Evaluate fitness from simulation results.

        This is not a surrogate-model prediction; fitness comes from actual
        simulation results.
        """
        fitness = {}
        
        if not chromosome.is_evaluated:
            # An unevaluated chromosome has no fitness until it is simulated.
            return fitness
        
        for obj_name, obj_spec in self.target_objectives.items():
            if obj_name in chromosome.simulation_results:
                value = chromosome.simulation_results[obj_name]
                target = obj_spec.get("target_value")
                obj_type = obj_spec.get("type", "maximize")
                
                if target is not None:
                    # With a target value, compute normalized error.
                    error = abs(value - target) / (abs(target) + 1e-10)
                    fitness[obj_name] = 1.0 / (1.0 + error)  # Smaller errors yield higher fitness.
                else:
                    # Without a target, use the normalized value directly.
                    fitness[obj_name] = value
        
        chromosome.fitness_values = fitness
        return fitness
    
    def estimate_fitness_without_simulation(self, chromosome: Chromosome) -> Dict[str, float]:
        """
        Estimate fitness from historical samples before selection.

        This is only a rough estimate and does not replace an actual simulation.
        """
        if len(self.feedback_history) < 3:
            return {}
        
        # Find the nearest historical sample.
        genes = chromosome.genes
        min_dist = float('inf')
        best_match = None
        
        for fb in self.feedback_history:
            dist = (
                ((genes.get("E", 0) - fb.input_params.get("E", 0)) / 1000) ** 2 +
                ((genes.get("G", 0) - fb.input_params.get("G", 0)) / 1000) ** 2 +
                ((genes.get("nu", 0) - fb.input_params.get("nu", 0)) / 0.1) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                best_match = fb
        
        if best_match:
            # Return the nearest sample's fitness as an estimate.
            estimated_fitness = {}
            locality_penalty = 1.0 / (1.0 + min_dist)
            for obj_name, obj_spec in self.target_objectives.items():
                if obj_name in best_match.output_properties:
                    value = best_match.output_properties[obj_name]
                    target = obj_spec.get("target_value")
                    if target is not None:
                        error = abs(value - target) / (abs(target) + 1e-10)
                        estimated_fitness[obj_name] = locality_penalty / (1.0 + error)
                    else:
                        estimated_fitness[obj_name] = locality_penalty * value
            return estimated_fitness
        
        return {}
    
    def get_landscape_analysis(self) -> Dict[str, Any]:
        """
        Return fitness-landscape analysis data.

        The result describes current landscape statistics for context injection.
        """
        analysis = {
            "total_samples": len(self.feedback_history),
            "registered_objectives": len(self.target_objectives),
            "gradient_cached_objectives": len(self._gradient_cache),
            "landscape_version": self._landscape_version,
            "hot_regions": 0,
            "local_optima": 0,
        }
        
        if len(self.feedback_history) < 3:
            return analysis
        
        # Analyze gradient-direction consistency to detect local-optimum traps.
        if self._gradient_cache:
            gradient_magnitudes = [np.linalg.norm(g) for g in self._gradient_cache.values()]
            avg_magnitude = np.mean(gradient_magnitudes) if gradient_magnitudes else 0
            
            # Very small gradients may indicate a local optimum.
            if avg_magnitude < 0.01:
                analysis["local_optima"] = 1
                analysis["warning"] = "A local optimum may have been reached; consider increasing the mutation rate."
        
        # Analyze exploration hot regions with dense parameter-space samples.
        if len(self.feedback_history) >= 5:
            E_values = [fb.input_params.get("E", 0) for fb in self.feedback_history]
            G_values = [fb.input_params.get("G", 0) for fb in self.feedback_history]
            
            # Simple hot-region detection based on parameter-distribution standard deviation.
            E_std = np.std(E_values) if E_values else 0
            G_std = np.std(G_values) if G_values else 0
            
            # A small standard deviation means exploration is concentrated.
            if E_std < 100 and G_std < 100:
                analysis["hot_regions"] = 1
            elif E_std > 1000 or G_std > 1000:
                analysis["hot_regions"] = 3  # Several dispersed regions.
            else:
                analysis["hot_regions"] = 2
        
        return analysis


class HybridPopulationManager:
    """
    Feature 1: Hybrid Population Coevolution (HPCE).

    Manage two heterogeneous subpopulations:
    - AI-generated subpopulation: microstructures produced by the diffusion model.
    - Database subpopulation: microstructures retrieved from the database.

    Cross-population crossover combines strengths from both sources.
    """
    
    def __init__(self, population_size: int = 20, elite_ratio: float = 0.2):
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        
        # Two subpopulations.
        self.ai_subpop: List[Chromosome] = []
        self.db_subpop: List[Chromosome] = []
        
        # Combined population used for selection and crossover.
        self.combined_population: List[Chromosome] = []
        
        # Elite archive retaining historical best candidates.
        self.elite_archive: List[Chromosome] = []
        self.max_archive_size = 50
        
        # Generation counter.
        self.current_generation = 0
        
    def add_ai_individual(self, genes: Dict[str, float], 
                         microstructure_id: str,
                         simulation_results: Dict[str, Any] = None,
                         is_evaluated: bool = False,
                         provenance: Dict[str, Any] = None) -> Chromosome:
        """Add an AI-generated individual."""
        chr = Chromosome(
            genes=genes,
            source=GeneticSourceType.AI_GENERATED,
            generation=self.current_generation
        )
        chr.microstructure_id = microstructure_id
        
        if simulation_results:
            chr.simulation_results = simulation_results
        chr.is_evaluated = bool(is_evaluated)
        if provenance:
            chr.provenance = copy.deepcopy(provenance)
        
        self.ai_subpop.append(chr)
        self._update_combined()
        return chr
    
    def add_db_individual(self, genes: Dict[str, float],
                          microstructure_id: str,
                          simulation_results: Dict[str, Any] = None,
                          is_evaluated: bool = False,
                          provenance: Dict[str, Any] = None) -> Chromosome:
        """Add a database-retrieved individual."""
        chr = Chromosome(
            genes=genes,
            source=GeneticSourceType.DATABASE,
            generation=self.current_generation
        )
        chr.microstructure_id = microstructure_id
        
        if simulation_results:
            chr.simulation_results = simulation_results
        chr.is_evaluated = bool(is_evaluated)
        if provenance:
            chr.provenance = copy.deepcopy(provenance)
        
        self.db_subpop.append(chr)
        self._update_combined()
        return chr
    
    def _update_combined(self):
        """Update the combined population."""
        self.combined_population = self.ai_subpop + self.db_subpop
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Return population statistics."""
        evaluated_ai = sum(1 for c in self.ai_subpop if c.is_evaluated)
        evaluated_db = sum(1 for c in self.db_subpop if c.is_evaluated)
        
        return {
            "generation": self.current_generation,
            "ai_subpop_size": len(self.ai_subpop),
            "db_subpop_size": len(self.db_subpop),
            "total_size": len(self.combined_population),
            "evaluated_ai": evaluated_ai,
            "evaluated_db": evaluated_db,
            "elite_archive_size": len(self.elite_archive)
        }
    
    def update_elite_archive(self, pareto_front: List[Chromosome]):
        """Update the elite archive."""
        for chr in pareto_front:
            if chr.id not in [e.id for e in self.elite_archive]:
                self.elite_archive.append(chr.clone())
        
        # Limit archive size.
        if len(self.elite_archive) > self.max_archive_size:
            # Retain diverse candidates by crowding distance.
            self.elite_archive.sort(key=lambda x: x.crowding_distance, reverse=True)
            self.elite_archive = self.elite_archive[:self.max_archive_size]
    
    def advance_generation(self):
        """Advance to the next generation."""
        self.current_generation += 1


class GeneticOperators:
    """
    Collection of genetic operators.

    Includes:
    - Feature 3: Parameter-Space Genetic Encoding (PSGE).
    - Feature 4: Crowding-Guided Selection (CGSP).
    - Feature 5: History-Driven Mutation Operator (HDMO).
    """
    
    def __init__(self, 
                 fitness_landscape: SimulationAwareFitnessLandscape,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 mutation_strength: float = 0.1):
        self.landscape = fitness_landscape
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        
        # Parameter bounds.
        self.param_bounds = {
            "E": (100, 100000),
            "G": (50, 50000),
            "nu": (0.1, 0.49),
        }
    
    # ==================== Selection operators ====================
    
    def tournament_selection(self, population: List[Chromosome], 
                             tournament_size: int = 3) -> Chromosome:
        """
        Feature 4: Crowding-guided tournament selection (CGSP).

        Selection rules:
        1. Prefer individuals with lower Pareto rank.
        2. For equal rank, prefer larger crowding distance to preserve diversity.
        """
        candidates = random.sample(population, min(tournament_size, len(population)))
        
        # Sort by (pareto_rank, -crowding_distance).
        candidates.sort(key=lambda x: (
            x.pareto_rank if x.pareto_rank >= 0 else float('inf'),
            -x.crowding_distance
        ))
        
        return candidates[0]
    
    def crowding_guided_selection(self, population: List[Chromosome], 
                                   n_select: int) -> List[Chromosome]:
        """
        Batch selection guided by crowding distance.

        The strategy keeps selected individuals evenly distributed in objective space.
        """
        if len(population) <= n_select:
            return population.copy()
        
        selected = []
        remaining = population.copy()
        
        # Select Pareto-front individuals first.
        pareto_front = [c for c in remaining if c.pareto_rank == 0]
        
        for _ in range(n_select):
            if pareto_front and len(pareto_front) > 0:
                # Prefer Pareto-front members ordered by crowding distance.
                pareto_front.sort(key=lambda x: x.crowding_distance, reverse=True)
                chosen = pareto_front.pop(0)
            elif remaining:
                # Once the Pareto front is exhausted, use tournament selection.
                chosen = self.tournament_selection(remaining)
            else:
                break
            
            selected.append(chosen)
            if chosen in remaining:
                remaining.remove(chosen)
        
        return selected
    
    # ==================== Crossover operators ====================
    
    def simulated_binary_crossover(self, parent1: Chromosome, 
                                    parent2: Chromosome,
                                    eta: float = 20) -> Tuple[Chromosome, Chromosome]:
        """
        Simulated binary crossover (SBX).

        Crossover real-valued chromosomes to generate two offspring.
        """
        if random.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()
        
        child1_genes = {}
        child2_genes = {}
        
        for param in ["E", "G", "nu"]:
            p1_val = parent1.genes.get(param, 0)
            p2_val = parent2.genes.get(param, 0)
            lb, ub = self.param_bounds[param]
            
            if abs(p1_val - p2_val) > 1e-14:
                if p1_val > p2_val:
                    p1_val, p2_val = p2_val, p1_val
                
                # SBX formula.
                rand = random.random()
                beta = 1.0 + (2.0 * (p1_val - lb) / (p2_val - p1_val + 1e-10))
                alpha = 2.0 - beta ** (-(eta + 1))
                
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                
                c1 = 0.5 * ((p1_val + p2_val) - beta_q * (p2_val - p1_val))
                c2 = 0.5 * ((p1_val + p2_val) + beta_q * (p2_val - p1_val))
                
                # Repair bounds.
                c1 = np.clip(c1, lb, ub)
                c2 = np.clip(c2, lb, ub)
            else:
                c1, c2 = p1_val, p2_val
            
            child1_genes[param] = c1
            child2_genes[param] = c2
        
        child1 = Chromosome(
            genes=child1_genes,
            source=GeneticSourceType.CROSSOVER,
            parent_ids=[parent1.id, parent2.id],
            generation=max(parent1.generation, parent2.generation) + 1
        )
        
        child2 = Chromosome(
            genes=child2_genes,
            source=GeneticSourceType.CROSSOVER,
            parent_ids=[parent1.id, parent2.id],
            generation=max(parent1.generation, parent2.generation) + 1
        )
        
        return child1, child2
    
    def hybrid_crossover(self, ai_parent: Chromosome, 
                         db_parent: Chromosome) -> Chromosome:
        """
        Feature 1 extension: hybrid crossover operator.

        This operator is designed for crossover between AI-generated and
        database-retrieved solutions.

        Paper formula:
        c_child = w_AI * c_AI + (1 - w_AI) * c_DB + epsilon
        where w_AI = f_AI / (f_AI + f_DB)
        epsilon ~ N(0, 0.05^2 * I)
        """
        child_genes = {}
        
        # Compute weights from fitness (paper formula: w_AI = f_AI / (f_AI + f_DB)).
        ai_fitness = sum(ai_parent.fitness_values.values()) if ai_parent.fitness_values else 0
        db_fitness = sum(db_parent.fitness_values.values()) if db_parent.fitness_values else 0
        
        total = ai_fitness + db_fitness + 1e-10
        w_ai = ai_fitness / total  # Paper's w_AI.
        
        for i, param in enumerate(["E", "G", "nu"]):
            ai_val = ai_parent.genes.get(param, 0)
            db_val = db_parent.genes.get(param, 0)
            
            # Read bounds.
            lb, ub = self.param_bounds[param]
            
            # Adaptive noise standard deviation based on the parameter range.
            sigma = 0.05 * (ub - lb)
            epsilon = np.random.normal(0, sigma)
            
            # Paper formula: c_child = w_AI * c_AI + (1 - w_AI) * c_DB + epsilon.
            child_genes[param] = w_ai * ai_val + (1 - w_ai) * db_val + epsilon
            
            # Repair bounds.
            child_genes[param] = np.clip(child_genes[param], lb, ub)
        
        child = Chromosome(
            genes=child_genes,
            source=GeneticSourceType.HYBRID,
            parent_ids=[ai_parent.id, db_parent.id],
            generation=max(ai_parent.generation, db_parent.generation) + 1
        )
        
        return child
    
    # ==================== Mutation operators ====================
    
    def polynomial_mutation(self, chromosome: Chromosome, 
                            eta: float = 20) -> Chromosome:
        """
        Polynomial mutation.

        Mutate a real-valued chromosome.
        """
        mutant = chromosome.clone()
        mutant.source = GeneticSourceType.MUTATION
        mutant.parent_ids = [chromosome.id]
        mutant.generation = chromosome.generation + 1
        
        for param in ["E", "G", "nu"]:
            if random.random() < self.mutation_rate:
                val = mutant.genes.get(param, 0)
                lb, ub = self.param_bounds[param]
                
                delta1 = (val - lb) / (ub - lb + 1e-10)
                delta2 = (ub - val) / (ub - lb + 1e-10)
                
                rand = random.random()
                mut_pow = 1.0 / (eta + 1)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val_mut = (2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))) ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val_mut = 1.0 - (2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))) ** mut_pow
                
                new_val = val + val_mut * (ub - lb)
                mutant.genes[param] = np.clip(new_val, lb, ub)
        
        return mutant
    
    def history_driven_mutation(self, chromosome: Chromosome,
                                 target_objective: str,
                                 maximize: bool = True,
                                 alpha: float = 0.3,
                                 beta: float = 0.1) -> Chromosome:
        """
        Feature 5: Enhanced History-Driven Mutation Operator (HDMO).

        Use simulation-history gradients for directed mutation and faster
        convergence toward the objective.

        Enhancements:
        1. Adapt alpha (gradient-guidance strength) from gradient confidence.
        2. Increase beta (random-exploration strength) when confidence is low.
        3. Add momentum to smooth the gradient direction.

        Paper formula:
        c' = c + alpha_eff * grad_c(p_j) + beta_eff * N(0, sigma_g^2 * I)
        
        where:
        - alpha_eff = alpha * confidence (confidence weighted)
        - beta_eff = beta * (2 - confidence) (more exploration at low confidence)
        - sigma_g: standard deviation of random exploration, decaying by generation

        In ablation mode, CHATMS_ABLATION_HISTORY_MUTATION=1 uses purely random
        mutation without gradient guidance.
        
        Args:
            chromosome: Input chromosome.
            target_objective: Objective-property name.
            maximize: Whether to maximize this objective.
            alpha: Base gradient-guidance weight.
            beta: Base random-exploration weight.
        """
        mutant = chromosome.clone()
        mutant.source = GeneticSourceType.MUTATION
        mutant.parent_ids = [chromosome.id]
        mutant.generation = chromosome.generation + 1
        
        # === Ablation check ===
        if not is_history_mutation_enabled():
            # Ablation mode uses pure random mutation without gradient guidance.
            # This isolates HDMO's contribution to optimization behavior.
            print("[SAES-HDMO] Ablation mode: using purely random mutation and skipping gradient guidance")
            for param in ["E", "G", "nu"]:
                if random.random() < self.mutation_rate:
                    val = mutant.genes.get(param, 0)
                    lb, ub = self.param_bounds[param]
                    # Purely random Gaussian perturbation: similar scale to full sigma_g without a gradient direction.
                    sigma = 0.1 * (ub - lb)
                    new_val = val + np.random.normal(0, sigma)
                    mutant.genes[param] = np.clip(new_val, lb, ub)
            return mutant
        
        # === Full mode: use gradient guidance ===
        # Retrieve gradient direction and confidence.
        gradient = self.landscape.get_gradient_direction(target_objective, maximize)
        confidence = self.landscape.get_gradient_confidence(target_objective)
        
        # Adapt weights from confidence: higher confidence relies more on the
        # gradient, while lower confidence relies more on random exploration.
        alpha_eff = alpha * confidence
        beta_eff = beta * (2.0 - confidence)  # At zero confidence, beta doubles.
        
        # Compute sigma_g with generation decay, as described in the paper.
        # Decay from 0.1 to 0.01.
        generation = chromosome.generation
        max_gen = 50  # Default maximum number of generations.
        sigma_g = 0.1 * (1 - generation / max_gen) + 0.01
        
        # Generate random exploration: N(0, sigma_g^2 * I), per the paper.
        random_component = np.random.normal(0, sigma_g, 3)  # Components correspond to E, G, and nu.
        
        params = ["E", "G", "nu"]
        for i, param in enumerate(params):
            val = mutant.genes.get(param, 0)
            lb, ub = self.param_bounds[param]
            
            # Scale the gradient to the parameter range.
            grad_normalized = gradient[i] * (ub - lb) if np.linalg.norm(gradient) > 1e-10 else 0
            
            # Updated formula: c' = c + alpha_eff * grad + beta_eff * N(0, sigma_g^2).
            new_val = val + alpha_eff * grad_normalized + beta_eff * random_component[i] * (ub - lb)
            
            # Repair bounds.
            mutant.genes[param] = np.clip(new_val, lb, ub)
        
        return mutant
    
    def adaptive_mutation(self, chromosome: Chromosome,
                          generation: int,
                          max_generations: int = 50) -> Chromosome:
        """
        Adaptive mutation rate.

        Increase mutation early for exploration and reduce it later for convergence.
        """
        # Adjust mutation strength dynamically.
        progress = generation / max_generations
        adaptive_strength = self.mutation_strength * (1 - 0.5 * progress)  # Linear decay.
        
        old_strength = self.mutation_strength
        self.mutation_strength = adaptive_strength
        
        mutant = self.polynomial_mutation(chromosome)
        
        self.mutation_strength = old_strength
        
        return mutant


class SAES:
    """
    Simulation-Aware Evolutionary Search optimizer.

    The controller coordinates simulation-history updates, candidate proposals,
    and Pareto selection inside one SAES loop.
    """
    
    def __init__(self, 
                 population_size: int = 20,
                 max_generations: int = 10,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2):
        """
        Initialize the SAES optimizer.

        Args:
            population_size: Population size.
            max_generations: Maximum generations.
            crossover_rate: Crossover rate.
            mutation_rate: Mutation rate.
        """
        self.population_size = population_size
        self.max_generations = max_generations
        
        # Core components.
        self.fitness_landscape = SimulationAwareFitnessLandscape()
        self.population_manager = HybridPopulationManager(population_size)
        self.genetic_operators = GeneticOperators(
            self.fitness_landscape,
            crossover_rate,
            mutation_rate
        )
        
        # Optimization objectives.
        self.objectives: Dict[str, Dict[str, Any]] = {}
        self._last_saes_direction = np.zeros(3)
        
        # Evolution history.
        self.evolution_history: List[Dict[str, Any]] = []
        
        # Convergence detection follows the paper's ten-generation cap and
        # three-generation stagnation window.
        self.convergence_threshold = 0.005
        self.stagnation_count = 0
        self.max_stagnation = 3
        self.min_generations_before_convergence = 3
        
        # Pareto front.
        self.pareto_front: List[Chromosome] = []
        self._feedback_recorded_ids = set()
        
        print("[SAES] Simulation-Aware Evolutionary Search optimizer initialized")
    
    def register_objectives(self, objectives: Dict[str, Dict[str, Any]]):
        """
        Register optimization objectives.
        
        Args:
            objectives: {
                "thermal_conductivity": {
                    "type": "maximize",
                    "target_value": 26,
                    "weight": 1.0
                },
                "youngs_modulus": {
                    "type": "maximize", 
                    "target_value": 3500,
                    "weight": 1.0
                }
            }
        """
        self.objectives = objectives
        self.fitness_landscape.register_objectives(objectives)
        print(f"[SAES] Registered {len(objectives)} optimization objectives")
    
    def add_ai_solution(self, params: Dict[str, float], 
                        microstructure_id: str,
                        simulation_results: Dict[str, Any] = None,
                        provenance: Dict[str, Any] = None):
        """Add an AI-generated solution."""
        verified = self.has_complete_simulation_results(simulation_results)
        chr = self.population_manager.add_ai_individual(
            genes=params,
            microstructure_id=microstructure_id,
            simulation_results=simulation_results,
            is_evaluated=verified,
            provenance=provenance,
        )
        
        if verified:
            self._record_simulation_feedback(chr, params, simulation_results)
        
        return chr
    
    def add_db_solution(self, params: Dict[str, float],
                        microstructure_id: str,
                        simulation_results: Dict[str, Any] = None,
                        provenance: Dict[str, Any] = None):
        """Add a database solution."""
        verified = self.has_complete_simulation_results(simulation_results)
        chr = self.population_manager.add_db_individual(
            genes=params,
            microstructure_id=microstructure_id,
            simulation_results=simulation_results,
            is_evaluated=verified,
            provenance=provenance,
        )
        
        if verified:
            self._record_simulation_feedback(chr, params, simulation_results)
        
        return chr
    
    @staticmethod
    def _is_finite_scalar(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return False
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    def has_complete_simulation_results(self, simulation_results: Optional[Dict[str, Any]]) -> bool:
        """Only complete, finite simulator outputs are eligible for SAES evaluation."""
        if not simulation_results or not self.objectives:
            return False
        return all(
            objective_name in simulation_results
            and self._is_finite_scalar(simulation_results[objective_name])
            for objective_name in self.objectives
        )

    def update_verified_simulation(self, chromosome: Chromosome,
                                   simulation_results: Dict[str, Any]) -> bool:
        """Merge simulator output and record it exactly once after full verification."""
        chromosome.simulation_results.update(simulation_results)
        if not self.has_complete_simulation_results(chromosome.simulation_results):
            chromosome.is_evaluated = False
            chromosome.fitness_values = {}
            return False

        chromosome.is_evaluated = True
        return self._record_simulation_feedback(
            chromosome,
            chromosome.genes,
            chromosome.simulation_results,
        )

    def _record_simulation_feedback(self, chromosome: Chromosome,
                                     input_params: Dict[str, float],
                                     simulation_results: Dict[str, Any]) -> bool:
        """Record simulation feedback in the fitness landscape."""
        if (
            chromosome.id in self._feedback_recorded_ids
            or not self.has_complete_simulation_results(simulation_results)
        ):
            return False

        target_props = {
            obj_name: obj_spec.get("target_value", 0)
            for obj_name, obj_spec in self.objectives.items()
        }
        
        errors = {}
        for obj_name in self.objectives:
            if obj_name in simulation_results and obj_name in target_props:
                actual = simulation_results[obj_name]
                target = target_props[obj_name]
                if target != 0:
                    errors[obj_name] = abs(actual - target) / abs(target)
        
        feedback = SimulationFeedback(
            chromosome_id=chromosome.id,
            input_params=input_params,
            output_properties=simulation_results,
            target_properties=target_props,
            relative_errors=errors,
            timestamp=datetime.now().isoformat(),
            success=True
        )
        
        self.fitness_landscape.record_feedback(feedback)
        self._feedback_recorded_ids.add(chromosome.id)
        return True
    
    def compute_pareto_front(self) -> List[Chromosome]:
        """
        Compute the current population's Pareto front.

        Uses NSGA-II non-dominated sorting.
        """
        population = self.population_manager.combined_population
        evaluated = [
            c for c in population
            if c.is_evaluated and self.has_complete_simulation_results(c.simulation_results)
        ]
        
        if not evaluated:
            self.pareto_front = []
            return []
        
        # Compute fitness.
        for chr in evaluated:
            self.fitness_landscape.evaluate_fitness(chr)
        
        # Non-dominated sorting.
        fronts = self._fast_non_dominated_sort(evaluated)
        
        # Compute crowding distances.
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        # Update Pareto ranks.
        for rank, front in enumerate(fronts):
            for chr in front:
                chr.pareto_rank = rank
        
        self.pareto_front = fronts[0] if fronts else []
        
        # Update the elite archive.
        self.population_manager.update_elite_archive(self.pareto_front)
        
        return self.pareto_front
    
    def _fast_non_dominated_sort(self, population: List[Chromosome]) -> List[List[Chromosome]]:
        """Perform fast non-dominated sorting."""
        fronts: List[List[Chromosome]] = [[]]
        
        domination_count = {c.id: 0 for c in population}
        dominated_solutions = {c.id: [] for c in population}
        
        for p in population:
            for q in population:
                if p.id != q.id:
                    if self._dominates(p, q):
                        dominated_solutions[p.id].append(q)
                    elif self._dominates(q, p):
                        domination_count[p.id] += 1
            
            if domination_count[p.id] == 0:
                p.pareto_rank = 0
                fronts[0].append(p)
        
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p.id]:
                    domination_count[q.id] -= 1
                    if domination_count[q.id] == 0:
                        q.pareto_rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return [f for f in fronts if f]
    
    def _dominates(self, p: Chromosome, q: Chromosome) -> bool:
        """Return whether p dominates q."""
        dominated_all = True
        better_at_least_one = False
        
        for obj_name, obj_spec in self.objectives.items():
            p_val = p.fitness_values.get(obj_name, 0)
            q_val = q.fitness_values.get(obj_name, 0)
            
            obj_type = obj_spec.get("type", "maximize")
            
            if obj_type == "maximize":
                if p_val < q_val:
                    dominated_all = False
                if p_val > q_val:
                    better_at_least_one = True
            else:  # minimize
                if p_val > q_val:
                    dominated_all = False
                if p_val < q_val:
                    better_at_least_one = True
        
        return dominated_all and better_at_least_one
    
    def _calculate_crowding_distance(self, front: List[Chromosome]):
        """Compute crowding distance."""
        if len(front) <= 2:
            for c in front:
                c.crowding_distance = float('inf')
            return
        
        for c in front:
            c.crowding_distance = 0
        
        for obj_name in self.objectives:
            # Sort by this objective.
            front.sort(key=lambda x: x.fitness_values.get(obj_name, 0))
            
            # Assign infinite distance to boundary points.
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Obtain the objective range.
            min_val = front[0].fitness_values.get(obj_name, 0)
            max_val = front[-1].fitness_values.get(obj_name, 0)
            obj_range = max_val - min_val
            
            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    prev_val = front[i-1].fitness_values.get(obj_name, 0)
                    next_val = front[i+1].fitness_values.get(obj_name, 0)
                    front[i].crowding_distance += (next_val - prev_val) / obj_range
    
    def generate_offspring(self, n_offspring: int = None) -> List[Chromosome]:
        """
        Generate offspring.

        Combines several crossover and mutation strategies.
        """
        if n_offspring is None:
            n_offspring = self.population_size
        
        offspring = []
        population = self.population_manager.combined_population
        
        if len(population) < 2:
            return offspring
        
        # Select parents.
        selected = self.genetic_operators.crowding_guided_selection(
            population, n_offspring
        )
        
        # Generate offspring.
        i = 0
        while len(offspring) < n_offspring and i < len(selected) - 1:
            parent1 = selected[i]
            parent2 = selected[i + 1]
            
            # Determine whether to use hybrid crossover.
            if (parent1.source == GeneticSourceType.AI_GENERATED and 
                parent2.source == GeneticSourceType.DATABASE):
                # Hybrid crossover.
                child = self.genetic_operators.hybrid_crossover(parent1, parent2)
                offspring.append(child)
            elif (parent1.source == GeneticSourceType.DATABASE and 
                  parent2.source == GeneticSourceType.AI_GENERATED):
                child = self.genetic_operators.hybrid_crossover(parent2, parent1)
                offspring.append(child)
            else:
                # Standard SBX crossover.
                child1, child2 = self.genetic_operators.simulated_binary_crossover(
                    parent1, parent2
                )
                offspring.extend([child1, child2])
            
            i += 2
        
        # Mutate offspring.
        mutated_offspring = []
        for child in offspring[:n_offspring]:
            if random.random() < 0.5 and self.objectives:
                # Use history-driven mutation with 50 percent probability.
                target_obj = random.choice(list(self.objectives.keys()))
                obj_type = self.objectives[target_obj].get("type", "maximize")
                mutant = self.genetic_operators.history_driven_mutation(
                    child, target_obj, maximize=(obj_type == "maximize")
                )
            else:
                # Adaptive polynomial mutation.
                mutant = self.genetic_operators.adaptive_mutation(
                    child, 
                    self.population_manager.current_generation,
                    self.max_generations
                )
            mutated_offspring.append(mutant)
        
        return mutated_offspring[:n_offspring]

    def _select_saes_anchor(self) -> Optional[Chromosome]:
        """Select an evaluated high-quality anchor for local SAES updates."""
        candidates = self.pareto_front or [
            c for c in self.population_manager.combined_population if c.is_evaluated
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda c: sum(c.fitness_values.values()) if c.fitness_values else 0.0)

    def _build_saes_guided_candidates(self, n_candidates: int = 2) -> List[Chromosome]:
        """
        Paper-aligned SAES action phase.

        Generate candidate parameters from local gradients, objective-error
        signs, momentum, and exploration noise.
        """
        anchor = self._select_saes_anchor()
        if anchor is None or not anchor.simulation_results:
            return []

        weighted_direction = np.zeros(3)
        total_weight = 0.0

        for obj_name, obj_spec in self.objectives.items():
            gradient, confidence = self.fitness_landscape.estimate_local_gradient_at(
                anchor.genes, obj_name
            )
            if gradient is None or confidence <= 0:
                continue

            target = obj_spec.get("target_value")
            actual = anchor.simulation_results.get(obj_name)
            if target is not None and actual is not None:
                direction_sign = np.sign(target - actual)
                if direction_sign == 0:
                    continue
            else:
                direction_sign = 1.0 if obj_spec.get("type", "maximize") == "maximize" else -1.0

            weight = float(obj_spec.get("weight", 1.0)) * confidence
            weighted_direction += weight * direction_sign * gradient
            total_weight += weight

        if total_weight <= 0 or np.linalg.norm(weighted_direction) <= 1e-10:
            return []

        direction = weighted_direction / np.linalg.norm(weighted_direction)
        direction = 0.7 * direction + 0.3 * self._last_saes_direction
        if np.linalg.norm(direction) > 1e-10:
            direction = direction / np.linalg.norm(direction)
        self._last_saes_direction = direction

        guided = []
        params = ["E", "G", "nu"]
        for _ in range(n_candidates):
            genes = {}
            noise = np.random.normal(0, 1.0, 3)
            for idx, param in enumerate(params):
                current = anchor.genes.get(param, 0.0)
                lb, ub = self.fitness_landscape.param_bounds[param]
                scale = max(abs(current), 1.0)
                updated = (
                    current
                    + 0.1 * direction[idx] * scale
                    + 0.05 * noise[idx] * scale
                )
                genes[param] = float(np.clip(updated, lb, ub))
            guided.append(Chromosome(
                genes=genes,
                source=GeneticSourceType.SAES_GUIDED,
                parent_ids=[anchor.id],
                generation=anchor.generation + 1
            ))
        return guided
    
    def get_next_generation_params(self, n_params: int = 3) -> List[Dict[str, float]]:
        """
        Return generation parameters for the next generation.

        This is the primary integration interface for StructureGenerator and
        returns parameters for new AI-generated microstructures.
        """
        offspring = self.generate_offspring(n_params * 2)  # Retain existing search capacity.
        offspring.extend(self._build_saes_guided_candidates(n_params))
        
        if not offspring:
            # Without sufficient historical data, return random parameters.
            return self._generate_random_params(n_params)
        
        # Select the most promising parameter combinations. Favor offspring
        # with high crowding distance for diversity.
        for child in offspring:
            estimated_fitness = self.fitness_landscape.estimate_fitness_without_simulation(child)
            child.fitness_values = estimated_fitness
        
        # Sort by diversity proxy.
        offspring.sort(key=lambda x: sum(x.fitness_values.values()), reverse=True)
        
        selected_params = []
        for child in offspring[:n_params]:
            params = {
                "E": child.genes.get("E", 3000),
                "G": child.genes.get("G", 1000),
                "nu": child.genes.get("nu", 0.3),
            }
            selected_params.append(params)
        
        return selected_params
    
    def _generate_random_params(self, n: int) -> List[Dict[str, float]]:
        """Generate random parameters for cold start."""
        params_list = []
        bounds = self.fitness_landscape.param_bounds
        
        for _ in range(n):
            params = {
                "E": random.uniform(*bounds["E"]),
                "G": random.uniform(*bounds["G"]),
                "nu": random.uniform(*bounds["nu"]),
            }
            params_list.append(params)
        
        return params_list
    
    def check_convergence(self) -> Tuple[bool, str]:
        """
        Detect convergence.

        Termination conditions, including a minimum-generation guard:
        0. Do not evaluate convergence before min_generations_before_convergence.
        1. At least one Pareto-optimal solution meets all targets within 10%.
        2. Hypervolume stagnates for k=10 generations with improvement below 0.5%.
        3. The maximum generation limit is reached (50 by default).
        
        Returns:
            (converged, reason)
        """
        generation = self.population_manager.current_generation
        
        # Condition 1: maximum generation reached.
        if generation >= self.max_generations:
            return True, f"maximum generation reached ({self.max_generations})"
        
        # Minimum-generation guard: do not converge due to stagnation or target
        # satisfaction during the first N generations.
        if generation < self.min_generations_before_convergence:
            return False, f"minimum-generation guard active ({generation}/{self.min_generations_before_convergence})"
        
        # Condition 2: hypervolume-stagnation detection.
        if len(self.evolution_history) >= 2:
            current_hv = self._compute_hypervolume()
            
            # Get the previous generation's hypervolume.
            prev_hv = self.evolution_history[-1].get("hypervolume", 0)
            
            # Compute hypervolume improvement rate.
            if prev_hv > 0:
                hv_improvement = (current_hv - prev_hv) / prev_hv
            else:
                hv_improvement = 1.0 if current_hv > 0 else 0.0
            
            # Count stagnation when hypervolume improvement is below threshold.
            if hv_improvement < self.convergence_threshold:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0  # Reset the count whenever improvement occurs.
            
            if self.stagnation_count >= self.max_stagnation:
                return True, f"hypervolume has no significant improvement for {self.max_stagnation} generations (threshold={self.convergence_threshold}, k={self.max_stagnation})"
        
        # Condition 3: all objectives are satisfied.
        if self._all_objectives_satisfied():
            return True, "all objectives are satisfied within 10 percent tolerance"
        
        return False, "continue evolution"
    
    def _compute_hypervolume(self) -> float:
        """
        Compute the current Pareto front's hypervolume.

        Hypervolume is a standard multi-objective metric that measures the
        dominated-space volume between the Pareto front and a reference point.
        
        Returns:
            Hypervolume value.
        """
        if not self.pareto_front:
            return 0.0
        
        # Collect objective values from all Pareto-optimal solutions.
        pareto_points = []
        for chr in self.pareto_front:
            obj_values = []
            for obj_name, obj_spec in self.objectives.items():
                if obj_name in chr.fitness_values:
                    value = chr.fitness_values[obj_name]
                    # Negate minimization objectives to use a maximization convention.
                    if obj_spec.get("type", "maximize") == "minimize":
                        value = -value
                    obj_values.append(value)
            if len(obj_values) == len(self.objectives):
                pareto_points.append(obj_values)
        
        if not pareto_points:
            return 0.0
        
        n_objectives = len(self.objectives)
        
        # Determine the reference point from objective-wise worst values.
        reference_point = []
        for i in range(n_objectives):
            all_values = [p[i] for p in pareto_points]
            min_val = min(all_values)
            reference_point.append(min_val - abs(min_val) * 0.1 - 1e-6)
        
        # Compute hypervolume.
        if n_objectives == 2:
            return self._compute_2d_hypervolume(pareto_points, reference_point)
        else:
            return self._compute_mc_hypervolume(pareto_points, reference_point)
    
    def _compute_2d_hypervolume(self, points: List[List[float]], 
                                 ref_point: List[float]) -> float:
        """
        Compute exact two-dimensional hypervolume.
        Sort by the first objective and accumulate rectangular areas.
        """
        if not points:
            return 0.0
        
        sorted_points = sorted(points, key=lambda x: x[0], reverse=True)
        hv = 0.0
        prev_y = ref_point[1]
        
        for p in sorted_points:
            x, y = p[0], p[1]
            if y > prev_y:
                width = x - ref_point[0]
                height = y - prev_y
                hv += width * height
                prev_y = y
        
        return hv
    
    def _compute_mc_hypervolume(self, points: List[List[float]],
                                 ref_point: List[float],
                                 n_samples: int = 10000) -> float:
        """
        Estimate high-dimensional hypervolume with Monte Carlo sampling.
        Count the fraction of randomly sampled points dominated by the Pareto front.
        """
        n_obj = len(ref_point)
        
        # Determine the ideal point.
        ideal_point = [max(p[i] for p in points) for i in range(n_obj)]
        
        # Monte Carlo sampling.
        dominated = 0
        for _ in range(n_samples):
            sample = [random.uniform(ref_point[i], ideal_point[i]) for i in range(n_obj)]
            for p in points:
                if all(p[i] >= sample[i] for i in range(n_obj)):
                    dominated += 1
                    break
        
        # Compute hypervolume.
        box_vol = 1.0
        for i in range(n_obj):
            box_vol *= (ideal_point[i] - ref_point[i])
        
        return box_vol * (dominated / n_samples)
    
    def _all_objectives_satisfied(self) -> bool:
        """Return whether all objectives are satisfied."""
        if not self.pareto_front:
            return False
        
        for chr in self.pareto_front:
            all_satisfied = True
            for obj_name, obj_spec in self.objectives.items():
                target = obj_spec.get("target_value")
                if target is None:
                    continue
                
                actual = chr.simulation_results.get(obj_name, 0)
                tolerance = abs(target) * 0.1  # Ten-percent tolerance.
                
                obj_type = obj_spec.get("type", "maximize")
                if obj_spec.get("goal") == "target_match":
                    if abs(actual - target) > tolerance:
                        all_satisfied = False
                        break
                elif obj_type == "maximize":
                    if actual < target - tolerance:
                        all_satisfied = False
                        break
                else:
                    if actual > target + tolerance:
                        all_satisfied = False
                        break
            
            if all_satisfied:
                return True
        
        return False
    
    def record_generation(self):
        """Record the current generation's state."""
        stats = self.population_manager.get_population_stats()
        stats["pareto_front_size"] = len(self.pareto_front)
        stats["timestamp"] = datetime.now().isoformat()
        
        # Compute and record hypervolume for convergence detection.
        stats["hypervolume"] = self._compute_hypervolume()
        
        # Record Pareto-front performance.
        if self.pareto_front:
            pareto_summary = []
            for chr in self.pareto_front:
                pareto_summary.append({
                    "id": chr.microstructure_id or chr.id,
                    "source": chr.source.value,
                    "properties": chr.simulation_results
                })
            stats["pareto_solutions"] = pareto_summary
        
        self.evolution_history.append(stats)
        
        # Advance to the next generation.
        self.population_manager.advance_generation()
    
    def get_best_solutions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the best solutions."""
        if not self.pareto_front:
            self.compute_pareto_front()
        
        # Sort by crowding distance.
        sorted_front = sorted(
            self.pareto_front,
            key=lambda x: x.crowding_distance,
            reverse=True
        )
        
        results = []
        for chr in sorted_front[:n]:
            results.append({
                "microstructure_id": chr.microstructure_id,
                "source": chr.source.value,
                "generation": chr.generation,
                "parameters": chr.genes,
                "properties": chr.simulation_results,
                "fitness": chr.fitness_values,
                "pareto_rank": chr.pareto_rank,
                "crowding_distance": chr.crowding_distance
            })
        
        return results
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Return an evolution-process summary."""
        return {
            "total_generations": self.population_manager.current_generation,
            "total_evaluations": len(self.fitness_landscape.feedback_history),
            "final_pareto_size": len(self.pareto_front),
            "ai_solutions_count": len(self.population_manager.ai_subpop),
            "db_solutions_count": len(self.population_manager.db_subpop),
            "elite_archive_size": len(self.population_manager.elite_archive),
            "convergence_status": self.check_convergence(),
            "objectives": self.objectives,
            "best_solutions": self.get_best_solutions(5)
        }
    
    def get_guidance_for_structure_generator(self) -> Dict[str, Any]:
        """
        Generate SAES-guided parameter suggestions for StructureGenerator.

        This is SAES's core interface to the agent system.
        """
        next_params = self.get_next_generation_params(3)
        
        guidance = {
            "type": "saes_guidance",
            "generation": self.population_manager.current_generation,
            "suggested_params": next_params,
            "pareto_front_size": len(self.pareto_front),
            "convergence_status": self.check_convergence(),
            "evolution_stats": self.population_manager.get_population_stats(),
            "recommendation": self._generate_recommendation()
        }
        
        return guidance
    
    def _generate_recommendation(self) -> str:
        """Generate optimization recommendations."""
        lines = []
        
        gen = self.population_manager.current_generation
        pareto_size = len(self.pareto_front)
        
        lines.append(f"[SAES] Evolution status, generation {gen}:")
        lines.append(f"  - Pareto front: {pareto_size} non-dominated solutions")
        
        if pareto_size > 0:
            # Analyze source distribution.
            ai_count = sum(1 for c in self.pareto_front if c.source == GeneticSourceType.AI_GENERATED)
            db_count = sum(1 for c in self.pareto_front if c.source == GeneticSourceType.DATABASE)
            
            lines.append(f"  - AI generated: {ai_count}, database: {db_count}")
            
            if ai_count > db_count * 2:
                lines.append("  -> Recommendation: AI-generated candidates perform well; continue exploring new structures.")
            elif db_count > ai_count * 2:
                lines.append("  -> Recommendation: database solutions perform better; consider broadening retrieval.")
            else:
                lines.append("  -> Recommendation: sources are balanced; continue hybrid-crossover optimization.")
        
        converged, reason = self.check_convergence()
        if converged:
            lines.append(f"  [Warning] Convergence status: {reason}")
        
        return "\n".join(lines)


# Global instance.
_global_saes: Optional[SAES] = None


def get_saes(reset: bool = False) -> SAES:
    """Return the global SAES instance."""
    global _global_saes
    if _global_saes is None or reset:
        _global_saes = SAES()
    return _global_saes


def reset_saes():
    """Reset SAES."""
    global _global_saes
    _global_saes = None
