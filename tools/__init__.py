# AutoGen microstructure design system tools package initialization.

# SAES guidance helpers - Pareto optimization guidance module.
from tools.pareto_optimizer import (
    ParetoOptimizer,
    Objective,
    ObjectiveType,
    Solution,
    get_pareto_optimizer,
    reset_pareto_optimizer,
    inject_pareto_context
)

from tools.saes_guidance import (
    SAESGuidance,
    get_saes_guidance,
    reset_saes_guidance,
    is_multi_objective_scenario
)

def clear_gpu_memory_after_simulation():
    """Best-effort GPU memory cleanup without importing simulation backends."""
    import gc

    gc.collect()
    for module_name, clear_code in (
        ("jax", "clear_caches"),
        ("torch", "cuda"),
        ("cupy", "mempool"),
    ):
        try:
            if module_name == "jax":
                import jax
                jax.clear_caches()
            elif module_name == "torch":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            elif module_name == "cupy":
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass

__all__ = [
    # Pareto Optimizer
    'ParetoOptimizer',
    'Objective', 
    'ObjectiveType',
    'Solution',
    'get_pareto_optimizer',
    'reset_pareto_optimizer',
    'inject_pareto_context',
    # SAES Guidance
    'SAESGuidance',
    'get_saes_guidance',
    'reset_saes_guidance',
    'is_multi_objective_scenario',
    # GPU memory management
    'clear_gpu_memory_after_simulation',
]
