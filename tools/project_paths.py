"""Centralized project-relative paths used across the AutoMS workflow."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
WORKSHOP_DIR = DATA_DIR / "workshop"
OBJ_DIR = DATA_DIR / "obj"
RESULTS_DIR = DATA_DIR / "results"
PICTURE_DIR = DATA_DIR / "picture"
GENERATED_DIR = DATA_DIR / "generated"
PETL_RESULTS_DIR = GENERATED_DIR / "petl_results"
DATABASE_DIR = DATA_DIR / "database"
DATABASE_CSV_PATH = DATABASE_DIR / "converted_properties.csv"
DATABASE_OBJ_DIR = DATABASE_DIR / "obj_files"
PLASTICITY_EXAMPLES_DIR = DATA_DIR / "plasticity_examples"

EXTERNAL_DIR = PROJECT_ROOT / "external"
MIND_DIR = EXTERNAL_DIR / "mind"
MIND_GENERATE_ALL_SCRIPT = MIND_DIR / "generate_all.py"
MIND_VALIDATION_DIR = MIND_DIR / "nfd_vali"
MIND_NETWORK_PATH = MIND_VALIDATION_DIR / "ckpts" / "dataset2_ae4_010117.pkl"
MIND_GEN_FROM_TRI_SCRIPT = MIND_VALIDATION_DIR / "gen_from_tri-exp-9.sh"
TRIPLANE_WORK_DIR = EXTERNAL_DIR / "triplane_inverse_design" / "ms" / "edm"
