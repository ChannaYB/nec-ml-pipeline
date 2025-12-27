
import os

# FILE PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
VALIDATION_REPORTS_DIR = os.path.join(BASE_DIR, 'data', 'validation_reports')

# Raw data files (the 3 files you uploaded)
DEMAND_FILE = 'demand.csv'
PLANTS_FILE = 'plants.csv'
GENERATION_COSTS_FILE = 'generation_costs.csv'

# Processed data files (after merging)
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

# Full paths
DEMAND_PATH = os.path.join(RAW_DATA_DIR, DEMAND_FILE)
PLANTS_PATH = os.path.join(RAW_DATA_DIR, PLANTS_FILE)
GENERATION_COSTS_PATH = os.path.join(RAW_DATA_DIR, GENERATION_COSTS_FILE)

TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, TRAIN_FILE)
TEST_PATH = os.path.join(PROCESSED_DATA_DIR, TEST_FILE)


# DATA SCHEMA

# Identifier columns
ID_COLUMNS = ['Demand ID', 'Plant ID']

# Target variable (what we're predicting)
TARGET_COLUMN = 'Cost_USD_per_MWh'

# Demand features (from demand.csv)
DEMAND_CATEGORICAL_FEATURES = ['DF_region', 'DF_daytype']
DEMAND_NUMERICAL_FEATURES = [
    'DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'DF6',
    'DF7', 'DF8', 'DF9', 'DF10', 'DF11', 'DF12'
]

# Plant features (from plants.csv)
PLANT_CATEGORICAL_FEATURES = ['Plant Type', 'Region']
PLANT_NUMERICAL_FEATURES = [
    'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6',
    'PF7', 'PF8', 'PF9', 'PF10', 'PF11', 'PF12',
    'PF13', 'PF14', 'PF15', 'PF16', 'PF17', 'PF18'
]

# All features combined
CATEGORICAL_FEATURES = DEMAND_CATEGORICAL_FEATURES + PLANT_CATEGORICAL_FEATURES
NUMERICAL_FEATURES = DEMAND_NUMERICAL_FEATURES + PLANT_NUMERICAL_FEATURES
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# All columns in merged dataset
EXPECTED_COLUMNS = ID_COLUMNS + [TARGET_COLUMN] + CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Columns that cannot have missing values
MANDATORY_COLUMNS = ID_COLUMNS + [TARGET_COLUMN]

# DATASET CHARACTERISTICS

NUM_DEMAND_SCENARIOS = 500  # Total unique demands
NUM_PLANTS = 64             # Total number of plants
PLANTS_PER_DEMAND = 64      # Each demand has 64 plant options


# VALIDATION SETTINGS

MAX_MISSING_PERCENTAGE = 5.0  # Maximum 5% missing values allowed
MIN_PLANTS_PER_DEMAND = 10    # Minimum plants needed per demand

# Group column for LOGO CV (Leave-One-Group-Out Cross Validation)
GROUP_COLUMN = 'Demand ID'

# TRAIN/TEST SPLIT SETTINGS

RANDOM_SEED = 42                    # For reproducibility
TRAIN_TEST_SPLIT_RATIO = 0.8        # 80% train, 20% test


# LOGGING SETTINGS

VERBOSE = True                      # Show detailed logs
SAVE_VALIDATION_REPORTS = True      # Save validation reports to file

TEST_SIZE = 1 - TRAIN_TEST_SPLIT_RATIO  # 0.2 (20% test)
RANDOM_STATE = RANDOM_SEED 

# CREATE DIRECTORIES

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(VALIDATION_REPORTS_DIR, exist_ok=True)