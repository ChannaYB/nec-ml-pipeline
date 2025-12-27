# NEC ML Pipeline - Power Plant Selection System

---

## Table of Contents

- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Quick Start](#quick-start)
- [Installation Guide](#installation-guide)
- [Project Structure](#project-structure)
- [Member 1: Data Pipeline](#member-1-data-pipeline)
- [Usage Guide](#usage-guide)
- [Testing](#testing)
- [Documentation](#documentation)

---

##  Project Overview

This project develops an **end-to-end machine learning pipeline** for the National Energy Consortium (NEC) to predict power generation costs and select optimal power plants for different demand scenarios.

### Business Problem

NEC needs to select the most cost-effective power plant for each energy demand scenario from 64 available plants. The system must:
- Predict generation costs for each (Demand, Plant) combination
- Select the plant with minimum predicted cost
- Provide robust evaluation metrics (Selection Error Rate & RMSE)

### Dataset

- **Demand Scenarios:** 500 unique scenarios with 12 numerical features + 2 categorical
- **Power Plants:** 64 plants with 18 numerical features + 2 categorical  
- **Generation Costs:** 32,000 cost records (500 demands × 64 plants)
- **Target Variable:** Cost_USD_per_MWh

**After cleaning:** 415 complete demands (26,560 rows total)

---

##  Team Members

| **Member 1** | Data Pipeline & Validation
| **Member 2** | Preprocessing & Feature Engineering 
| **Member 3** | Model Training & Custom Scorer
| **Member 4** | Evaluation & Cross-Validation
| **Member 5** | Hyperparameter Tuning & Integration

---

##  Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Installation (5 minutes)
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd nec-ml-pipeline

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# 4. Install requirements
pip install -r requirements.txt

# 5. Run Member 1 pipeline
python -m src.data_ingestion
```

**Expected Output:**
```
DATA INGESTION COMPLETE
Train: (21248, 37)
Test: (5312, 37)
```

---

##  Installation Guide

### Step 1: Create Virtual Environment

**What is a Virtual Environment?**  
A virtual environment is an isolated Python environment that keeps project dependencies separate from your system Python.

**Why use it?**
-  Prevents package conflicts
-  Easy to reproduce on other machines
-  Professional best practice

**Create venv:**
```bash
# Navigate to project folder
cd nec-ml-pipeline

# Create virtual environment named 'venv'
python -m venv venv
```

This creates a `venv/` folder containing:
- Python interpreter
- pip package manager
- Installed packages (isolated from system)

---

### Step 2: Activate Virtual Environment

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

If you get an error about execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

**On Windows (Git Bash):**
```bash
source venv/Scripts/activate
```

**On Mac/Linux:**
```bash
source venv/bin/activate
```

**Verify activation:**
- Your terminal should show `(venv)` at the start
- Run: `which python` (Mac/Linux) or `where python` (Windows)
- Should show path inside `venv/` folder

---

### Step 3: Install Dependencies
```bash
# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Packages installed:**
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `pytest` - Testing

**Verify installation:**
```bash
pip list
```

---

### Step 4: Deactivate (When Done)
```bash
deactivate
```

---

##  Project Structure
```
nec-ml-pipeline/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── data_ingestion.py        # Data loading & merging (Member 1)
│   ├── data_validation.py       # Data quality checks (Member 1)
│   ├── data_splitting.py        # LOGO CV utilities (Member 1)
│   └── eda_utils.py             # Exploratory analysis (Member 1)
│
├── tests/                        # Unit tests
│   └── test_data_ingestion.py  # Data pipeline tests
│
├── data/
│   ├── raw/                     # Original CSV files
│   │   ├── demand.csv          # 500 demand scenarios
│   │   ├── plants.csv          # 64 power plants
│   │   └── generation_costs.csv # 32,000 cost records
│   │
│   ├── processed/               # Cleaned & split data
│   │   ├── train.csv           # Training set (21,248 rows)
│   │   └── test.csv            # Testing set (5,312 rows)
│   │
│   └── validation_reports/      # Data quality reports
│
├── docs/                        # Documentation
│   └── member1_guide.md        # Member 1 detailed guide
│
├── notebooks/                   # Jupyter notebooks (optional)
│
├── venv/                        # Virtual environment (not in Git)
│
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔧 Member 1: Data Pipeline & Validation


### Overview

Member 1 is responsible for the **foundation** of the entire ML pipeline:
- Loading and merging 3 separate data files
- Data quality validation
- Handling missing values
- Creating grouped train/test splits
- Ensuring LOGO CV compatibility

### Key Accomplishments

 **Data Ingestion**
- Loads 3 CSV files: demand.csv, plants.csv, generation_costs.csv
- Merges into unified dataset (37 columns total)
- Combines demand features + plant features + target costs

 **Data Cleaning**
- Identified and removed 96 rows with missing target values
- Removed 85 demands (17%) with incomplete plant sets
- Ensures data quality for downstream modeling

 **Grouped Train/Test Split**
- Train: 332 demands (21,248 rows)
- Test: 83 demands (5,312 rows)
- Split ratio: 80/20
- **Critical:** All 64 plants kept together per demand (required for LOGO CV)
- Zero demand leakage between train and test

 **Data Validation**
- Zero missing values in target variable
- All demands have exactly 64 plants
- Schema validation passed
- Quality reports generated

### Data Quality Metrics

| Metric | Train | Test |
|--------|-------|------|
| Rows | 21,248 | 5,312 |
| Demands | 332 | 83 |
| Plants per Demand | 64 | 64 |
| Missing Costs | 0 | 0 |
| Columns | 37 | 37 |

### Files Created

- `src/config.py` - Central configuration
- `src/data_ingestion.py` - Data loading & merging
- `src/data_validation.py` - Quality checks
- `src/data_splitting.py` - LOGO CV utilities
- `src/eda_utils.py` - Exploratory analysis
- `tests/test_data_ingestion.py` - Unit tests
- `data/processed/train.csv` - Training data
- `data/processed/test.csv` - Testing data

### Critical Design Decisions

**1. Grouped Splitting by Demand ID**
- **Why:** Required for Leave-One-Group-Out (LOGO) cross-validation
- **Impact:** Prevents data leakage, ensures valid evaluation
- **Trade-off:** Cannot use simple random split

**2. Removing Incomplete Demands**
- **Issue:** 85 demands had missing cost values (incomplete plant sets)
- **Decision:** Remove entirely rather than impute
- **Rationale:** Maintains LOGO CV integrity (each demand needs all 64 plants)
- **Impact:** Lost 17% of demands, but ensured data quality

**3. Merged Data Architecture**
- **Approach:** Costs → Demand → Plants (left joins)
- **Benefit:** Preserves all cost records, adds features systematically
- **Result:** 37 columns (2 IDs + 1 target + 34 features)

---

##  Usage Guide

### For Team Members

**Member 2 (Preprocessing):**
```python
from src.data_ingestion import create_train_test
from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# Load clean data
train_df, test_df = create_train_test(verbose=False)

# Separate features
X_train = train_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
y_train = train_df['Cost_USD_per_MWh']

# Your preprocessing code here...
```

**Member 3 (Models):**
```python
from src.config import TARGET_COLUMN, GROUP_COLUMN

# After Member 2's preprocessing
# Your model training code here...
```

**Member 4 (Evaluation):**
```python
from src.data_splitting import get_logo_splits

# Create LOGO CV folds
splits = get_logo_splits(train_df, n_splits=5)

for fold, (train_idx, val_idx) in enumerate(splits):
    train_fold = train_df.iloc[train_idx]
    val_fold = train_df.iloc[val_idx]
    # Your evaluation code here...
```

### Running the Pipeline

**Option 1: Run data ingestion directly**
```bash
python -m src.data_ingestion
```

**Option 2: Use in Python script**
```python
from src.data_ingestion import create_train_test

# Load data
train_df, test_df = create_train_test(verbose=True)

print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")
```

**Option 3: Load saved files**
```python
import pandas as pd

train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')
```

---

##  Testing

### Run All Tests
```bash
# Activate virtual environment first
venv\Scripts\activate

# Run unit tests
python tests/test_data_ingestion.py
```

### Run Individual Modules
```bash
# Test data ingestion
python -m src.data_ingestion

# Test data validation
python -m src.data_validation

# Test data splitting
python -m src.data_splitting

# Test EDA utilities
python -m src.eda_utils
```

### Expected Test Results

All tests should pass with:
-  Data loaded successfully
-  No missing target values
-  All demands have 64 plants
-  No train/test leakage
-  Correct data shapes

---

##  Documentation

### Main Documents

- **README.md** - Project overview and setup
- **requirements.txt** - Python dependencies
- **data/validation_reports/** - Data quality reports

### Code Documentation

All Python modules include:
- Module-level docstrings
- Function-level docstrings
- Inline comments
- Type hints where appropriate

---

##  Key Configuration

**Important variables in `src/config.py`:**
```python
# Data files
DEMAND_PATH = 'data/raw/demand.csv'
PLANTS_PATH = 'data/raw/plants.csv'
GENERATION_COSTS_PATH = 'data/raw/generation_costs.csv'

# Target and grouping
TARGET_COLUMN = 'Cost_USD_per_MWh'
GROUP_COLUMN = 'Demand ID'

# Split settings
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% train, 20% test
RANDOM_SEED = 42              # For reproducibility

# Features
CATEGORICAL_FEATURES = ['DF_region', 'DF_daytype', 'Plant Type', 'Region']
NUMERICAL_FEATURES = ['DF1', ..., 'DF12', 'PF1', ..., 'PF18']  # 30 features
```

---

##  Important Notes

### Data Quality

**Missing Values:**
- Original dataset had 96 missing cost values (0.3%)
- Affected 85 demands (17% of total)
- All removed to maintain data quality
- Final dataset has **zero missing values** in target

**Demand Structure:**
- Each demand **must** have exactly 64 plants
- Required for LOGO cross-validation
- Incomplete demands were removed entirely

### Virtual Environment

**Always activate venv before running code:**
```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**Never commit venv to Git:**
- Already in `.gitignore`
- Each team member creates their own venv

### Git Workflow
```bash
# Daily workflow
git pull origin main
git checkout -b feature/your-feature
# Make changes
git add .
git commit -m "descriptive message"
git push origin feature/your-feature
# Create pull request on GitHub
```

---

##  Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Solution:**
```bash
# Make sure you're in project root
cd nec-ml-pipeline

# Run as module
python -m src.data_ingestion
```

### "FileNotFoundError: demand.csv not found"

**Solution:**
```bash
# Check files are in correct location
ls data/raw/

# Should show: demand.csv, plants.csv, generation_costs.csv
```

### Import errors

**Solution:**
```bash
# Make sure venv is activated
venv\Scripts\activate

# Reinstall requirements
pip install -r requirements.txt
```

---


##  Assessment Alignment

This project addresses:

- **ILO1:** ML algorithm selection and justification
- **ILO2:** Complex data analysis and pattern identification
- **ILO3:** Predictive model development
- **ILO4:** Model performance evaluation
- **ILO5:** End-to-end ML pipeline design

---
