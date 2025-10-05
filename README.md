# Failed Models Log Analysis Tool

This directory contains a utility script for identifying models with incorrect solutions and copying their logs for detailed analysis.

## Purpose

After running optimization experiments, some models may produce incorrect results. This script identifies two types of failures:

1. **Wrong Optimal Solutions**: Solver reports "optimal" but the objective value differs from ground truth
2. **False Infeasible Solutions**: Solver reports "infeasible" but the problem is actually feasible

The script:
1. Reads the Excel results file to identify failed models
2. Calculates ground truth (minimum objective value and feasibility status)
3. Identifies both wrong optimal solutions and false infeasible reports
4. Finds the corresponding log folders using timestamp matching and verification
5. Copies both the failed strategy logs and BigM reformulation logs for comparison

## Usage

### Prerequisites

- Python 3.x
- Required packages: `pandas`, `numpy`, `openpyxl` (for Excel support)

### Steps

1. **After running your optimization experiments**, copy the `find_and_copy_failed_logs.py` script to your project's main directory:
   ```bash
   cp find_and_copy_failed_logs.py /path_to_project/exact_quadratic_hull/random_quadratic/random_quadratic/
   ```

2. **Run the script from your project directory**:
   ```bash
   python find_and_copy_failed_logs.py
   ```


## Output Structure

The script creates an organized directory structure with failed model logs:

```
data/failed_models_logs/
├── failed_{strategy}/           # e.g., failed_hull, failed_hull_exact
│   ├── {model_id}/              # e.g., n5_d3_k2_c2_f3_s42
│   │   ├── {strategy_name}/     # e.g., hull_exact
│   │   │   └── {timestamp}/     # e.g., 2024-10-01_14-30-15
│   │   │       ├── original/
│   │   │       ├── bigm/
│   │   │       └── ...
│   │   └── bigm/                # BigM comparison logs
│   │       └── {timestamp}/
│   │           └── ...
└── summary.txt                  # Detailed summary of all failures
```


## What the Script Does

### 1. Ground Truth Calculation
- **Objective Ground Truth**: Finds the minimum objective value among all optimal solutions for each model
- **Feasibility Ground Truth**: If any solver found an optimal solution, the problem is feasible

### 2. Failure Identification

The script identifies two types of failures:

#### A. Wrong Optimal Solutions
A solution is flagged as wrong optimal if:
- Status is "optimal"
- Duration < time_limit (finished within time)
- Objective value differs from ground truth by more than `obj_tolerance`

#### B. False Infeasible Solutions
A solution is flagged as false infeasible if:
- Status is "infeasible"
- Duration < time_limit (finished within time)
- Ground truth shows the problem is actually feasible (at least one solver found optimal)

### 3. Log Folder Matching
Uses a hybrid approach:
- Searches folders with timestamps near the Excel run time
- Verifies matches using JSON data (objective value, solution time, model parameters)

### 4. Log Copying
For each failed model:
- Copies the failed strategy's log folder
- Copies the BigM reformulation's log folder for comparison
- Organizes everything in a structured directory

## Output Files

### summary.txt
Contains:
- Total failures found (broken down by type)
  - Wrong optimal solutions
  - False infeasible solutions
- Successfully copied model counts
- Errors encountered
- Detailed information for each failed model:
  - Model name and ID
  - Failure type (wrong_optimal or false_infeasible)
  - For wrong optimal: Failed objective vs. ground truth, difference
  - For false infeasible: Failed status vs. ground truth status
  - BigM comparison data
  - File locations

## Notes

- The script preserves the original timestamp folders for traceability
- All model parameters are verified to ensure correct log matching
- If multiple candidate folders exist, the script uses verification to select the correct one
- Warnings are printed for any models where logs cannot be found

## Troubleshooting

**"No candidate folders found within time window"**
- Increase `time_tolerance_minutes` parameter
- Check that log folders use the correct timestamp format: `YYYY-MM-DD_HH-MM-SS`

**"Could not find BigM reformulation"**
- Ensure BigM experiments were run for all models
- Check that the Strategy column in Excel contains `'gdp.bigm'`

**"No verified match found"**
- The script will use the closest folder by timestamp
- Verify that model parameters in Excel match the JSON files
- Check `obj_tolerance` and `time_tolerance` settings
