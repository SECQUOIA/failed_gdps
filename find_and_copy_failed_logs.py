"""
Script to identify models with failed solutions and copy their logs.

This script identifies two types of failures:
- Wrong Optimal: Solver reports "optimal" but objective differs from ground truth
- False Infeasible: Solver reports "infeasible" but problem is actually feasible

The script:
1. Reads the Excel results file
2. Identifies models with failed solutions (based on ground truth)
3. Finds the corresponding log folders using timestamp matching and verification
4. Also finds the BigM reformulation logs for the same models
5. Copies all logs to a structured directory for analysis
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def calculate_ground_truth(df: pd.DataFrame, obj_tolerance: float = 1e-4) -> Dict[str, float]:
    """
    Calculate ground truth objective values for each model.
    
    Ground truth is the minimum objective value among all optimal solutions
    for a given model.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results
    obj_tolerance : float
        Tolerance for objective value comparison
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping model names to ground truth objective values
    """
    # Only consider optimal solutions for ground truth
    optimal_solutions = df[df["Status"] == "optimal"]
    ground_truth = optimal_solutions.groupby("Model Name")["Objective Value"].min().to_dict()
    
    return ground_truth


def identify_wrong_solutions(
    df: pd.DataFrame,
    time_limit: float = 1800,
    obj_tolerance: float = 1e-4
) -> pd.DataFrame:
    """
    Identify models with wrong solutions.
    
    A solution is considered wrong if:
    - Status is "optimal"
    - Duration < time_limit
    - Objective value differs from ground truth by more than tolerance
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results
    time_limit : float
        Time limit in seconds
    obj_tolerance : float
        Tolerance for objective value comparison
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing only wrong solutions
    """
    # Calculate ground truth
    ground_truth = calculate_ground_truth(df, obj_tolerance)
    
    # Filter for solutions that finished within time limit with optimal status
    finished_optimal = df[
        (df["Duration (sec)"] < time_limit) & 
        (df["Status"] == "optimal")
    ].copy()
    
    # Add ground truth column
    finished_optimal["Ground Truth"] = finished_optimal["Model Name"].map(ground_truth)
    
    # Identify wrong solutions (objective differs from ground truth)
    finished_optimal["Objective Diff"] = abs(
        finished_optimal["Objective Value"] - finished_optimal["Ground Truth"]
    )
    
    wrong_solutions = finished_optimal[
        finished_optimal["Objective Diff"] > obj_tolerance
    ].copy()
    
    return wrong_solutions


def identify_false_infeasible(
    df: pd.DataFrame,
    time_limit: float = 1800
) -> pd.DataFrame:
    """
    Identify models with false infeasible status.
    
    A solution is considered false infeasible if:
    - Status is "infeasible"
    - Duration < time_limit
    - Ground truth shows the problem is actually feasible (some solver found optimal)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the results
    time_limit : float
        Time limit in seconds
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing only false infeasible solutions
    """
    # Identify feasible models (models where at least one solver found optimal)
    feasible_models = set(df[df["Status"] == "optimal"]["Model Name"].unique())
    
    # Filter for solutions that finished within time limit with infeasible status
    finished_infeasible = df[
        (df["Duration (sec)"] < time_limit) & 
        (df["Status"] == "infeasible")
    ].copy()
    
    # Identify false infeasible (reported infeasible but model is actually feasible)
    false_infeasible = finished_infeasible[
        finished_infeasible["Model Name"].isin(feasible_models)
    ].copy()
    
    # Add ground truth status
    false_infeasible["Ground Truth Status"] = "feasible"
    
    return false_infeasible


def extract_timestamp_from_folder(folder_name: str) -> Optional[datetime]:
    """
    Extract timestamp from folder name.
    
    Parameters
    ----------
    folder_name : str
        Folder name in format YYYY-MM-DD_HH-MM-SS
        
    Returns
    -------
    Optional[datetime]
        Parsed datetime or None if parsing fails
    """
    try:
        return datetime.strptime(folder_name, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def verify_folder_match(
    excel_row: pd.Series,
    json_path: Path,
    obj_tolerance: float = 1e-4,
    time_tolerance: float = 5.0
) -> bool:
    """
    Verify that the folder matches the Excel row.
    
    Compares objective value and solution time from JSON with Excel data.
    
    Parameters
    ----------
    excel_row : pd.Series
        Row from Excel file
    json_path : Path
        Path to solution_data_original.json file
    obj_tolerance : float
        Tolerance for objective value comparison
    time_tolerance : float
        Tolerance for time comparison (in seconds)
        
    Returns
    -------
    bool
        True if folder matches, False otherwise
    """
    if not json_path.exists():
        return False
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract data from JSON
        params = data.get('model_parameters', {})
        solution = data.get('solution', {})
        performance = data.get('performance', {})
        
        # Verify model parameters match
        params_match = (
            params.get('n_dimensions') == excel_row['n_dimensions'] and
            params.get('n_disjunctions') == excel_row['n_disjunctions'] and
            params.get('n_disjuncts_per_disjunction') == excel_row['n_disjuncts_per_disjunction'] and
            params.get('n_constraints_per_disjunct') == excel_row['n_constraints_per_disjunct'] and
            params.get('n_feasible_regions') == excel_row['n_feasible_regions'] and
            params.get('random_seed') == excel_row['random_seed']
        )
        
        if not params_match:
            return False
        
        # Verify objective value matches (within tolerance)
        obj_value = solution.get('objective_value')
        if obj_value is not None and pd.notna(excel_row['Objective Value']):
            obj_diff = abs(obj_value - excel_row['Objective Value'])
            if obj_diff > obj_tolerance:
                return False
        
        # Verify solution time matches (within tolerance)
        solution_time = performance.get('solution_time_seconds')
        if solution_time is not None and pd.notna(excel_row['Duration (sec)']):
            time_diff = abs(solution_time - excel_row['Duration (sec)'])
            if time_diff > time_tolerance:
                return False
        
        return True
        
    except Exception as e:
        print(f"Error reading JSON {json_path}: {str(e)}")
        return False


def find_log_folder(
    excel_row: pd.Series,
    data_dir: Path,
    obj_tolerance: float = 1e-4,
    time_tolerance_minutes: float = 60.0
) -> Optional[Path]:
    """
    Find the log folder for a given Excel row.
    
    Uses hybrid approach:
    1. Searches for folders with timestamps before or equal to Excel run time
    2. Verifies using JSON data (objective value, solution time, parameters)
    
    Parameters
    ----------
    excel_row : pd.Series
        Row from Excel file
    data_dir : Path
        Base data directory
    obj_tolerance : float
        Tolerance for objective value comparison
    time_tolerance_minutes : float
        Time window to search for folders (in minutes)
        
    Returns
    -------
    Optional[Path]
        Path to log folder or None if not found
    """
    solver = excel_row['Solver']
    subsolver = excel_row['Subsolver']
    strategy = excel_row['Strategy']
    mode = excel_row['Mode']
    run_time_str = excel_row['Run Time']
    
    # Parse Excel run time
    try:
        excel_run_time = datetime.strptime(run_time_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print(f"Error: Could not parse run time '{run_time_str}'")
        return None
    
    # Construct search directory
    if subsolver and subsolver != "None":
        solver_dir = f"{solver}_{subsolver}_{strategy}"
    else:
        solver_dir = f"{solver}_direct_{strategy}"
    
    search_dir = data_dir / solver_dir / mode
    
    if not search_dir.exists():
        print(f"Warning: Search directory does not exist: {search_dir}")
        return None
    
    # Find candidate folders
    candidates = []
    
    for folder in search_dir.iterdir():
        if not folder.is_dir():
            continue
        
        folder_time = extract_timestamp_from_folder(folder.name)
        if folder_time is None:
            continue
        
        # Only consider folders with timestamp before or equal to Excel time
        # and within the time tolerance window
        time_diff_seconds = (excel_run_time - folder_time).total_seconds()
        
        # Folder time should be before Excel time (or same) and within tolerance
        if -60 <= time_diff_seconds <= time_tolerance_minutes * 60:
            candidates.append((folder, folder_time, time_diff_seconds))
    
    if not candidates:
        print(f"Warning: No candidate folders found within time window for {excel_row['Model Name']}")
        return None
    
    # Sort by time difference (prefer closest match before Excel time)
    # Negative values mean folder is after Excel time (less preferred)
    # Positive values mean folder is before Excel time (preferred)
    candidates.sort(key=lambda x: (-x[2] if x[2] >= 0 else float('inf'), abs(x[2])))
    
    # Verify candidates
    for folder, folder_time, time_diff in candidates:
        json_path = folder / "original" / "solution_data_original.json"
        
        if verify_folder_match(excel_row, json_path, obj_tolerance):
            print(f"  Found matching folder: {folder.name} (time diff: {time_diff:.1f}s)")
            return folder
    
    # If no verified match, return the closest folder with a warning
    print(f"Warning: No verified match found for {excel_row['Model Name']}, using closest folder")
    return candidates[0][0] if candidates else None


def copy_logs_to_destination(
    source_folder: Path,
    dest_base: Path,
    failed_strategy: str,
    model_identifier: str,
    strategy_name: str
) -> Path:
    """
    Copy log folder to destination with organized structure.
    
    Parameters
    ----------
    source_folder : Path
        Source folder containing logs
    dest_base : Path
        Base destination directory
    failed_strategy : str
        The strategy that failed (e.g., "hull", "hull_exact")
    model_identifier : str
        Identifier for the model
    strategy_name : str
        Name of the strategy being copied (e.g., "failed_strategy", "bigm")
        
    Returns
    -------
    Path
        Path to the copied folder
    """
    # Create destination path: dest_base/failed_{strategy}/model_identifier/strategy_name/timestamp/
    timestamp = source_folder.name
    strategy_folder = f"failed_{failed_strategy.replace('gdp.', '')}"
    dest_path = dest_base / strategy_folder / model_identifier / strategy_name / timestamp
    
    # Copy the entire folder
    if dest_path.exists():
        shutil.rmtree(dest_path)
    
    shutil.copytree(source_folder, dest_path)
    
    return dest_path


def main():
    """Main function to identify and copy failed model logs."""
    
    # Configuration
    data_dir = Path("/home/sergey-gusev/Desktop/research/projects/random_quadratic/data")
    excel_path = data_dir / "results.xlsx"
    output_base = data_dir / "failed_models_logs"
    
    time_limit = 1800  # seconds
    obj_tolerance = 1e-4
    time_tolerance_minutes = 60.0  # minutes
    
    print("="*80)
    print("IDENTIFYING AND COPYING FAILED MODEL LOGS")
    print("="*80)
    
    # Read Excel file
    print(f"\nReading results from: {excel_path}")
    df = pd.read_excel(excel_path)
    
    # Filter to original problems only
    df = df[df["Problem Type"] == "Original"]
    
    print(f"Total entries: {len(df)}")
    print(f"Unique models: {df['Model Name'].nunique()}")
    print(f"Strategies: {df['Strategy'].unique().tolist()}")
    
    # Identify wrong optimal solutions
    print(f"\n{'='*80}")
    print("IDENTIFYING WRONG OPTIMAL SOLUTIONS")
    print(f"{'='*80}")
    print(f"Tolerance: {obj_tolerance}")
    wrong_solutions = identify_wrong_solutions(df, time_limit, obj_tolerance)
    
    print(f"\nFound {len(wrong_solutions)} wrong optimal solutions:")
    if len(wrong_solutions) > 0:
        print(wrong_solutions[['Model Name', 'Strategy', 'Objective Value', 'Ground Truth', 'Objective Diff']])
    
    # Identify false infeasible solutions
    print(f"\n{'='*80}")
    print("IDENTIFYING FALSE INFEASIBLE SOLUTIONS")
    print(f"{'='*80}")
    false_infeasible = identify_false_infeasible(df, time_limit)
    
    print(f"\nFound {len(false_infeasible)} false infeasible solutions:")
    if len(false_infeasible) > 0:
        print(false_infeasible[['Model Name', 'Strategy', 'Status', 'Ground Truth Status']])
    
    # Combine all failed solutions
    all_failures = pd.concat([wrong_solutions, false_infeasible], ignore_index=True)
    
    if len(all_failures) == 0:
        print("\nNo failed solutions found. Exiting.")
        return
    
    print(f"\n{'='*80}")
    print(f"TOTAL FAILURES TO PROCESS: {len(all_failures)}")
    print(f"  - Wrong optimal solutions: {len(wrong_solutions)}")
    print(f"  - False infeasible solutions: {len(false_infeasible)}")
    print(f"{'='*80}")
    
    # Create output directory
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Process each failed solution
    results_summary = []
    errors = []
    
    for idx, row in all_failures.iterrows():
        model_name = row['Model Name']
        strategy = row['Strategy']
        status = row['Status']
        
        print(f"\n{'='*80}")
        print(f"Processing: {model_name}")
        print(f"  Failed Strategy: {strategy}")
        print(f"  Status: {status}")
        
        # Display different info based on failure type
        if status == "optimal":
            # Wrong optimal solution
            print(f"  Failure Type: Wrong optimal solution")
            print(f"  Objective: {row['Objective Value']:.6f}")
            print(f"  Ground Truth: {row['Ground Truth']:.6f}")
            print(f"  Difference: {row['Objective Diff']:.6f}")
        elif status == "infeasible":
            # False infeasible
            print(f"  Failure Type: False infeasible")
            print(f"  Reported Status: infeasible")
            print(f"  Ground Truth Status: {row['Ground Truth Status']}")
        
        print(f"{'='*80}")
        
        # Find log folder for failed strategy
        print(f"\nSearching for logs of failed strategy ({strategy})...")
        failed_folder = find_log_folder(row, data_dir, obj_tolerance, time_tolerance_minutes)
        
        if failed_folder is None:
            error_msg = f"ERROR: Could not find log folder for {model_name} with strategy {strategy}"
            print(error_msg)
            errors.append(error_msg)
            continue
        
        print(f"  Found: {failed_folder}")
        
        # Find BigM reformulation for the same model
        print(f"\nSearching for BigM reformulation logs...")
        bigm_row = df[
            (df['Model Name'] == model_name) & 
            (df['Strategy'] == 'gdp.bigm')
        ]
        
        if len(bigm_row) == 0:
            error_msg = f"ERROR: Could not find BigM reformulation for {model_name}"
            print(error_msg)
            errors.append(error_msg)
            continue
        
        bigm_row = bigm_row.iloc[0]
        bigm_folder = find_log_folder(bigm_row, data_dir, obj_tolerance, time_tolerance_minutes)
        
        if bigm_folder is None:
            error_msg = f"ERROR: Could not find log folder for {model_name} with BigM strategy"
            print(error_msg)
            errors.append(error_msg)
            continue
        
        print(f"  Found: {bigm_folder}")
        
        # Create model identifier (clean model name for folder)
        # Remove .pkl extension and clean up
        model_id = model_name.replace('.pkl', '').replace('model_', '')
        
        # Copy failed strategy logs
        print(f"\nCopying failed strategy logs...")
        strategy_name = strategy.replace('gdp.', '')
        failed_dest = copy_logs_to_destination(
            failed_folder,
            output_base,
            strategy,  # Pass the original strategy name
            model_id,
            strategy_name  # Use actual strategy name (e.g., "hull_exact")
        )
        print(f"  Copied to: {failed_dest}")
        
        # Copy BigM logs
        print(f"\nCopying BigM logs...")
        bigm_dest = copy_logs_to_destination(
            bigm_folder,
            output_base,
            strategy,  # Group BigM under the same failed strategy
            model_id,
            "bigm"
        )
        print(f"  Copied to: {bigm_dest}")
        
        # Save summary info
        summary_entry = {
            'model_name': model_name,
            'model_id': model_id,
            'failed_strategy': strategy,
            'failure_type': 'wrong_optimal' if status == 'optimal' else 'false_infeasible',
            'status': status,
            'failed_folder': str(failed_folder),
            'failed_dest': str(failed_dest),
            'bigm_folder': str(bigm_folder),
            'bigm_dest': str(bigm_dest),
            'bigm_status': bigm_row['Status'],
        }
        
        # Add type-specific info
        if status == 'optimal':
            summary_entry['failed_objective'] = row['Objective Value']
            summary_entry['ground_truth'] = row['Ground Truth']
            summary_entry['objective_diff'] = row['Objective Diff']
            summary_entry['bigm_objective'] = bigm_row['Objective Value']
        else:  # infeasible
            summary_entry['ground_truth_status'] = row['Ground Truth Status']
            summary_entry['bigm_objective'] = bigm_row.get('Objective Value', 'N/A')
        
        results_summary.append(summary_entry)
    
    # Save summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    summary_file = output_base / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Failed Models Log Copy Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total failures found: {len(all_failures)}\n")
        f.write(f"  - Wrong optimal solutions: {len(wrong_solutions)}\n")
        f.write(f"  - False infeasible solutions: {len(false_infeasible)}\n")
        f.write(f"Successfully copied: {len(results_summary)}\n")
        f.write(f"Errors: {len(errors)}\n\n")
        
        f.write("Organization: Logs are grouped by failed strategy\n")
        f.write("Structure: failed_{strategy}/model_id/{strategy_name}|bigm/timestamp/\n")
        f.write("  - strategy_name is the actual strategy (e.g., hull, hull_exact, binary_multiplication)\n\n")
        
        f.write("Failure Types:\n")
        f.write("  - wrong_optimal: Solver reported optimal but objective differs from ground truth\n")
        f.write("  - false_infeasible: Solver reported infeasible but problem is actually feasible\n\n")
        
        if results_summary:
            f.write("Successfully Copied Models:\n")
            f.write("-"*80 + "\n")
            
            # Group by strategy for better readability
            by_strategy = {}
            for result in results_summary:
                strat = result['failed_strategy']
                if strat not in by_strategy:
                    by_strategy[strat] = []
                by_strategy[strat].append(result)
            
            for strategy, models in sorted(by_strategy.items()):
                f.write(f"\n### Failed Strategy: {strategy} ({len(models)} models) ###\n\n")
                for result in models:
                    f.write(f"Model: {result['model_name']}\n")
                    f.write(f"  Model ID: {result['model_id']}\n")
                    f.write(f"  Failure Type: {result['failure_type']}\n")
                    
                    if result['failure_type'] == 'wrong_optimal':
                        f.write(f"  Failed Objective: {result['failed_objective']:.6f}\n")
                        f.write(f"  Ground Truth: {result['ground_truth']:.6f}\n")
                        f.write(f"  Difference: {result['objective_diff']:.6f}\n")
                        f.write(f"  BigM Objective: {result['bigm_objective']:.6f}\n")
                    else:  # false_infeasible
                        f.write(f"  Failed Status: infeasible\n")
                        f.write(f"  Ground Truth Status: {result['ground_truth_status']}\n")
                        f.write(f"  BigM Status: {result['bigm_status']}\n")
                        f.write(f"  BigM Objective: {result['bigm_objective']}\n")
                    
                    strategy_folder = f"failed_{strategy.replace('gdp.', '')}"
                    f.write(f"  Location: {output_base / strategy_folder / result['model_id']}\n")
                    f.write("\n")
        
        if errors:
            f.write("\n\nErrors:\n")
            f.write("-"*80 + "\n")
            for error in errors:
                f.write(f"{error}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"\nLogs copied to: {output_base}")
    print(f"Successfully processed: {len(results_summary)}/{len(all_failures)} models")
    print(f"  - Wrong optimal solutions: {sum(1 for r in results_summary if r['failure_type'] == 'wrong_optimal')}")
    print(f"  - False infeasible solutions: {sum(1 for r in results_summary if r['failure_type'] == 'false_infeasible')}")
    
    if errors:
        print(f"\n{'='*80}")
        print("WARNINGS - LOGS NOT FOUND:")
        print(f"{'='*80}")
        for error in errors:
            print(f"  {error}")
        print(f"\nTotal models without logs: {len(errors)}")
        print(f"See {summary_file} for full details.")
    
    print(f"\n{'='*80}")
    print("COMPLETED")
    print(f"{'='*80}")
    print(f"Successfully processed: {len(results_summary)}/{len(all_failures)} models")
    print(f"  - Wrong optimal solutions: {sum(1 for r in results_summary if r['failure_type'] == 'wrong_optimal')}")
    print(f"  - False infeasible solutions: {sum(1 for r in results_summary if r['failure_type'] == 'false_infeasible')}")
    if errors:
        print(f"Could not find logs for: {len(errors)} models")
    print(f"Results saved to: {output_base}")


if __name__ == "__main__":
    main()

