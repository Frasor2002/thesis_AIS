import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Union

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "log", "xil_plot")

STRATEGIES = ["simplicity", "simplicity_class", "random"]
STRAT_COLORS = {
    "simplicity": "C1", 
    "simplicity_class": "mediumorchid", 
    "random": "mediumturquoise"
}

METRICS = ["conf_sampled", "attr_on_conf", "accuracy"]
Y_LABELS = ["Confounded samples sampled", "% Attr on confounder", "Accuracy"]

# Ensure log directory exists
os.makedirs(LOG_DIR, exist_ok=True)


def get_scenario_name(bs: List[float]) -> str:
    """Generates a descriptive name for the bias ratio list."""
    if not bs:
        return "empty"
    
    last_val = bs[-1]
    count = sum(1 for x in reversed(bs) if x == last_val)
            
    if count == len(bs):
        return f"all {last_val}"
        
    return f"last {count} {last_val}"

def _extract_queries_and_values(results_data: Union[Dict, List[Dict]], strat: str, value_key: str):
    """Helper to extract and format queries and values from the results dictionary or list of dictionaries."""
    if isinstance(results_data, list):
        all_values = []
        queries = None
        for run in results_data:
            log = run.get(strat, {})
            queries_raw = log.get("query", [])
            values_raw = log.get(value_key, [])
            if queries_raw and values_raw:
                queries = queries_raw[0] if isinstance(queries_raw[0], list) else queries_raw
                all_values.append(values_raw)
        
        if not all_values:
            return None, None
        return queries, np.array(all_values)
        
    else:
        log = results_data.get(strat, {})
        queries_raw = log.get("query", [])
        values_raw = log.get(value_key, []) 
        
        if not queries_raw or not values_raw:
            return None, None
            
        # Standardize queries to 1D
        queries = queries_raw[0] if isinstance(queries_raw[0], list) else queries_raw
        values = np.array(values_raw)
        
        return queries, values

def _setup_and_save_plot(dataset_name: str, scenario_name: str, measure: str, value_key: str, plot_type: str):
    """Applies standardized formatting and saves the matplotlib figure."""
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Sampling comparison {dataset_name} {scenario_name}", fontsize=20)
    plt.xlabel("Explained samples", fontsize=15)
    plt.ylabel(measure, fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.tight_layout()
    
    safe_scenario = scenario_name.replace(" ", "_")
    filename = f"plot_{plot_type}_{dataset_name}_{safe_scenario}_{value_key}.pdf"
    save_path = os.path.join(LOG_DIR, filename)
    
    plt.savefig(save_path, format='pdf')
    plt.close() 
    print(f"Saved {plot_type} plot to {save_path}")


# --- Core Plotting & Table Functions ---

def save_metric_table_pdf(results_dict: Union[Dict, List[Dict]], value_key: str, measure: str, dataset_name: str, scenario_name: str):
    """Renders metrics as a table using Matplotlib and saves it as a PDF."""
    data_dict = {}
    
    for strat in STRATEGIES:
        queries, values = _extract_queries_and_values(results_dict, strat, value_key)
        if queries is None:
            continue
            
        # Average multiple runs if data is 2D
        if values.ndim == 2:
            values = np.mean(values, axis=0)
            
        for q, v in zip(queries, values):
            if q not in data_dict:
                data_dict[q] = {s: "-" for s in STRATEGIES} 
            data_dict[q][strat] = round(v, 4) if isinstance(v, (float, np.floating)) else v

    sorted_queries = sorted(data_dict.keys())
    cell_text = [[q] + [data_dict[q][strat] for strat in STRATEGIES] for q in sorted_queries]
    columns = ["Explained Samples"] + STRATEGIES

    fig_height = max(len(cell_text) * 0.3 + 1, 2) 
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    ax.axis('tight')
    ax.axis('off')
    plt.title(f"{measure} (Average) - {dataset_name} {scenario_name}", pad=20, fontsize=14, weight='bold')

    table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5) 

    # Style header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2') 

    safe_scenario = scenario_name.replace(" ", "_")
    filename = f"table_{dataset_name}_{safe_scenario}_{value_key}.pdf"
    save_path = os.path.join(LOG_DIR, filename)
    
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close() 
    print(f"Saved PDF table to {save_path}")


def plot_avg_std(results_dict: Union[Dict, List[Dict]], value_key: str, measure: str, dataset_name: str, scenario_name: str):
    """Plots the average line and standard deviation shading for multiple runs."""
    plt.figure(figsize=(10, 6))
    
    for strat in STRATEGIES:
        queries, values = _extract_queries_and_values(results_dict, strat, value_key)
        if queries is None:
            continue
            
        if values.ndim == 2:
            mean_vals = np.mean(values, axis=0)
            std_vals = np.std(values, axis=0)
        else:
            mean_vals = values
            std_vals = np.zeros_like(values)
            
        plt.plot(queries, mean_vals, marker='', linestyle='-', color=STRAT_COLORS[strat], label=strat)
        plt.fill_between(queries, mean_vals - std_vals, mean_vals + std_vals, color=STRAT_COLORS[strat], alpha=0.2)
        
    _setup_and_save_plot(dataset_name, scenario_name, measure, value_key, "avg_std")


def plot_single_run(results_dict: Union[Dict, List[Dict]], value_key: str, measure: str, dataset_name: str, scenario_name: str):
    """Plots a single standalone run. If multiple runs are provided, it plots the first run."""
    plt.figure(figsize=(10, 6))
    
    for strat in STRATEGIES:    
        queries, values = _extract_queries_and_values(results_dict, strat, value_key)
        if queries is None:
            continue
            
        # Extract the first run if data is 2D
        if values.ndim == 2:
            values = values[0]
            
        plt.plot(queries, values, marker='', linestyle='-', color=STRAT_COLORS[strat], label=strat)
  
    _setup_and_save_plot(dataset_name, scenario_name, measure, value_key, "single")


def plot_single_run_comparison(results_dict: Dict, dataset_name: str, scenario_name: str):
    """Entry point: Iterates through metrics and generates single-run plots and tables."""
    print(f"--- Generating Single Run Plots for {dataset_name} {scenario_name} ---")
    for metric, ylabel in zip(METRICS, Y_LABELS):
        save_metric_table_pdf(results_dict, metric, ylabel, dataset_name, scenario_name)
        plot_single_run(results_dict, metric, ylabel, dataset_name, scenario_name)


def plot_multi_run_comparison(results_dict: List[Dict], dataset_name: str, scenario_name: str):
    """Entry point: Iterates through metrics and generates average/std plots and tables."""
    print(f"--- Generating Multi-Run Plots for {dataset_name} {scenario_name} ---")
    for metric, ylabel in zip(METRICS, Y_LABELS):
        save_metric_table_pdf(results_dict, metric, ylabel, dataset_name, scenario_name)
        plot_avg_std(results_dict, metric, ylabel, dataset_name, scenario_name)