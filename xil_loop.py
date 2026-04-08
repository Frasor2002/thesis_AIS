from experiments.xil_loop import exp_xil_loop
from experiments.utils import create_common_checkpoint
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log", "xil_plot")

SEED = 123
MODEL="ModernLeNet"
BS_LIST = [
  #[0.99]*10,
  #[0.50]*10,
  [0,0,0,0.99,0.99,0.99,0.99,0.99,0.99,0.99],
  #[0,0,0,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
  #[0,0,0,0,0,0.99,0.99,0.99,0.99,0.99],
  #[0,0,0,0,0,0.5,0.5,0.5,0.5,0.5],
  #[0,0,0,0,0,0,0,0.99,0.99,0.99],
  #[0,0,0,0,0,0,0,0.5,0.5,0.5]
  #[0,0,0,0,0,0,0,0,0,0.99],
]
CONF_TYPE=2
BUDGET=1
STEP=1
INITIAL=0


def plot_helper(results_dict, value_key:str, measure: str, dataset_name, scenario_name):
  strategies = ["simplicity", "simplicity_class", "random"]
  colors = {
    "simplicity": "C1", 
    "simplicity_class": "mediumorchid", 
    "random": "mediumturquoise"
  }
  # Plot confounded samples sampled at each iteraton
  plt.figure(figsize=(10, 6))
  for strat in strategies:    
    log = results_dict[strat]
    queries = log.get("query", [])
    conf_sampled = log.get(value_key, []) 
        
    plt.plot(queries, conf_sampled, marker='', linestyle='-', color=colors[strat], label=strat)
  
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.title(f"Sampling comparison {dataset_name} ({scenario_name})", fontsize=20)
  plt.xlabel("Explained samples", fontsize=15)
  plt.ylabel(measure, fontsize=15)
  plt.tick_params(axis='both', which='major', labelsize=14)
  plt.grid(True)
  plt.legend(fontsize=15)
  plt.tight_layout()
  filename_conf = f"strat_comparison_{dataset_name}_{scenario_name}_{value_key}.pdf"
  save_dir = os.path.join(LOG_DIR,filename_conf)
  os.makedirs(save_dir, exist_ok=True)
  plt.savefig(save_dir)
  plt.close() # Close figure to free memory
  print(f"Saved plot to {save_dir}")


def plot_strat_comparison(results_dict: dict, dataset_name: str, scenario_name: str):
  metrics = ["conf_sampled", "attr_on_conf", "accuracy"]
  x_axis_names = ["Confounded samples sampled", "% Attr on confounder", "Accuracy"]
  for m, x in zip(metrics, x_axis_names):
    plot_helper(results_dict, m, x, dataset_name, scenario_name)

if __name__ == "__main__":
  create_common_checkpoint(SEED, MODEL)
  
  datasets_to_run = [
    ("DecoyMNIST", 1e-1),
    #("DecoyFashionMNIST", 1e-3)
  ]

  for DATASET, rr_reg in datasets_to_run:
    print(f"\nDataset {DATASET}")
        
    for idx, bs in enumerate(BS_LIST):
      print(f"XIL loop bias ratio: {bs}")    
      current_results = {}
            
      scenario_name = f"scenario_{idx+1}"

      print("Simplicity")
      current_results["simplicity"] = exp_xil_loop(
        seed=SEED,
        model_name=MODEL,
        dataset=DATASET,
        bias_ratio=bs,
        conf_type=CONF_TYPE,
        sampling_strategy="simplicity",
        budget=BUDGET,
        step=STEP,
        initial_query=INITIAL,
        rr_reg=rr_reg
      )    
      print("simplicity-class")
      current_results["simplicity_class"] = exp_xil_loop(
        seed=SEED,
        model_name=MODEL,
        dataset=DATASET,
        bias_ratio=bs,
        conf_type=CONF_TYPE,
        sampling_strategy="simplicity_class",
        budget=BUDGET,
        step=STEP,
        initial_query=INITIAL,
        rr_reg=rr_reg
      )  
      print("random")
      current_results["random"] = exp_xil_loop(
        seed=SEED,
        model_name=MODEL,
        dataset=DATASET,
        bias_ratio=bs,
        conf_type=CONF_TYPE,
        sampling_strategy="random",
        budget=BUDGET,
        step=STEP,
        initial_query=INITIAL,
        rr_reg=rr_reg
      )
            
      plot_strat_comparison(current_results, DATASET, scenario_name)
