from experiments.xil_loop import run_wb_xil
from experiments.utils import create_common_checkpoint
from experiments.xil_plots import get_scenario_name, plot_single_run_comparison, plot_multi_run_comparison

SEEDS = [123, 111, 222, 333, 444]
MODEL = "ResNet"
bs7 = ([0]*3 + [0.99]*7)
bs5 = ([0]*5 + [0.99]*5)
bs3 = ([0]*7 + [0.99]*3)
bs1 = ([0]*9 + [0.99]*1)
BS = [bs7, bs5, bs3, bs1]

def run_wb(seed):
  results = {}
  for strat in ["simplicity_class", "simplicity", "random"]:
    print(strat)
    results[strat] = run_wb_xil(
      seed=seed,
      sampling_strategy=strat,
      budget=25,
      step=1,
      initial_query=0
    )
  return results


if __name__ == "__main__":
  # wb study
  wb_res = []
  for seed in SEEDS:
    create_common_checkpoint(seed, MODEL)
    res = run_wb(seed)
    wb_res.append(res)

    # plot single run
    plot_single_run_comparison(res, "Waterbirds", "")

  # plot multiple runs
  plot_multi_run_comparison(mnist_res, "Waterbirds", "")

