from experiments.xil_loop import run_wb_xil
from experiments.utils import create_common_checkpoint
from experiments.xil_plots import get_scenario_name, plot_single_run_comparison, plot_multi_run_comparison

SEEDS = [123, 111, 222]
MODEL = "ResNet" # Other params are default


def run_wb(seed):
  results = {}
  for strat in ["simplicity", "random", "adaptive"]:
    print(strat)
    results[strat] = run_wb_xil(
      seed=seed,
      sampling_strategy=strat,
      budget=25,
      step=1,
      initial_query=0
    )

    #print(results)
  return results


if __name__ == "__main__":
  # wb study
  wb_res = []
  for seed in SEEDS:
    create_common_checkpoint(seed, MODEL, "_wb")
    res = run_wb(seed)
    wb_res.append(res)

    # plot single run
    plot_single_run_comparison(res, f"Waterbirds_{seed}", "")
    print(f"Partial result Waterbird_{seed}:", res)

  print("Final results")
  print(wb_res)
  # plot multiple runs
  plot_multi_run_comparison(wb_res, "Waterbirds", "")

