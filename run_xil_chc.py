from experiments.xil_loop import run_celeba_xil
from experiments.utils import create_common_checkpoint
from experiments.xil_plots import plot_single_run_comparison, plot_multi_run_comparison

SEEDS = [123, 111, 222, 333, 444]
MODEL = "ResNet" # Other params are default
DYNAMIC_SIMPLICITY = False


def run_celeba(seed):
  results = {}
  for strat in ["simplicity", "random", "adaptive"]:
    print(strat)
    results[strat] = run_celeba_xil(
      seed=seed,
      sampling_strategy=strat,
      budget=25,
      step=1,
      initial_query=0,
    )

    #print(results)
  return results


if __name__ == "__main__":
  c_res = []
  for seed in SEEDS:
    create_common_checkpoint(seed, MODEL)
    res = run_celeba(seed)
    c_res.append(res)

    # plot single run
    plot_single_run_comparison(res, f"Celeba_{seed}", "")
    print(f"Partial result Celeba_{seed}:", res)

  print("Final results celeba")
  print(c_res)
  # plot multiple runs
  plot_multi_run_comparison(c_res, "Celeba", "")

