from experiments.xil_loop import run_mnist_xil, run_fmnist_xil
from experiments.utils import create_common_checkpoint
from experiments.xil_plots import get_scenario_name, plot_single_run_comparison, plot_multi_run_comparison

SEEDS = [123, 111, 222] # 333, 444
MODEL = "ModernLeNet"
bs7 = ([0]*3 + [0.99]*7)
bs5 = ([0]*5 + [0.99]*5)
bs3 = ([0]*7 + [0.99]*3)
bs1 = ([0]*9 + [0.99]*1)
BS = [bs7]

def run_mnist(seed, model, bs):
  results = {}
  for strat in ["simplicity_class", "simplicity", "random"]:
    print(strat)
    results[strat] = run_mnist_xil(
      seed,
      model,
      bias_ratio=bs,
      conf_type=2,
      train_patch=False,
      sampling_strategy=strat,
      budget=25,
      step=1,
      initial_query=0)
  return results

def run_fmnist(seed, model, bs):
  results = {}
  for strat in ["simplicity_class", "simplicity", "random"]:
    print(strat)
    results[strat] = run_fmnist_xil(
      seed,
      model,
      bias_ratio=bs,
      conf_type=2,
      train_patch=False,
      sampling_strategy=strat,
      budget=25,
      step=1,
      initial_query=0)
  return results

if __name__ == "__main__":
  # mnist study
  for bs in BS:
    mnist_res = []
    for seed in SEEDS:
      create_common_checkpoint(seed, MODEL)
      res = run_mnist(seed, MODEL, bs)
      mnist_res.append(res)

      # plot single run
      plot_single_run_comparison(res, f"DecoyMNIST_{seed}", get_scenario_name(bs))

    # plot multiple runs
    plot_multi_run_comparison(mnist_res, "DecoyMNIST",  get_scenario_name(bs))


  # fmnist study
  for bs in BS:
    fmnist_res = []
    for seed in SEEDS:
      create_common_checkpoint(seed, MODEL)
      res = run_fmnist(seed, MODEL, bs)
      fmnist_res.append(res)

      # plot single run
      plot_single_run_comparison(res, f"DecoyFMNIST_{seed}", get_scenario_name(bs))

    # plot multiple runs
    plot_multi_run_comparison(fmnist_res, "DecoyFMNIST",  get_scenario_name(bs))