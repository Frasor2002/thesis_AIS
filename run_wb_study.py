from experiments.data_study import run_wb_study
from experiments.utils import create_common_checkpoint

SEED=123

if __name__ == "__main__":
  create_common_checkpoint(SEED, "Resnet") # Other param default
  run_wb_study(SEED)