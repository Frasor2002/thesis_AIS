from experiments.select_classes import run_class_selector

SEED = 123
MODEL_NAME = "ModernLeNet"
DATASET = "DecoyFashionMNIST"
BIAS_RATIO = [0]*5 + [0.99] * 5
CONF_TYPE = 2
TRAIN_PATCH = False

if __name__ == "__main__":
  run_class_selector(
    SEED,
    MODEL_NAME,
    DATASET,
    BIAS_RATIO,
    CONF_TYPE,
    TRAIN_PATCH
  )