from experiments.select_classes import run_class_selector

SEED = 123
MODEL_NAME = "ModernLeNet"
CONF_TYPE = 2
TRAIN_PATCH = False

def gen_plots():
  # Plot generation for different cases
  DATA = ["DecoyMNIST", "DecoyFashionMNIST"]
  BRS = [
    [0]*3 + [0.99] * 7,
    [0]*5 + [0.99] * 5,
    [0]*7 + [0.99] * 3,
    [0]*9 + [0.99] * 1,

    #[0.99]*3 + [0] * 7,
    #[0.99]*5 + [0] * 5,
    #[0.99]*7 + [0] * 3,
    #[0.99]*9 + [0] * 1,
  ]
  for data in DATA:
    for bs in BRS:
      run_class_selector(
        SEED,
        MODEL_NAME,
        data,
        bs,
        CONF_TYPE,
        TRAIN_PATCH
      )

if __name__ == "__main__":
  #run_class_selector( SEED,MODEL_NAME,DATASET,BIAS_RATIO,CONF_TYPE,TRAIN_PATCH)

  gen_plots()

  