from experiments.confounder_study import exp_confounder_study
from experiments.utils import create_common_checkpoint

if __name__ == "__main__":
  BS_LIST = [
    [0.99]*10,
    #[0.50]*10,
    #[0,0,0,0.99,0.99,0.99,0.99,0.99,0.99,0.99],

    #[0,0,0,0,0,0.99,0.99,0.99,0.99,0.99],

    #[0,0,0,0,0,0,0,0.99,0.99,0.99],

    [0,0,0,0,0,0,0,0,0,0.99],

  ]


  SEED = 123
  MODEL="ModernLeNet"
  DATASET="DecoyMNIST"
  CONF_TYPES=[0,1,2] # [0,1,2]

  create_common_checkpoint(SEED, MODEL)


  print(f"Dataset {DATASET}")
  for i in range(len(BS_LIST)):
    for conf in CONF_TYPES:
      print(f"Confounder: {conf} | Bias ratio: {BS_LIST[i]} | Model: {MODEL}")
      exp_confounder_study(
        seed=SEED,
        model_name=MODEL,
        dataset=DATASET,
        bias_ratio=BS_LIST[i],
        conf_type=conf,
        add=str(i)
      )


  DATASET="DecoyFashionMNIST"
  print(f"Dataset {DATASET}")
  for i in range(len(BS_LIST)):
    for conf in CONF_TYPES:
      print(f"Confounder: {conf} | Bias ratio: {BS_LIST[i]} | Model: {MODEL}")
      exp_confounder_study(
        seed=SEED,
        model_name=MODEL,
        dataset=DATASET,
        bias_ratio=BS_LIST[i],
        conf_type=conf,
        add=str(i)
      )
 