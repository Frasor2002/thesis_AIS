from experiments.confounder_study import exp_confounder_study

if __name__ == "__main__":
  BS_LIST = [
    #[0.99]*10,
    #[0.50]*10,
    #[0,0,0,0.99,0.99,0.99,0.99,0.99,0.99,0.99],
    #[0,0,0,0.5,0.5,0.5,0.5,0.5,0.5,0.5],

    #[0,0,0,0,0,0.99,0.99,0.99,0.99,0.99],
    #[0,0,0,0,0,0.5,0.5,0.5,0.5,0.5],

    #[0,0,0,0,0,0,0,0.99,0.99,0.99],
    #[0,0,0,0,0,0,0,0.5,0.5,0.5]

    [0,0,0,0,0,0,0,0,0,0.99],

  ]


  SEED = 123
  MODEL_LIST=["ModernLeNet"]
  MODEL="ModernLeNet"
  DATASET="DecoyMNIST"
  CONF_TYPES=[2] # [0,1,2]

  print(f"Dataset {DATASET}")
  for i in range(len(BS_LIST)):
    for conf in CONF_TYPES:
      for model in MODEL_LIST:
        print(f"Confounder: {conf} | Bias ratio: {BS_LIST[i]} | Model: {model}")
        exp_confounder_study(
          seed=SEED,
          model_name=model,
          dataset=DATASET,
          bias_ratio=BS_LIST[i],
          conf_type=conf,
          add=str(i)
        )


  DATASET="DecoyFashionMNIST"
  print(f"Dataset {DATASET}")
  for i in range(len(BS_LIST)):
    for conf in CONF_TYPES:
      for model in MODEL_LIST:
        print(f"Confounder: {conf} | Bias ratio: {BS_LIST[i]} | Model: {model}")
        exp_confounder_study(
          seed=SEED,
          model_name=model,
          dataset=DATASET,
          bias_ratio=BS_LIST[i],
          conf_type=conf,
          add=str(i)
        )
 