from experiments.xil_loop import exp_xil_loop

SEED = 123
MODEL="ModernLeNet"
DATASET="DecoyMNIST"
BS_LIST = [
  #[0.99]*10,
  #[0.50]*10,
  [0,0,0,0.99,0.99,0.99,0.99,0.99,0.99,0.99],
  #[0,0,0,0.5,0.5,0.5,0.5,0.5,0.5,0.5],
  [0,0,0,0,0,0.99,0.99,0.99,0.99,0.99],
  #[0,0,0,0,0,0.5,0.5,0.5,0.5,0.5],
  [0,0,0,0,0,0,0,0.99,0.99,0.99],
  #[0,0,0,0,0,0,0,0.5,0.5,0.5]
  [0,0,0,0,0,0,0,0,0,0.99],
]

CONF_TYPE=2
BUDGET=1000
STEP=10
INITIAL=0
RR_REG=1e-1

if __name__ == "__main__":
  
  print(f"Dataset {DATASET}")
  for bs in BS_LIST:
    print(f"XIL loop bias ratio: {bs}")
    print("Simplicity")
    exp_xil_loop(
      seed=SEED,
      model_name=MODEL,
      dataset=DATASET,
      bias_ratio=bs,
      conf_type=CONF_TYPE,
      sampling_strategy="simplicity",
      budget=BUDGET,
      step=STEP,
      initial_query=INITIAL,
      rr_reg=RR_REG
    ) 
    print("simplicity-class")
    exp_xil_loop(
      seed=SEED,
      model_name=MODEL,
      dataset=DATASET,
      bias_ratio=bs,
      conf_type=CONF_TYPE,
      sampling_strategy="simplicity_class",
      budget=BUDGET,
      step=STEP,
      initial_query=INITIAL,
      rr_reg=RR_REG
    )
    print("random")
    exp_xil_loop(
      seed=SEED,
      model_name=MODEL,
      dataset=DATASET,
      bias_ratio=bs,
      conf_type=CONF_TYPE,
      sampling_strategy="random",
      budget=BUDGET,
      step=STEP,
      initial_query=INITIAL,
      rr_reg=RR_REG
    )
  

  DATASET="DecoyFashionMNIST"
  print(f"Dataset {DATASET}")
  for bs in BS_LIST:
    print(f"XIL loop bias ratio: {bs}")
    print("Simplicity")
    exp_xil_loop(
      seed=SEED,
      model_name=MODEL,
      dataset=DATASET,
      bias_ratio=bs,
      conf_type=CONF_TYPE,
      sampling_strategy="simplicity",
      budget=BUDGET,
      step=STEP,
      initial_query=INITIAL,
      rr_reg=1e-3
    )
    print("simplicity-class")
    exp_xil_loop(
      seed=SEED,
      model_name=MODEL,
      dataset=DATASET,
      bias_ratio=bs,
      conf_type=CONF_TYPE,
      sampling_strategy="simplicity_class",
      budget=BUDGET,
      step=STEP,
      initial_query=INITIAL,
      rr_reg=1e-3
    )
    print("random")
    exp_xil_loop(
      seed=SEED,
      model_name=MODEL,
      dataset=DATASET,
      bias_ratio=bs,
      conf_type=CONF_TYPE,
      sampling_strategy="random",
      budget=BUDGET,
      step=STEP,
      initial_query=INITIAL,
      rr_reg=1e-3
    )

