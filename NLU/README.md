# Finetuning DeBERTaV3 on GLUE benchmark using SalientLoRA

This directory includes the SalientLoRA implementation and guidelines for reproducing the results in our paper.


## Install dependencies

```bash
conda create -n NLU python=3.7
conda activate NLU 
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e .  # Install `transformers`
pip install -e ../loralib/
```



## Code Usage

The floder `scripts` contains examples of adapting DeBERTaV3-base with SalientLoRA on GLUE datasets. 

Execute the shell script directly as follows:

```
sh examples/scripts/run_debertav3_cola.sh  
```



### Hyperparameter Setup

+ `average_initial_rank`:  The initial rank of each incremental matrix. 
+ `average_target_rank`: The average target rank of each incremental matrix. 
+ `initial_warmup`:  The number of steps to warm up the training before rank allocation.
+ `allocation_step`: The number of steps during the rank allocation phase.
+ `initial_time_window `: The initial window size of time-series.
+ `final_time_window`: The final window size of time-series.
+ `beta`: The correlation threshold. 
+ `gamma`: The slope threshold for dependency calculation. 
+ `lambda_para`: The hyperparameter controlling the degree of contribution in salience measurement.



