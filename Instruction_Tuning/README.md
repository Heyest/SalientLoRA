# Adapating LLaMA/LLaMA2 on the Alpaca dataset using SalientLoRA

This directory includes the SalientLoRA implementation and guidelines for reproducing the results in our paper.

## Install dependencies

```bash
conda create -n Instruction_Tuning python=3.9
conda activate Instruction_Tuning 
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e ../loralib/
```



## Download the Alpaca Dadaset

```
sh data/download.sh
```



## Code Usage

The floder `scripts` contains examples of adapting LLaMA/LLaMA2 with SalientLoRA on Alpaca datasets. 

Execute the shell script directly as follows:

```
sh scripts/run_LLaMA_Alpaca.sh  
```



## Evalution

During the evaluation phase,  you can use the [Vicuna eval](https://github.com/lm-sys/vicuna-blog-eval) code to generate responses for MT-Bench using the fine-tuned model, obtaining scores produced by GPT-4.



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

