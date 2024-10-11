# Preface


# Code Structure
- `artifacts`: This directory stores some artifacts, e.g., result files.
   - `exp_pool`: This directory stores the experience pool files, which will be used for LLM adaptation.
   - `results`: This directory stores the result files.
- `data`: This directory stores datasets and pre-trained model checkpoints of baselines.
   - `traces`: This directory stores the bandwidth trace datasets.
   - `videos`: This directory stores the video specifications.
   - `ft_plms`: This directory stores the fine-tuned (adapted) LLMs.
   - `all_models`: This directory stores the model checkpoints of baselines.
- `baseline_special`: This directory stores the codes for running baselines. Most of the codes are from the Genet's repository.
- `plm_special`: This directory stores the codes for running NetLLM.
   - `data`: This directory stores the codes related to the training datasets for LLM adaptation.
      - `exp_pool.py`: Implements the experience pool for collecting trajectories.
      - `dataset.py`: Implements a dataset class that wraps the experience pool.
   - `models`: This directory stores the codes related to NetLLM.
      - `state_encoder.py`: Implements the feature encoder for encoding states.
      - `gpt2.py`, `llama.py`, `opt.py`, `mistral.py`, `t5.py`: Customized LLMs.
      - `low_rank.py`: Implements the low rank matrices.
      - `rl_policy.py`: Implements the Transformer-based offline RL policy.
   - `utils`: This directory stores some utilization codes.
      - `plm_utils.py`: Some codes for loading LLMs.
      - `utils.py`: Some codes for data processing.
   - `trainer.py`: Some codes for training (adapting) LLMs.
   - `evaluate.py`: Some codes for evaluating the performance of adapted-LLMs.
   - `test.py`: Some codes for testing the performance of adapted LLMs(This is the official test.py from NetLLM).
   - `testing.py`: This is the code which will be tested via AQM-LLM model.
- `generate_exp_pool.py`: Implements the generation of experience pool (i.e., training dataset for LLM).
- `run_baseline.py`: The main file for running baselines.
- `run_plm.py`: The main file for running NetLLM.
- `exp_pool.pkl`: This is the file which will be used to train the model of AQM-LLM. The address of this is AQM-LLM\AQM-LLM\AQM-LLM\exp_pool.pkl.

There are several parts are changed to let AQM-LLM working with new exp pool file, compare with NetLLM:
- `run_plm.py`
- `state_encoder.py`
- `rl_policy.py`
- 'exp_pool.pkl'

# Environment Setup
Create a conda environment for NetLLM:
```
'conda create -n abr_netllm python>=3.8.10'
 ```

Activating the Conda environment
```
'conda activate abr_netllm'
```

Then install the following depdendencies one by one:
```
python==3.8.10
torch==2.1.0
numpy==1.24.4
munch==4.0.0
openprompt==1.0.1
transformers==4.34.1
peft==0.6.2
```

Or use this:
```
python -m pip install --upgrade pip && pip install openprompt==1.0.1 && pip install numpy==1.24.4 && pip install peft==0.6.2 && pip install transformers==4.34.1 && pip install --upgrade huggingface_hub && pip install scikit-learn && pip install munch
```

# Usage
To run NetLLM, first we need to download some LLMs. For example, if you want to use Llama2-7b as the foundation model, please download Llama2-7b in the directory: ../downloaded_plms/llama2/base. In the following, we will use the Llama2-7b as the example to illustrate the usage of NetLLM.

## Finetune LLM
If you want to finetune LLM, please run the following command. The number after --num-epochs specifies how many times the model needs to be run in its entirety on the training dataset:
```
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --device-out cuda:1 --lr 0.0001 --warmup-steps 2000 --num-epochs 20 --eval-per-epoch 2 --exp-pool-path ./exp_pool.pkl
```
Reminder: After installing every environmental configurations, if there is anything you missed, at this stage the terminal will warn you and provide the missing part.

## Test LLM
After training LLM need to re-tune run_plm.py, please use run_plm_for_testing.py for training, currently how to integrate run_plm and run_plm_for_testing is still being solved, and run_plm_for_testing.py will only be used when testing LLM.
```
python run_plm.py --test --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --device-out cuda:1 --lr 0.0001 --warmup-steps 2000 --num-epochs 20 --eval-per-epoch 2 --exp-pool-path ./exp_pool.pkl
```

Or use your own address
```
python run_plm.py --test --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --device-out cuda:1 --lr 0.0001 --warmup-steps 2000 --num-epochs 20 --eval-per-epoch 2 --exp-pool-path your_exp_pool_path
```


