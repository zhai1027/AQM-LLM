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
   - `test.py`: Some codes for testing the performance of adapted LLMs.
- `generate_exp_pool.py`: Implements the generation of experience pool (i.e., training dataset for LLM).
- `run_baseline.py`: The main file for running baselines.
- `run_plm.py`: The main file for running NetLLM.
