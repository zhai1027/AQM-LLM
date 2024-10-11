# Preface
The code for this project is revised which is based on the [NetLLM](https://github.com/duowuyms/NetLLM.git) repository.

# Code Structure
- `artifacts`: This directory stores some artifacts, e.g., result files.
   - `exp_pool`: This directory stores the experience pool files, which will be used for LLM adaptation.
   - `results`: This directory stores the result files.

- `data`: This directory stores datasets and pre-trained model checkpoints of baselines.
   - `traces`: This directory stores the bandwidth trace datasets.
   - `videos`: This directory stores the video specifications.
   - `ft_plms`: This directory stores the fine-tuned (adapted) LLMs.
   - `all_models`: This directory stores the model checkpoints of baselines.

- `baseline_specical`: This directory stores the codes for runing baselines. Most of the codes are from the Genet's repository.
- `plm_special`: This directory stores the codes for running NetLLM.
   - `data`: This directory stores the codes related to the training datasets for LLM adaptation.
      - `exp_pool.py`: Implements the experience pool for collecting trajectories.
      - `dataset.py`: Implements a dataset class that wraps the experience pool.
    - `models`: This directory stores the codes related to NetLLM.
      - `state_encoder.py`: Implements the feature encoder for encoding states.
      - `gpt2.py, llama.py, opt.py, mistral.py, t5.py`: Customized LLMs.
      - `low_rank.py`: Implements the low rank matrices.
      - `rl_policy.py`: Implements the Transformer-based offline RL policy.
    - `utils`: This directory stores some utilization codes.
      - `plm_utils.py`: Some codes for loading LLMs.
      - `utils.py`: Some codes for data processing.
    - `trainer.py`: Some codes for training (adapting) LLMs. 
    - `evaluate.py`: Some codes for evaluting the performance of adapted-LLMs.
    - `test.py`: Some codes for testing the performance of adapted LLMs.
- `generate_exp_pool.py`: Implements the generation of experience pool (i.e., training dataset for LLM).
- `run_baseline.py`: The main file for running baselines. 
- `run_plm.py`: The main file for running NetLLM.

# Environment Setup
## Environment for NetLLM
1. Create a conda environment for NetLLM:

   `conda create -n abr_netllm python>=3.8.10`

2. Then install the following depdendencies:

   ```
   python==3.8.10
   torch==2.1.0
   numpy==1.24.4
   munch==4.0.0
   openprompt==1.0.1
   transformers==4.34.1
   peft==0.6.2
   ```

## Environment for baselines
To run baselines, we need a different environment, since they are mainly written in tensforflow v1.

1. First, create a conda environment with `python=3.7`. Please note that you must install `python=3.7`, since the greater versions of python do not support installing tensorflow 1.x any more.

   `conda create -n abr_tf python=3.7`

2. Next, install the following dependencies.
   ```sh
   conda activate abr_tf
   pip install tensorflow-gpu==1.15
   pip install tensorboard==1.15.0
   pip install tensorboard-plugin-wit==1.8.0
   pip install tflearn==0.5.0
   pip install numba==0.53.1
   pip install gym==0.18.0
   pip install stable-baselines[mpi]==2.10.1
   pip install pandas==1.1.5
   pip install tqdm==4.62.2
   ```
# Usage
## Usage of NetLLM
To run NetLLM, first we need to download some LLMs. For example, if you want to use Llama2-7b as the foundation model, please download Llama2-7b in the directory: `../downloaded_plms/llama2/base`. In the following, we will use the Llama2-7b as the example to illustrate the usage of NetLLM.

**Finetune LLM**

If you want to finetune LLM, please run the following command:
```sh
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 
```
This command will finetune Llama2 on the default experience pool we provided at `artifacts/exp_pools/exp_pool.pkl`.
If you want to use your own experience pool, first use the `generate_exp_pool.py` to generate a new experience pool.
```sh
conda activate abr_tf  # since we need to use baselines to interact with environments, we need to activate the baseline environment first.
python generate_exp_pool.py --models genet --traces fcc-train --video video1 --trace-num 100 --cuda-id 0
```
Next, specify the path to your own experience pool with argument `--exp-pool-path` and run the following command:
```sh
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2--exp-pool-path your_exp_pool_path
```

**Test LLM**

If you want to test the performance of the finetuned LLM, please run the following command:
```sh
python run_plm.py --test --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2
```
You can also specify the path to the finetuned LLM with argument `--model-dir`:
```sh
python run_plm.py --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --model-dir you_finetune_llm_dir
```

# Citation
If you find this repository useful, please kindly cite the following paper:
```
@misc{wu2024netllm,
      title={NetLLM: Adapting Large Language Models for Networking}, 
      author={Duo Wu and Xianda Wang and Yaqi Qiao and Zhi Wang and Junchen Jiang and Shuguang Cui and Fangxin Wang},
      year={2024},
      eprint={2402.02338},
      archivePrefix={arXiv},
      primaryClass={cs.NI},
      url={https://arxiv.org/abs/2402.02338}, 
}

@inproceedings{xia2022genet,
  title={Genet: automatic curriculum generation for learning adaptation in networking},
  author={Xia, Zhengxu and Zhou, Yajie and Yan, Francis Y and Jiang, Junchen},
  booktitle={Proceedings of the ACM SIGCOMM 2022 Conference},
  pages={397--413},
  year={2022}
}
```
