<!-- ShallowJail: Steering Jailbreaks against Large Language Models -->

<p align="center" style="font-size: 1.5em; font-weight: bold;">
    ShallowJail: Steering Jailbreaks against Large Language Models
</p>

<p align="center">
    Shang Liu, Hanyu Pei, Zeyan Liu*
</p>

<p align="center">
    <a href="mailto:shang.liu@louisville.edu">shang.liu@louisville.edu</a>
</p>



![framework](./images/framework.png)


# 1. Requirements
## 1.1 Dependencies
We recommend using [uv](https://github.com/ultralytics/ultralytics) to install dependencies.

```
> uv venv --python 3.12.12
> source .venv/bin/activate
> uv pip install -r requirements.txt
```

## 1.2 Hardware
We have tested the code in NVIDIA RTX 5090. Here is the detailed GPU and driver environment in Ubuntu:

nvidia-smi:
```
NVIDIA-SMI 580.95.05
Driver Version: 580.95.05
CUDA Version: 13.0
```

nvcc:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
```

cudnn:
```
root@3a00f0d2bf79:~/shang/ShallowJail# python -c "import torch; print(torch.backends.cudnn.version())"
91501
```

# 2. Start

## 2.1 Download models

Plese modify the `model_id` and `local_dir` in the `download_models.py` file, and then run the following command:
```
python download_models.py
```

We recommend download the `Qwen/Qwen3-4B-Instruct-2507` and `Qwen/Qwen3Guard-Gen-4B` at first to start the experiments.

## 2.2 Run the code
Single command run:
```
python jailbreak.py -model_path ./models/Qwen3-4B-Instruct-2507 \
                    -guard_path ./models/Qwen3Guard-Gen-4B \
                    -prompt_path ./data/advbench.txt \
                    -alpha 5.5 \
                    -pre_tokens 50 \
                    -beta 0.5 \
                    -max_new_tokens 700
```

Multiple command run:
```
python run.py
```

if config in run.py is:
```
config = {
    "python": ["jailbreak.py"], 
    "model_path": ["./models/Qwen3-4B-Instruct-2507"], 
    "guard_path": ["./models/Qwen3Guard-Gen-4B"],
    "prompt_path": ["./data/advbench.txt"],
    "alpha": [5, 5.5],
    "pre_tokens": [50],
    "beta": [0.5],
    "max_new_tokens": [700],
}
```

it equals with:
```
python jailbreak.py -model_path ./models/Qwen3-4B-Instruct-2507 \
                    -guard_path ./models/Qwen3Guard-Gen-4B \
                    -prompt_path ./data/advbench.txt \
                    -alpha 5 \
                    -pre_tokens 50 \
                    -beta 0.5 \
                    -max_new_tokens 700

python jailbreak.py -model_path ./models/Qwen3-4B-Instruct-2507 \
                    -guard_path ./models/Qwen3Guard-Gen-4B \
                    -prompt_path ./data/advbench.txt \
                    -alpha 5.5 \
                    -pre_tokens 50 \
                    -beta 0.5 \
                    -max_new_tokens 700
```



# 3. Logs
We collected all experiments logs and they could be downloaded from [Google drive](https://drive.google.com/file/d/1Qu3pEfdQDkjHmNd90UxRhnTNExURIL2N/view?usp=sharing). It's about 180M. Feel free to analyze it.

# 4. Citation
If you have any questions, please start a issue or contact shangliu@louisville.edu.

If you find this code useful, please consider citing:
```
@misc{liu2026shallowjailsteeringjailbreakslarge,
      title={ShallowJail: Steering Jailbreaks against Large Language Models}, 
      author={Shang Liu and Hanyu Pei and Zeyan Liu},
      year={2026},
      eprint={2602.07107},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2602.07107}, 
}
```