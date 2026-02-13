import subprocess
from itertools import product

def run_cmd(config):
    keys = list(config.keys())
    values = list(config.values())
    for combo in product(*values):
        args = dict(zip(keys, combo))
        print(f"🚀 Running: {args}")
        cmd = []
        for key, val in args.items():
            if key == "python":
                flag = f"{key}"
            else:
                flag = f"-{key}" 
            cmd.append(flag)
            cmd.append(str(val))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # 和args一一对应就行
    config = {
        "python": ["./jailbreak.py"], # 要执行的脚本名
        
        "model_path": ["/root/shang/fakespeech/models/Qwen3-4B-Instruct-2507"], 
        "prompt_path": ["./data/advbench.txt"],
        "alpha": [5.5],
        "pre_tokens": [50],
        "beta": [0.5],
        "max_new_tokens": [700],
    }

    run_cmd(config)