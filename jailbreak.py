import torch

import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

from selflogger import logger
from args import get_args_parser
from qwen3guard import Qwen3Guard


class ShallowJail:
    def __init__(self, args, victim_model_path):
        self.refuse_messages = [
            [{"role": "user", "content": "Sorry, I cannot assist with that request."}],
            [{"role": "user", "content": "I am unable to provide information on that topic."}],
            [{"role": "user", "content": "I cannot help with that."}],
            [{"role": "user", "content": "I'm sorry, but I can't assist with that."}],
            [{"role": "user", "content": "Apologies, but I won't be able to help with that."}],
            [{"role": "user", "content": "I'm unable to provide guidance on that matter."}],
            [{"role": "user", "content": "Apologies, but I cannot assist with that request."}],
            [{"role": "user", "content": "I cannot provide information on that subject."}],
            [{"role": "user", "content": "I'm sorry, but I cannot help with that."}],
            [{"role": "user", "content": "Apologies, but I won't be able to assist with that."}],
        ]

        self.ok_messages = [
            [{"role": "user", "content": "Sure, I can help you with that."}],
            [{"role": "user", "content": "Of course, here's how you can do it."}],
            [{"role": "user", "content": "Absolutely, let me guide you through it."}],
            [{"role": "user", "content": "Certainly, here's the information you need."}],
            [{"role": "user", "content": "Yes, I can provide assistance with that."}],
            [{"role": "user", "content": "Definitely, here's how to proceed."}],
            [{"role": "user", "content": "Of course, let me help you with that."}],
            [{"role": "user", "content": "Sure, here's what you need to know."}],
            [{"role": "user", "content": "Absolutely, I can assist you with that."}],
            [{"role": "user", "content": "Certainly, here's how to do it."}],
        ]

        self.max_new_tokens = args.max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(victim_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(victim_model_path, device_map="auto", dtype="auto")

        self.layers_diff = self.calc_diff_input_embedding()["layers_diffs"]

        self.alpha = args.alpha
        self.beta = args.beta
        self.pre_tokens = args.pre_tokens

        self.prompts = self.load_prompts(args.prompt_path)

    def load_prompts(self, data_path):
        messages_new = []
        # 读取txt
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                messages_new.append([{"role": "user", "content": line.strip()}])
        return messages_new


    def calc_diff_input_embedding(self):
        layers_diff = {}

        # 最后一个token在每一层的hidden state
        for i, msg_1 in enumerate(self.refuse_messages):
            for j, msg_2 in enumerate(self.ok_messages):
                inputs_1 = self.tokenizer.apply_chat_template(
                    msg_1,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_tensors="pt",
                ).to(self.model.device)

                inputs_2 = self.tokenizer.apply_chat_template(
                    msg_2,
                    add_generation_prompt=False,
                    tokenize=True,
                    return_tensors="pt",
                ).to(self.model.device)

                with torch.no_grad():
                    outputs_1 = self.model(inputs_1["input_ids"], output_hidden_states=True)
                    outputs_2 = self.model(inputs_2["input_ids"], output_hidden_states=True)
                
                for layer in range(len(outputs_1.hidden_states)):
                    state_1 = outputs_1.hidden_states[layer][:, -1, :]
                    state_2 = outputs_2.hidden_states[layer][:, -1, :]

                    diff = state_2 - state_1
                    # diff = diff / (diff.norm(p=2) + 1e-12)

                    if layer not in layers_diff:
                        layers_diff[layer] = [diff]
                    else:
                        layers_diff[layer].append(diff)

            #     break
            # break

        # 做平均然后归一化
        for k, v in layers_diff.items():
            layers_diff[k] = torch.stack(v).mean(dim=0)
            layers_diff[k] = layers_diff[k] / (layers_diff[k].norm(p=2) + 1e-12)
        
        # self.layers_diff = layers_diff
        return {
            "layers_diffs": layers_diff,
        }


    def calc_normal_output(self, msg):
        inputs = self.tokenizer.apply_chat_template(
            msg,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95, top_k=50, repetition_penalty=1.3)

        perplexity = self.calc_perplexity(outputs, input_len)

        output = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        return {
            "output": output,
            "perplexity": perplexity,
            "distinct_2": self.get_distinct_n(output, n=2),
        }


    def calc_jailbreak_layer_diff(self, msg):
        inputs = self.tokenizer.apply_chat_template(
            msg,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            def get_hook_function(layer_idx):
                count = 0
                def hook_function(module, input, output):
                    nonlocal count
                    
                    tmp = self.layers_diff[layer_idx][None, :, :]
                    
                    if count < self.pre_tokens:
                        modified_output = output + self.alpha * tmp
                    else:
                        modified_output = output + self.beta * self.alpha * tmp

                    count += 1
                    return modified_output
                return hook_function
            hooks = []
            for i in range(len(self.model.model.layers)):
                if i == 0:
                    continue
                
                hook = self.model.model.layers[i].register_forward_hook(get_hook_function(i))
                hooks.append(hook)
            
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.1)
            
            for hook in hooks:
                hook.remove()

            perplexity = self.calc_perplexity(outputs, input_len)

        decoded_output = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

        return {
            "output": decoded_output,
            "perplexity": perplexity,
            "distinct_2": self.get_distinct_n(decoded_output, n=2),
        }


    def calc_perplexity(self, outputs, input_len):
        labels = outputs.clone()
        # 将 Prompt 部分的 label 设为 -100，使其不参与 Loss 计算
        labels[:, :input_len] = -100 
        outputs_loss = self.model(outputs, labels=labels).loss
        perplexity = torch.exp(outputs_loss)
        return perplexity.item()


    def get_distinct_n(self, text, n=2):
        tokens = list(text) 
        if len(tokens) < n:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        return len(set(ngrams)) / len(ngrams) if ngrams else 0


    def run(self, qwen3guard):
        jail_check_all = 0
        perplexity_all = 0
        distincts_all = 0

        # layer_pers = calc_layer_perplexity(
        for idx, msg in enumerate(self.prompts):
            jail_values = self.calc_jailbreak_layer_diff(msg)
        
            # jail_check = jailbreak_check(jail_values['output'])
            jail_check = qwen3guard.is_unsafe(msg[0]['content'], jail_values['output'])
            
            perplexity_all += jail_values['perplexity']
            jail_check_all += jail_check
            distincts_all += jail_values['distinct_2']

            logger.info("--------------------------------------------------")
            logger.info(f"Processing {idx+1} / {len(self.prompts)}")
            logger.info(f"Jailbreak input: \n {msg}")
            logger.info(f"Jailbreak output: \n {jail_values['output']}")
            logger.info(f"Jailbreak perplexity: {jail_values['perplexity']}")
            logger.info(f"Jailbreak check: {jail_check}")
            logger.info(f"Jailbreak successful rate so far: {jail_check_all} / {idx+1} = {jail_check_all / (idx+1):.4f}")
            logger.info(f"Distinct-2 so far: {distincts_all} / {idx+1} = {distincts_all / (idx+1):.4f}")
            logger.info("--------------------------------------------------")

            # if idx == 49:
            #     break

        logger.info(vars(args))
        logger.info(f"Total jailbreak checks: {jail_check_all} / {len(self.prompts)} = {jail_check_all / len(self.prompts):.4f}")
        logger.info(f"Average perplexity: {perplexity_all / len(self.prompts)}")
        logger.info(f"Average Distinct-2: {distincts_all / len(self.prompts):.4f}")
        

if __name__ == "__main__":
    args = get_args_parser()

    local_path = args.model_path
    qwen3guard_path = args.guard_path

    shallowjail = ShallowJail(args, local_path)
    logger.info(f"Loaded model from {local_path}")

    qwen3guard = Qwen3Guard(qwen3guard_path)
    logger.info(f"Loaded Qwen3Guard model from {qwen3guard_path}")

    logger.info(f"model layers, {len(shallowjail.model.model.layers)}")
    logger.info(f"model dtype, {shallowjail.model.dtype}")
    logger.info(vars(args))


    shallowjail.run(qwen3guard)


    