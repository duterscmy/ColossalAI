# -*- coding: utf-8 -*-
import argparse
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, T5Tokenizer, AutoConfig, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from typing import List, Optional
from huggingface_hub import snapshot_download
from absl import flags
import random
import os
import json
import shortuuid
import time

from colossalai.moe.expert_idx import expert_idxs_list, global_layer_list, prune_layer_list, layer_num_list


class StopAfterEosTextGenerated(LogitsProcessor):
    """Logits processor (to use with HuggingFace `generate()` method :
    https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/
    text_generation#transformers.generation_utils.GenerationMixin).

   Sometimes our model output '▁</', 's', '>' seperately as stopping signal(not '▁</s>' as a whole),
   which is unable to be captured by a single eos token and can cause a very long generation.
   This logitsprocessor will force generation stop after ' </', 's', '>'.

    Args:
        base_len (int): Size of the given context. Used to know if this is
            the first character to generate.
        eos_token_id (int): ID of the EOS token.
    """

    def __init__(self, base_len: int, eos_token_id: int):
        super().__init__()
        self.base_len = base_len
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.size(1) > self.base_len:
            forced_eos = torch.full(
                (scores.size(1),), -float("inf")).to(scores.device)
            forced_eos[self.eos_token_id] = 0
            # If the last three tokens of input_ids are the stop_token_ids, a eos will be forced to generate afterwards
            stop_token_ids = torch.Tensor([15501, 281, 926]).to(
                scores.device)  # ids for tokens '▁</', 's', '>'
            stop_sample_ids = torch.eq(
                input_ids[:, -len(stop_token_ids):], stop_token_ids).all(dim=1)
            scores[stop_sample_ids] = forced_eos
        return scores


def compute_ppl(model, tokenizer, input_strs, gen_kwargs,
                add_special_tokens=True, split_special_tokens=False, output_only=True, verbose=False):

    model = model.eval()

    # Tokenization
    def encode_text_batch(input_strs):
        inputs = tokenizer.batch_encode_plus(input_strs,
                                             padding='longest',
                                             #  add_special_tokens=add_special_tokens,
                                             #  split_special_tokens=split_special_tokens,
                                             return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        return input_ids

    batch_size = 1  # 批处理大小
    num_texts = len(input_strs)
    loss_sum = 0.0

    for i in range(0, len(input_strs), batch_size):
        text_list_batch = input_strs[i:i+batch_size]
        input_ids = encode_text_batch(text_list_batch)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.mean()
            print("mean loss {}".format(loss))
        loss_sum += loss.item()
        print("loss sum {}".format(loss_sum))

    mean_loss = loss_sum / num_texts  # 计算整个数据集的损失均值
    mean_ppl = torch.exp(torch.tensor(mean_loss))
    return mean_ppl


def apply_llama_chat_template(tokenizer, input_strs, sys_prompt):
    # Use LLaMA's Chat Template(A bit diffrent from original one at the beginning part, we may correct it to the standard llama prompt template later)
    # input_strs = [('user_input', 'user'), ('AI_response', 'assistant'), ...]
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"
    system_prompt = {'content': sys_prompt, 'role': 'system'}
    chat = [system_prompt] + [{'content': input_str,
                               'role': role} for input_str, role in input_strs]
    input_str = tokenizer.apply_chat_template(chat,
                                              tokenize=False,
                                              add_generation_prompt=True)
    return input_str


# @markdown 1. Path to the checkpoint repo
# @param {type:"string"}
pytorch_checkpoint_path = "OrionZheng/openmoe-8b-chat"
pytorch_checkpoint_path = "OrionZheng/openmoe-8b"
# pytorch_checkpoint_path = "OrionZheng/openmoe-base"
model_name = pytorch_checkpoint_path.split("/")[-1]
# @param ["", "0", "0,1", "0,1,2"] {allow-input: true}
available_gpu_ids_str = "0"
memory_per_gpu = "38GiB"  # @param ["", "38GiB"] {allow-input: true}
cpu_memory = '50GiB'  # @param ["50GiB"] {allow-input: true}
model_dtype = 'bfloat16'  # @param ["float32", "bfloat16"]
offload = False  # @param {type:"boolean"}

if torch.cuda.is_available():
    cuda_list = available_gpu_ids_str.split(',')
    print("cuda list{}".format(cuda_list))
else:
    print("no_cuda")
    available_gpu_ids_str, memory_per_gpu = "", ""
    model_dtype = "float32"
    cuda_list = []


no_split_module_classes = "OpenMoeDecoderLayer"

# 1. Allocate Devices for Inference
available_memory = {int(cuda): memory_per_gpu for cuda in cuda_list}
available_memory['cpu'] = cpu_memory
print('Available Devices and Memory: ', available_memory)

# 2. Load the Model (init with empty weight to save memory)
config = AutoConfig.from_pretrained(pytorch_checkpoint_path)
weights_location = snapshot_download(repo_id=pytorch_checkpoint_path)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config,
                                             torch_dtype=eval(
                                                 f'torch.{model_dtype}'),
                                             trust_remote_code=True)
print('Model dtype: ', model.dtype)
device_map = infer_auto_device_map(model,
                                   max_memory=available_memory,
                                   no_split_module_classes=no_split_module_classes)
print('Inferred Device Map: \n', device_map)
if offload:
    model = load_checkpoint_and_dispatch(model, weights_location,
                                         device_map=device_map,
                                         offload_folder="offload",
                                         offload_state_dict=True,
                                         dtype=eval(f'torch.{model_dtype}'),
                                         no_split_module_classes=[no_split_module_classes])
else:
    model = load_checkpoint_and_dispatch(model, weights_location,
                                         device_map=device_map,
                                         dtype=eval(f'torch.{model_dtype}'),
                                         no_split_module_classes=[no_split_module_classes])
print('Fine-grained Device Map: \n', model.hf_device_map)


gen_strategy = "greedy"  # @param ["greedy", "top_p"]
# @markdown Please select the prompt template if chat model is being used. For raw language model, please leave this field blank.
prompt_template = "openmoe"  # @param ["openmoe", ""]
max_new_tokens = 512  # @param {type:"slider", min:1, max:512, step:1}
debug_verbose = True  # @param {type:"boolean"}


# 2. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    pytorch_checkpoint_path, trust_remote_code=True)

# 3. Inference
gen_kwargs = {
    # Greedy Search
    "greedy": {"do_sample": False, "num_beams": 1, "max_new_tokens": max_new_tokens},
    # Top-p Sampling
    "top_p": {"do_sample": True, "temperature": 0.5, "top_p": 0.8, "max_new_tokens": max_new_tokens},
}


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./dataset/questions.jsonl",
                    help="MTBench数据集路径")
parser.add_argument("--output-dir", default="./mt_bench_output",
                    help="结果路径")
parser.add_argument("--score-mode", type=str, default="l1", help="层间对专家排序的指标")
# parser.add_argument("--prune-layer", type=int, default=-
#                     1, help="进行剪枝的层索引，-1为不对任何层剪枝")
# parser.add_argument("--num-expert", type=int, default=32, help="模型专家的总数")
# parser.add_argument("--num-layer", type=int, default=4, help="模型专家的总数")
parser.add_argument("--batch-size", type=int, default=4, help="并行解码的样本数量")
# parser.add_argument("--expert-idxs", type=str,
#                     help="进行剪枝的层使用专家的索引，多个专家用短横线分割，如1-6-9-11")

args = parser.parse_args()


# read benchmark
with open(args.input, 'r') as fp:
    questions = []
    for line in fp:
        line = line.strip()
        if line:
            question = json.loads(line)
            questions.append(question)
raw_questions = list(map(lambda x: x["turns"][0], questions))

# prune_layer = args.prune_layer
# expert_idxs = args.expert_idxs
batch_size = args.batch_size
output_path = args.output_dir
score_mode = args.score_mode

if pytorch_checkpoint_path == "OrionZheng/openmoe-base":
    num_layer = 3
    num_expert = 8
else:
    num_layer = 4
    num_expert = 32
print(f"{pytorch_checkpoint_path} num_layer {num_layer} num_expert {num_expert}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# prune layer idx and expert idx
if score_mode == "l1":
    prune_layer_tmp = [1]*5 + [2]*5
    expert_idxs_tmp = [
        '27', "27-20", "27-20-28-26", "27-20-28-26-10-19-7-25", "27-20-28-26-10-19-7-25-8-4-14-12-15-9-16-30",
        '24', "24-19", "24-19-17-25", "24-19-17-25-3-31-8-22", "24-19-17-25-3-31-8-22-15-21-12-18-27-5-2-29",
    ]
elif score_mode == "ww_alpha":
    prune_layer_tmp = [0]*5 + [1]*5 + [2]*5 + [3]*5
    expert_idxs_tmp = [
        '18',
        '18-5',
        '18-5-2-26',
        '18-5-2-26-4-20-15-30',
        '18-5-2-26-4-20-15-30-24-6-1-14-8-23-25-9',
        '20',
        '20-23',
        '20-23-0-13',
        '20-23-0-13-30-3-5-28',
        '20-23-0-13-30-3-5-28-7-4-31-21-9-22-16-19',
        '2',
        '2-17',
        '2-17-22-19',
        '2-17-22-19-6-13-7-10',
        '2-17-22-19-6-13-7-10-14-4-29-30-23-18-8-31',
        '24',
        '24-20',
        '24-20-23-22',
        '24-20-23-22-1-3-17-29',
        '24-20-23-22-1-3-17-29-18-13-2-16-31-19-28-25',
    ]
elif score_mode == "random":
    prune_layer_tmp = [0]*5 + [1]*5 + [2]*5 + [3]*5
    expert_idxs_tmp = []
    for _ in range(4):
        for num in [1, 2, 4, 8, 16]:
            expert_idxs = random.sample(range(32), num)
            expert_idxs_str = "-".join(list(map(str, expert_idxs)))
            expert_idxs_tmp.append(expert_idxs_str)

elif score_mode == 'test_route':
    prune_layer_tmp = [-1]
    expert_idxs_tmp = ["0"]
elif score_mode == "test_prune":
    prune_layer_tmp = [0]
    expert_idxs_tmp = ["0-1-3-4"]

print("prune layer: {}".format(prune_layer_tmp))
print("expert idxs: {}".format(expert_idxs_tmp))
# decode and eval ppl
for prune_layer, expert_idxs in zip(prune_layer_tmp, expert_idxs_tmp):
    expert_idxs = list(map(int, expert_idxs.split("-")))
    print("expert idxs {}".format(expert_idxs))

    expert_idxs_list.append(expert_idxs)
    prune_layer_list.append(prune_layer)
    layer_num_list.append(num_layer)

    # eval ppl on benchmark

    mean_ppl = compute_ppl(model, tokenizer, raw_questions, None)
    print("mean_ppl {}".format(mean_ppl))
    mean_ppl = mean_ppl.tolist()
    output = {"mean_ppl": mean_ppl}
    expert_idxs_str = expert_idxs
    model_id = "{}_pruneLayer{}_expert{}".format(
        model_name, prune_layer, expert_idxs_str)
    output_filename = "{}.json".format(model_id)
    output_filename = os.path.join(output_path, output_filename)
    json.dump(output, open(output_filename, 'w'))
