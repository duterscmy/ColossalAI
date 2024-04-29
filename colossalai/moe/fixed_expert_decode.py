# -*- coding: utf-8 -*-
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, T5Tokenizer, AutoConfig, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from typing import List, Optional
from huggingface_hub import snapshot_download
from absl import flags
import random
import os
import json
import time

from colossalai.moe.expert_idx import expert_idxs


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


def inference(model, tokenizer, input_strs, gen_kwargs,
              add_special_tokens=True, split_special_tokens=False, output_only=True, verbose=False):

    model = model.eval()

    # Tokenization
    inputs = tokenizer.batch_encode_plus(input_strs,
                                         padding='longest',
                                         add_special_tokens=add_special_tokens,
                                         split_special_tokens=split_special_tokens,
                                         return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    base_len = inputs.input_ids.size(-1)
    # if verbose:
    #     print("Input Tokens:\n", input_ids)
    #     print("Num of Input Tokens: ", base_len)
    #     print("Attention Mask:\n", attention_mask)
    logits_processor = LogitsProcessorList(
        [StopAfterEosTextGenerated(base_len, tokenizer.eos_token_id)])

    output_ids = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                bos_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                logits_processor=logits_processor,
                                **gen_kwargs)
    if output_only:  # Only preserve output tokens
        output_ids = output_ids[:, input_ids.size(1):]
    # if verbose:
    #     print("Generated Tokens:\n", output_ids)
    output_txts = tokenizer.batch_decode(output_ids,
                                         clean_up_tokenization_spaces=True,
                                         skip_special_tokens=False)
    return output_ids, output_txts


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


def fixed_expert_decode(input_texts, prompt="", model_name="OrionZheng/openmoe-8b-chat", num_fixed_expert=4, num_group=4, num_expert=32,
                        batch_size=4, max_new_tokens=32):
    '''固定选择n个专家来得到推理结果'''
    # print(input_texts[:5])
    # @markdown 1. Path to the checkpoint repo
    pytorch_checkpoint_path = model_name  # @param {type:"string"}
    # pytorch_checkpoint_path = "OrionZheng/openmoe-base"
    model_name = pytorch_checkpoint_path.split("/")[-1]
    # @markdown 2. (If any)Specify GPUs you want to use.
    # @markdown
    # @markdown - If single GPU memory is not enough, you can enter ids of multiple GPUs(seperated by comma). During inference, GPUs will be filed up sequentially.
    # @param ["", "0", "0,1", "0,1,2"] {allow-input: true}
    available_gpu_ids_str = "0"
    # @markdown - Specify available memory of each GPU
    # @markdown   - Leave some margin for data and activation.
    # @markdown For example, we used 38GB GPU memory for an A100(40GB)
    memory_per_gpu = "38GiB"  # @param ["", "38GiB"] {allow-input: true}
    # @markdown 3. Specify available CPU RAM
    # @markdown
    # @markdown - The Colab CPU High-RAM Runtime has 51GiB RAM
    cpu_memory = '50GiB'  # @param ["50GiB"] {allow-input: true}
    # @markdown 3. Specify the model parameter's precision

    # @markdown - The CPU runtime only supports inference in float32 precision
    # @markdown - The `bfloat16` is only available on A100 Colab runtime
    # @markdown - Please use float32/bfloat16 for inference. We observed issues with the model output when running in float16, which may be due to underflow caused by our large vocabulary size.
    model_dtype = 'float32'  # @param ["float32", "bfloat16"]
    # @markdown (Not recommended, very slow) Offload model weights to CPU memory if GPU's is insufficient, then offload to disk if CPU memory is insufficient.
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
                                             dtype=eval(
                                                 f'torch.{model_dtype}'),
                                             no_split_module_classes=[no_split_module_classes])
    else:
        model = load_checkpoint_and_dispatch(model, weights_location,
                                             device_map=device_map,
                                             dtype=eval(
                                                 f'torch.{model_dtype}'),
                                             no_split_module_classes=[no_split_module_classes])
    print('Fine-grained Device Map: \n', model.hf_device_map)

    gen_strategy = "greedy"  # @param ["greedy", "top_p"]
    max_new_tokens = max_new_tokens  # @param {type:"slider", min:1, max:512, step:1}
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

    name_to_outputs = {}
    for group_idx in range(num_group):  # 随机选择n次专家
        for _num_expert in range(num_expert):
            _num_expert += 1
            random_numbers = random.sample(range(_num_expert), num_fixed_expert)
            expert_idxs.append(random_numbers)
            print("using expert: {}".format(random_numbers))

            expert_idxs_str = ";".join(list(map(str, random_numbers)))
            model_id = "{}_expert{}".format(model_name, expert_idxs_str)

            outputs = []
            for sample_idx in range(0, len(input_texts), batch_size):
                end_sample_idx = min(len(input_texts), sample_idx + batch_size)
                input_strs = input_texts[sample_idx: end_sample_idx]

                final_input_strs = []
                for input_str in input_strs:
                    # print("input str: {}".format(input_str))
                    # SYS_LLAMA = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
                    # input_str = apply_llama_chat_template(tokenizer,
                    #                                       [(input_str, 'user')],
                    #                                       sys_prompt=prompt)
                    input_str = prompt + input_str
                    final_input_strs.append(input_str)
                # print("=========== The Actual Input =============")
                # [print(i) for i in final_input_strs]
                output_ids, output_txts = inference(model, tokenizer, final_input_strs, gen_kwargs[gen_strategy],
                                                    verbose=debug_verbose)

                # print(output_txts)
                print("============== Output Text ===============")
                for i, output_txt in enumerate(output_txts):
                    print(f"sample {i}: {final_input_strs[i]}")
                    output = output_txt.split('</s>')[0]
                    print("output: {}\n".format(output))
                    outputs.append(output)
            assert len(outputs) == len(input_texts)
            name_to_outputs[model_id] == outputs
        return name_to_outputs
