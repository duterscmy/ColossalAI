import pandas as pd
import argparse
import os
import random

from colossalai.moe.fixed_expert_decode import fixed_expert_decode
from colossalai.moe.expert_idx import expert_idxs


parser = argparse.ArgumentParser()
parser.add_argument("--num-fixed-expert", type=int,
                    default=1, help="推理时使用专家的数量")
parser.add_argument("--num-group", type=int, default=4, help="随机选择专家的次数")
parser.add_argument("--num-expert", type=int, default=32, help="模型专家的总数")
parser.add_argument("--batch-size", type=int, default=1, help="并行解码的样本数量")
parser.add_argument("--max-new-tokens", type=int, default=32, help="解码结果的最大长度")
parser.add_argument("--model-name", type=str, default="OrionZheng/openmoe-8b-chat",
                    choices=["OrionZheng/openmoe-8b-chat", "OrionZheng/openmoe-8b",
                             "OrionZheng/openmoe-34b-200B", "OrionZheng/openmoe-base",],
                    help="模型在huggingface上的命名")
parser.add_argument("--input", default="./dataset/wmt16-en-ro.parquet",
                    help="wmt16 en-ro test数据集路径")
parser.add_argument("--output-dir", default="./outputs/wmt16_enro",
                    help="结果路径")
args = parser.parse_args()


num_fixed_expert = args.num_fixed_expert
num_group = args.num_group
num_expert = args.num_expert
batch_size = args.batch_size
output_path = args.output_dir
model_name = args.model_name
max_new_tokens = args.max_new_tokens

if not os.path.exists(output_path):
    os.makedirs(output_path)

dataset = pd.read_parquet(args.input, engine='fastparquet')
en_texts = list(dataset["translation.en"])
prompt = "translate English to Romanian:"

name_to_translations = fixed_expert_decode(
    en_texts, prompt, model_name, num_fixed_expert, num_group, num_expert, batch_size, max_new_tokens)

for name, translations in name_to_translations.items():
    dataset[name] = translations

dataset.to_csv(os.path.join(output_path, "wmt16_enro.output.csv"))
