### install
``` 
git clone --branch=main https://github.com/Orion-Zheng/t5x
python -m pip install ./t5x
git clone --branch no_route https://github.com/duterscmy/ColossalAI.git
pip install ./ColossalAI
python -m pip install -r ./ColossalAI/examples/language/openmoe/requirements.txt
``` 
### mt bench
用于评估经过instruct finetuning的模型
最多8个专家&随机选择4次（openmoe正常情况的专家数量topk为8）： 
```
python decode.py --num-fixed-expert 8 --num-group 4 --num-expert 32
``` 
默认输出结果位于：`./mt_bench_output`

### wmt16 en-ro
用于评估base模型
最多8个专家&随机选择4次（openmoe正常情况的专家数量topk为8）： 
```
python eval_benchmark/decode_wmt_enro.py --model-name OrionZheng/openmoe-8b --num-fixed-expert 8 --num-group 4
``` 
默认输出结果位于：`./outputs/wmt16_enro`