### install
``` 
git clone --branch=main https://github.com/Orion-Zheng/t5x
python -m pip install ./t5x
git clone --branch no_route https://github.com/duterscmy/ColossalAI.git
pip install ./ColossalAI
python -m pip install -r ./ColossalAI/examples/language/openmoe/requirements.txt
``` 
### fixed expert decode

单个专家&随机选择4次： 
```
cd ./ColossalAI/colossalai/moe/
python decode.py --num-fixed-expert 1 --num-group 4 --num-expert 32
``` 
8个专家&随机选择4次（openmoe正常情况的专家数量topk为8）： 
```
python decode.py --num-fixed-expert 8 --num-group 4 --num-expert 32
``` 
结果位于：`./mt_bench_output`