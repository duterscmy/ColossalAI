### install
``` 
git clone --branch=main https://github.com/Orion-Zheng/t5x
python -m pip install ./t5x
git clone --branch no_route https://github.com/duterscmy/ColossalAI.git
pip install ./ColossalAI
python -m pip install -r ./ColossalAI/examples/language/openmoe/requirements.txt
``` 
### eval ppl on mt bench
用于评估经过instruct finetuning的模型
step1: 需要先跑一次剪枝模型的评估脚本，会报一个assert error，该error发生在自动下载的hugging face模型代码中，没找到提前改好的方式： 
```
python eval_ppl.py --score-mode test_route
``` 
报错代码位置如：/root/.cache/huggingface/modules/transformers_modules/OrionZheng/openmoe-base/21955687339d4877654ed75cd0add4b8eb70eea4/modeling_openmoe.py", 可以得到缓存模型目录
用本项目中的脚本覆盖掉缓存模型中的该脚本：
```
mv modeling_openmoe.py ${tmp_model_path}
``` 

step2: 
```
sh eval_ppl.sh
``` 
结果位于./output_l1  ./output_random  ./output_ww_alpha

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