#! /bin/bash

# no prune
python eval_ppl.py --prune-layer -1 --expert-idxs "0-1"

# prune last layer
python eval_ppl.py --prune-layer 3 --expert-idxs "7"
python eval_ppl.py --prune-layer 3 --expert-idxs "7-20"
python eval_ppl.py --prune-layer 3 --expert-idxs "7-20-17-15"
python eval_ppl.py --prune-layer 3 --expert-idxs "7-20-17-15-29-18-6-12"
#python eval_ppl.py --prune-layer 3 --expert-idxs "7-20-17-15-29-18-6-12-3-5-31-30-9-4-24-11"
#python eval_ppl.py --prune-layer 3 --expert-idxs "7-20-17-15-29-18-6-12-3-5-31-30-9-4-24-11-13-28-26-25-21-14-16-22-27-23-2-10-19-8"
#python eval_ppl.py --prune-layer 3 --expert-idxs "7-20-17-15-29-18-6-12-3-5-31-30-9-4-24-11-13-28-26-25-21-14-16-22-27-23-2-10-19-8-0-1"

# prune first layer
python eval_ppl.py --prune-layer 0 --expert-idxs "19"
python eval_ppl.py --prune-layer 0 --expert-idxs "19-12"
python eval_ppl.py --prune-layer 0 --expert-idxs "19-12-7-25"
python eval_ppl.py --prune-layer 0 --expert-idxs "19-12-7-25-3-13-23-18"
#python eval_ppl.py --prune-layer 0 --expert-idxs "19-12-7-25-3-13-23-18-5-10-26-29-24-21-6-27"
#python eval_ppl.py --prune-layer 0 --expert-idxs "19-12-7-25-3-13-23-18-5-10-26-29-24-21-6-27-15-28-8-22-4-20-11-17-31-30-9-2-16-14"
#python eval_ppl.py --prune-layer 0 --expert-idxs "19-12-7-25-3-13-23-18-5-10-26-29-24-21-6-27-15-28-8-22-4-20-11-17-31-30-9-2-16-14-0-1"