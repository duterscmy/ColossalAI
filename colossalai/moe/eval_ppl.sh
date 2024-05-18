#! /bin/bash

# test
# python eval_ppl.py --score-mode test_route

# prune by l1
python eval_ppl.py --score-mode l1 --output-dir output_l1

# prune by random
python eval_ppl.py --score-mode random --output-dir output_random

# prune by weightwatcher alpha
python eval_ppl.py --score-mode ww_alpha --output-dir output_ww_alpha

