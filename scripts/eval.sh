#!/bin/bash
if [ "$1" -eq 1 ]; then
    dataset="ETTh1"
elif [ "$1" -eq 2 ]; then
    dataset="ETTm1"
fi

python src/main_crossformer.py --data $dataset --out_len $2 --in_len $3 --seg_len $4 --dwin_size $5  --learning_rate $6 --e_layers $7 --itr $8

python Crossformer/main_crossformer.py --data $dataset --out_len $2 --in_len $3 --seg_len $4 --learning_rate $6 --itr $8
