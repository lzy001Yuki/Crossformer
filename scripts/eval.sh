#!/bin/bash
if [ "$1" -eq 1 ]; then
    dataset="ETTh1"
elif [ "$1" -eq 2 ]; then
    dataset="ETTm1"
fi
std_output="output/$dataset/result_out${2}_std.txt"
output="output/$dataset/result_out${2}.txt"

python src/main_crossformer.py --data $dataset --out_len $2 --seg_len 24  --patience 5 > $output
#python Crossformer/main_crossformer.py --data $dataset --out_len $2 --seg_len 24 > $std_output