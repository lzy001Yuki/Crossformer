#!/bin/bash
if [ "$1" -eq 1 ]; then
    dataset="ETTh1"
elif [ "$1" -eq 2 ]; then
    dataset="ETTm1"
fi

curdir="output/$dataset"

if [ "$1" -eq 1 ]; then
    if [ "$2" -eq 1 ]; then
        echo "experimenting on fixed_in_diff_out"
        mkdir -p "$curdir/in_${3}"
        s_steps=(24 48 168)
        for step in "${s_steps[@]}"; do 
            out_dir="$curdir/in_${3}/result_out${step}.txt"
            std_out_dir="$curdir/in_${3}/result_out${step}_std.txt"
            echo "expermenting on outlen $step"
            python src/main_crossformer.py --data $dataset --in_len $3 --out_len $step --seg_len 6  --itr 5  > $out_dir
            python Crossformer/main_crossformer.py --data $dataset  --in_len $3 --out_len $step --seg_len 6 --itr 5 > $std_out_dir
        done
        l_steps=(336 720)
        for step in "${l_steps[@]}"; do 
            out_dir="$curdir/in_${3}/result_out${step}.txt"
            std_out_dir="$curdir/in_${3}/result_out${step}_std.txt"
            echo "expermenting on outlen $step"
            python src/main_crossformer.py --data $dataset --in_len $3 --out_len $step --seg_len 24  --itr 5  --learning_rate 1e-5 --e_layers 2 > $out_dir
            python Crossformer/main_crossformer.py --data $dataset  --in_len $3 --out_len $step --seg_len 24 --itr 5 --learning_rate 1e-5 > $std_out_dir
        done
    elif [ "$2" -eq 2 ]; then
        echo "experimenting on dwin_size"
        python Crossformer/main_crossformer.py --data $dataset  --in_len 96 --out_len 720 --seg_len 24 --itr 5 --learning_rate 1e-5 > "$curdir/dwin_exp/result_dwin_std.txt"
        wins=(15 25 35 45)
        for win in "${wins[@]}"; do 
            echo "expermenting on dwin_size $win"
            out_dir="$curdir/dwin_exp/result_dwin${win}.txt"
            python src/main_crossformer.py --data $dataset --in_len 96 --out_len 720 --seg_len 24   --itr 5  --learning_rate 1e-5 --e_layers 2 > $out_dir
        done
    fi

elif [ "$1" -eq 2 ]; then
    echo "expermenting on ETTm1"
    
fi


#output="output/$dataset/result_out${2}_in${3}_dwin${4}.txt"
#python src/main_crossformer.py --data $dataset --out_len $2 --in_len $3 --seg_len 24 --dwin_size $4 --learning_rate 1e-5 --itr $5 --e_layers 2 >> $output


#std_output="output/$dataset/result_out${2}_in${3}_std.txt"
#python Crossformer/main_crossformer.py --data $dataset --out_len $2 --in_len $3 --seg_len 24 --learning_rate 1e-5 --itr $5 > $std_output


#python src/main_crossformer.py --data "ETTh1"  --seg_len 24 --learning_rate 1e-5  --itr 5 --out_len 720 --dwin_size 15 > "output/ETTh1/dwin_exp/result_out720_dwin15.txt"