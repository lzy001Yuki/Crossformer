# DecoFormer:Enhancing CrossFormer's Performance through DLinear Integration

**Decoformer** mainly addresses Time Series Forecasting(TSF) problem based on [CrossFormer](https://github.com/Thinklab-SJTU/Crossformer) (ICLR2023). 
Considering that TSF is time-related, 
whereas Crossformer's design of Cross-Time Stage(CTS) applies time-invariant Multi-Attention 
Mechanism, this work removes CTS and integrates another model called Decomposition Linear 
(also noted as *Dlinear*). The resulting hybrid architecture shows stronger forecasting accuracy than Crossformer in long-term prediction with fixed input length on two real-world benchmark. Moreover, a 1.25x speedup is achieved due to the removal of the CTS module while maintaining performance.

![](Decoformer.png)


## Project Structure

```plain
├── README.md
├── datasets                # datasets used in the project
│   ├── ETTh1.csv
│   └── ETTm1.csv
├── output                  # outputs of Decoformer in experiments
│   ├── ETTh1
│   └── ETTm1
├── paper.pdf        
├── pic                     # visualized pictures
├── scripts
├── src                     # source code of Decoformer
└── visualization
```

## Run the Project

```
git clone https://github.com/Thinklab-SJTU/Crossformer # clone code for baseline model Crossformer
bash scripts/eval.sh <datasets_id> <out_length> <in_length> <seg_length> <dwin_size> <learning_rate> <e_layers> <iteration_num>
```

Parameters explanation:

1) `datasets_id`: 1 for `ETTh1`, 2 for `ETTm1`
2) `in_length`: expected input length, in paper, we choose 96
3) `out_length`: prediction output length
4) `seg_length`: length of segment in dsw_embedding phase
5) `dwin_size`: stride to form seasonal feature
6) `learning_rate`: initial learning_rate
7) `e_layers`: layers of encoder during HED structure
8) `iteration_num`: experiment times

After experimenting, these parameters are suggested as follows:

| parameter          | short_term      | long_term |
|:-------------------|:----------------|:----------|
| out_length (EETh1) | 24, 48, 168     | 336, 720  |
|out_length (ETTm1)| 24, 48, 96, 288 | 672       |
| seg_length         | 6               | 24        |
| dwin_size          | 23/25      | 23/25|
| learning_rate      | 1e-4            | 1e-5      |
| e_layers           | 3               | 2         |

