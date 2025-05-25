import matplotlib.pyplot as plt
import random


def visual(datasets, x_axis, deco_res, cross_res, type):
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, deco_res, label="Decoformer", marker='o', linestyle='-', color='blue')
    plt.plot(x_axis, cross_res, label="Crossformer", marker='s', linestyle='--', color='red')
    plt.title("{} loss of Decoformer and Crossformer in {}".format(type, datasets), fontsize=14)
    plt.xlabel('prediction length', fontsize=12)
    plt.ylabel("{}_loss".format(type), fontsize=12)
    plt.legend()
    plt.savefig("pic/{}_{}.png".format(datasets, type))

# experimental results
x_h1 = [24, 48, 168, 336, 720]
x_m1 = [24, 48, 96, 288, 672]
deco_h1_mse = [0.303795725107193, 0.3452039062976837, 0.4299757480621338, 0.4818570911884308, 0.5112711787223816]
deco_h1_mae = [0.3652786910533905, 0.39096730947494507, 0.4414631724357605, 0.4739294648170471, 0.5180508494377136]
cross_h1_mse = [0.297194242477417, 0.35698017477989197, 0.47376549243927, 0.5866467952728271, 0.6810809969902039]
cross_h1_mae = [0.3498874008655548, 0.40031903982162476, 0.47370079159736633, 0.5507002472877502, 0.6173112988471985]
deco_m1_mse = [0.23721501231193542, 0.27673354744911194, 0.3329356610774994, 0.4212680160999298, 0.5689398646354675]
deco_m1_mae = [0.30652469396591187, 0.33476588129997253, 0.37372252345085144, 0.441145122051239, 0.5422918200492859]
cross_m1_mse = [0.2289648950099945, 0.2758387625217438, 0.3539612591266632, 0.47414496541023254, 0.665247917175293]
cross_m1_mae = [0.3021090626716614, 0.3421657979488373, 0.3914526104927063, 0.4853478670120239, 0.6061616539955139]

visual("ETTh1", x_h1, deco_h1_mse, cross_h1_mse, "mse")
visual("ETTh1", x_h1, deco_h1_mae, cross_h1_mae, "mae")
visual("ETTm1", x_m1, deco_m1_mse, cross_m1_mse, "mse")
visual("ETTm1", x_m1, deco_m1_mae, cross_m1_mae, "mae")

def visual_trend(type, std_output, deco_output):
    plt.figure(figsize=(8,6))
    x_iter=[1,2,3,4,5]
    plt.plot(x_iter, std_output, label="{} of crossformer (output length=720)".format(type), marker='o', linestyle='-', color='blue')
    i = 0
    all_color = ["black", "pink", "red", "green"]
    for key, value in deco_output.items():
        plt.plot(x_iter, value, label="{} of decoformer (dwin_size={})".format(type, key), marker='s', linestyle='--', color=all_color[i])
        i += 1
    plt.title("{} loss of Decoformer and Crossformer with different dlinear window size".format(type), fontsize=14)
    plt.xlabel('iteration', fontsize=12)
    plt.ylabel("{}_loss".format(type), fontsize=12)
    plt.legend()
    plt.savefig("pic/trend_{}.png".format(type))
        
    
    
std_output_mse=[0.6896241903305054, 0.681848406791687, 0.6653105616569519, 0.7062484622001648, 0.6980878114700317]
std_output_mae=[0.6316485404968262, 0.6309805512428284, 0.6189975142478943, 0.6407160758972168, 0.6360467076301575]
deco_15_mse=[0.49936643242836, 0.5012012124061584, 0.5035037398338318, 0.49781814217567444, 0.5140438675880432]
deco_15_mae=[0.5084335803985596, 0.510301411151886, 0.5115759372711182, 0.5076297521591187, 0.5198558568954468]
deco_23_mse=[0.49579864740371704, 0.5055443644523621, 0.4953390061855316, 0.49265357851982117, 0.49655818939208984]
deco_23_mae=[0.5052262544631958, 0.5136348009109497, 0.5046067833900452, 0.5034315586090088, 0.5063378214836121]
deco_25_mse=[0.49608805775642395, 0.5102717280387878, 0.5020611882209778, 0.5019807815551758, 0.5015705823898315]
deco_25_mae=[0.5057275295257568, 0.515655517578125, 0.5115998387336731, 0.5104116201400757, 0.5102278590202332]
deco_75_mse=[0.5019629001617432, 0.5001793503761292, 0.4967728555202484, 0.5005844831466675, 0.49854278564453125]
deco_75_mae=[0.5098021030426025, 0.5083468556404114, 0.505790650844574, 0.5101684927940369, 0.5072991847991943]
deco_mse={"15":deco_15_mse, "25": deco_25_mse, "75": deco_75_mse, "105": deco_23_mse}
deco_mae={"15":deco_15_mae, "25": deco_25_mae, "75": deco_75_mae, "105": deco_23_mae}
visual_trend("mse", std_output_mse, deco_mse)
visual_trend("mae", std_output_mae, deco_mae)

deco_15_mse_ = sum(deco_15_mse) / len(deco_15_mse)
deco_105_mse_ = sum(deco_23_mse) / len(deco_23_mse)
deco_25_mse_ = sum(deco_25_mse) / len(deco_25_mse)
deco_75_mse_ = sum(deco_75_mse) / len(deco_75_mse)
deco_15_mae_ = sum(deco_15_mae) / len(deco_15_mae)
deco_105_mae_ = sum(deco_23_mae) / len(deco_23_mae)
deco_25_mae_ = sum(deco_25_mae) / len(deco_25_mae)
deco_75_mae_ = sum(deco_75_mae) / len(deco_75_mae)
deco_215_mse_ = 0.5019629001083496
deco_215_mae_ = 0.5098021030427153
std_mse_ = sum(std_output_mse) / len(std_output_mse)
std_mae_ = sum(std_output_mae) / len(std_output_mae)
dwin_ = [15, 25, 75, 105, 215]


def visual_avg(type, std, deco, x_axis):
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, deco, label="Decoformer", marker='o', linestyle='--', color='blue')
    #plt.plot(x_axis, std, label="Crossformer", marker='s', linestyle='-', color='red')
    plt.title("{} loss of Decoformer".format(type), fontsize=14)
    plt.xlabel('dlinear window size', fontsize=12)
    plt.ylabel("{}_loss".format(type), fontsize=12)
    plt.legend()
    plt.savefig("pic/dwin_avg_{}.png".format(type))

std_mse_avg = [std_mse_, std_mse_, std_mse_, std_mse_]
std_mae_avg = [std_mae_, std_mae_, std_mae_, std_mae_]
deco_mse_avg = [deco_15_mse_, deco_25_mse_, deco_75_mse_, deco_105_mse_, deco_215_mse_]
deco_mae_avg = [deco_15_mae_, deco_25_mae_, deco_75_mae_, deco_105_mae_, deco_215_mae_]
visual_avg("mse", std_mse_avg, deco_mse_avg, dwin_)
visual_avg("mae", std_mae_avg, deco_mae_avg, dwin_)