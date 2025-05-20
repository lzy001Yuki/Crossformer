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
deco_h1_mse = [0.303795725107193, 0.3452039062976837, 0.4509919285774231, 0.5559276938438416, 0.6074722409248352]
deco_h1_mae = [0.3652786910533905, 0.39096730947494507, 0.46108725666999817, 0.5284407734870911, 0.5767600536346436]
cross_h1_mse = [0.297194242477417, 0.35698017477989197, 0.47376549243927, 0.5866467952728271, 0.6980878114700317]
cross_h1_mae = [0.3498874008655548, 0.40031903982162476, 0.47370079159736633, 0.5507002472877502, 0.6360467076301575]
deco_m1_mse = [0.23721501231193542, 0.27673354744911194, 0.3329356610774994, 0.4212680160999298, 0.5891537070274353]
deco_m1_mae = [0.30652469396591187, 0.33476588129997253, 0.37372252345085144, 0.441145122051239, 0.5599108934402466]
cross_m1_mse = [0.2289648950099945, 0.2758387625217438, 0.3539612591266632, 0.47414496541023254, 0.9110997915267944]
cross_m1_mae = [0.3021090626716614, 0.3421657979488373, 0.3914526104927063, 0.4853478670120239, 0.6916109919548035]

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
deco_15_mse=[0.7819510698318481, 0.7009202837944031, 0.6422424912452698, 0.9340476393699646, 0.5324704051017761]
deco_15_mae=[0.6758888959884644, 0.6357977986335754, 0.5959328413009644, 0.7634934782981873, 0.5242634415626526]
deco_23_mse=[0.6555052399635315, 0.8125855922698975, 0.6632139682769775, 0.6682431697845459, 0.6180737614631653]
deco_23_mae=[0.6033016443252563, 0.6955882906913757, 0.606340765953064, 0.6136918663978577, 0.5779538750648499]
deco_25_mse=[0.6329577565193176, 0.6074722409248352, 0.7377627491950989, 0.7843104600906372, 0.5478439927101135]
deco_25_mae=[0.5937305688858032, 0.5767600536346436, 0.6440995335578918, 0.6803514361381531, 0.5337207913398743]
deco_45_mse=[0.8022421002388, 0.7914454936981201, 0.7765752673149109, 0.7836933732032776, 0.5986522436141968]
deco_45_mae=[0.6885672807693481, 0.6818612217903137, 0.674583375453949, 0.6773258447647095, 0.5683072209358215]
deco_mse={"15":deco_15_mse, "23": deco_23_mse, "25": deco_25_mse, "45": deco_45_mse}
deco_mae={"15":deco_15_mae, "23": deco_23_mae, "25": deco_25_mae, "45": deco_45_mae}
visual_trend("mse", std_output_mse, deco_mse)
visual_trend("mae", std_output_mae, deco_mae)

deco_15_mse_ = sum(deco_15_mse) / len(deco_15_mse)
deco_23_mse_ = sum(deco_23_mse) / len(deco_23_mse)
deco_25_mse_ = sum(deco_25_mse) / len(deco_25_mse)
deco_45_mse_ = sum(deco_45_mse) / len(deco_45_mse)
deco_15_mae_ = sum(deco_15_mae) / len(deco_15_mae)
deco_23_mae_ = sum(deco_23_mae) / len(deco_23_mae)
deco_25_mae_ = sum(deco_25_mae) / len(deco_25_mae)
deco_45_mae_ = sum(deco_45_mae) / len(deco_45_mae)
std_mse_ = sum(std_output_mse) / len(std_output_mse)
std_mae_ = sum(std_output_mae) / len(std_output_mae)
dwin_ = [15, 23, 25, 45]


def visual_avg(type, std, deco, x_axis):
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, deco, label="Decoformer", marker='o', linestyle='--', color='blue')
    plt.plot(x_axis, std, label="Crossformer", marker='s', linestyle='-', color='red')
    plt.title("{} loss of Decoformer and Crossformer".format(type), fontsize=14)
    plt.xlabel('dlinear window size', fontsize=12)
    plt.ylabel("{}_loss".format(type), fontsize=12)
    plt.legend()
    plt.savefig("pic/dwin_avg_{}.png".format(type))

std_mse_avg = [std_mse_, std_mse_, std_mse_, std_mse_]
std_mae_avg = [std_mae_, std_mae_, std_mae_, std_mae_]
deco_mse_avg = [deco_15_mse_, deco_23_mse_, deco_25_mse_, deco_45_mse_]
deco_mae_avg = [deco_15_mae_, deco_23_mae_, deco_25_mae_, deco_45_mae_]
visual_avg("mse", std_mse_avg, deco_mse_avg, dwin_)
visual_avg("mae", std_mae_avg, deco_mae_avg, dwin_)