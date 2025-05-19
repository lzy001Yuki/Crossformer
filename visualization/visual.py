import matplotlib.pyplot as plt


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
x_m1 = [24, 96, 288, 672]
deco_h1_mse = [0.303795725107193, 0.3452039062976837, 0.4509919285774231, 0.5746532082557678, 0.7708274722099304]
deco_h1_mae = [0.3652786910533905, 0.39096730947494507, 0.46108725666999817, 0.5228120684623718, 0.66136234998703]
cross_h1_mse = [0.297194242477417, 0.35698017477989197, 0.47376549243927, 0.6816306710243225, 0.9291953444480896]
cross_h1_mae = [0.3498874008655548, 0.40031903982162476, 0.47370079159736633, 0.6306871175765991, 0.7502655386924744]
deco_m1_mse = [0.23721501231193542, 0.3329356610774994, 0.4212680160999298, 0.5891537070274353]
deco_m1_mae = [0.30652469396591187, 0.37372252345085144, 0.441145122051239, 0.5599108934402466]
cross_m1_mse = [0.2289648950099945, 0.3539612591266632, 0.47414496541023254, 0.9110997915267944]
cross_m1_mae = [0.3021090626716614, 0.3914526104927063, 0.4853478670120239, 0.6916109919548035]

visual("ETTh1", x_h1, deco_h1_mse, cross_h1_mse, "mse")
visual("ETTh1", x_h1, deco_h1_mae, cross_h1_mae, "mae")
visual("ETTm1", x_m1, deco_m1_mse, cross_m1_mse, "mse")
visual("ETTm1", x_m1, deco_m1_mae, cross_m1_mae, "mae")