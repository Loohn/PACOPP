"""
Main Script for the comparative experiment of COPP, COPP-RS and PACOPP.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
import json
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from arguments import set_args
from utils import (
    GaussianPolicy,
    MlpPolicy,
    Reward,
    generate_data,
    train_policy_net,
    QuantileNet,
    train_quantile_net,
    ConditionalGaussianNet,
    train_cond_reward,
)
from conformal import (
    COPP,
    RejectionSampling,
    PACOPP,
)


def main(args):

    torch.manual_seed(args.seed)
    dim_x = args.dim_x
    behav_std = args.behavior_std
    target_std = args.target_std
    epsilon = args.epsilon
    delta = args.delta
    mc_num = args.mc_num

    # Setting up the Policies BEHAVIOURAL and TARGET
    if args.use_continuous_action_policy:
        behav_policy = GaussianPolicy(std=behav_std)
        target_policy = GaussianPolicy(std=target_std)
    else:
        raise NotImplementedError("Only continuous policy is implemented")
    
    # Setting up the Reward function
    reward_function = Reward(std=args.reward_std)
    if args.use_GMM_reward:
        reward_function = Reward(std=args.reward_std, mixture_weight=args.mixture_weight)

    #Generate training data from the behavioural policy
    x_train, a_train, y_train = generate_data(
        n=args.n_train,
        policy=behav_policy,
        reward_func=reward_function,
        x_dim=dim_x,
        x_std=args.x_std,
    )

    # Generate calibration dataset from the behavioural policy
    x_cal, a_cal, y_cal = generate_data(
        n=args.n_cal,
        policy=behav_policy,
        reward_func=reward_function,
        x_dim=dim_x,
        x_std=args.x_std,
    )

    # Generate test dataset from the target policy
    x_test, a_test, y_test = generate_data(
        n=args.n_test,
        policy=target_policy,
        reward_func=reward_function,
        x_dim=dim_x,
        x_std=args.x_std,
    )

    ##########  Behavior Policy Estimation  ###########
    ###################################################
    model_behav_policy = ConditionalGaussianNet(input_dim=dim_x)
    dataloader_xa = DataLoader(TensorDataset(x_train, a_train), batch_size=10)
    train_policy_net(model_behav_policy, dataloader_xa, epochs=100, lr=0.0001)
    behav_policy_hat = MlpPolicy(model_behav_policy)
    

    ################  Models for COPP  ################
    ###################################################
    logging.info("In COPP:")
    print("In COPP:")
    model_quantile_lower = QuantileNet(input_dim=1)
    model_quantile_upper = QuantileNet(input_dim=1)
    dataloader_xy = DataLoader(TensorDataset(x_train, y_train), batch_size=10)
    train_quantile_net(model_quantile_lower, dataloader_xy, quantile=epsilon/2, epochs=100, lr=0.0001)
    train_quantile_net(model_quantile_upper, dataloader_xy, quantile=1-epsilon/2, epochs=100, lr=0.0001)

    model_cond_reward = ConditionalGaussianNet(input_dim=dim_x+1)
    dataloader_xay = DataLoader(TensorDataset(x_train, a_train, y_train), batch_size=10)
    train_cond_reward(model_cond_reward, dataloader_xay, epochs=100, lr=0.0001)
    
    coverage_COPP, aveLength_COPP = COPP(
            quantile=1-epsilon,
            x_cal=x_cal,
            y_cal=y_cal, 
            x_test=x_test,
            y_test=y_test,
            target_policy=target_policy,
            behav_policy=behav_policy_hat,
            model_quantile_lower=model_quantile_lower, 
            model_quantile_upper=model_quantile_upper,
            model_cond_reward=model_cond_reward,
            grid_points=100,
            mc_num=mc_num
    )
    logging.info("COPP finished!")
    print("COPP finished!")


    ################  Models for PACOPP  ##############
    ###################################################
    logging.info("In PACOPP:")
    print("In PACOPP:")
    # Get data after RS procedure
    get_data_rs = RejectionSampling(behav_policy=behav_policy_hat, target_policy=target_policy)
    x_train_rs, y_train_rs = get_data_rs(x_train, a_train, y_train)
    x_cal_rs, y_cal_rs = get_data_rs(x_cal, a_cal, y_cal)

    # Train new quantile models from data_rs
    model_quantile_lower_rs = QuantileNet(input_dim=1)
    model_quantile_upper_rs = QuantileNet(input_dim=1)
    dataloader_xy_rs = DataLoader(TensorDataset(x_train_rs, y_train_rs), batch_size=10)
    train_quantile_net(model_quantile_lower_rs, dataloader_xy_rs, quantile=epsilon/2, epochs=100, lr=0.0001)
    train_quantile_net(model_quantile_upper_rs, dataloader_xy_rs, quantile=1-epsilon/2, epochs=100, lr=0.0001)

    coverage_PACOPP, aveLength_PACOPP, coverage_COPPRS, aveLength_COPPRS = PACOPP(
        epsilon=epsilon,
        delta=delta,
        x_cal=x_cal_rs,
        y_cal=y_cal_rs, 
        x_test=x_test,
        y_test=y_test,
        model_quantile_lower=model_quantile_lower_rs, 
        model_quantile_upper=model_quantile_upper_rs,
    )
    logging.info("PACOPP finished!")
    print("PACOPP finished!")

    return coverage_COPP, aveLength_COPP, coverage_PACOPP, aveLength_PACOPP, coverage_COPPRS, aveLength_COPPRS



logging.basicConfig(
    format="%(asctime)s|%(levelname)s|%(filename)s| %(message)s",
    filename="OPE_comparative.log",
    filemode="w",
    level=logging.INFO)

if __name__ == "__main__":
    args = set_args()

    ROOT = args.root
    n_simu = args.n_simu
    delta = args.delta

    folder_name = "results_of_comparative_experiment"
    folder_path = f"{ROOT}/{folder_name}"
    figure_path = f"{folder_path}/figures"

    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    results_file = f"{folder_path}/results.json"
    args_file = f"{folder_path}/args.json"
  
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp, indent=4)

    if os.path.exists(results_file):
        with open(results_file, "r") as fp:
            results = json.load(fp)
    else:
        results = {
            "torch.manual_seed": [None] * n_simu,
            "coverage_COPP": [None] * n_simu,
            "aveLength_COPP": [None] * n_simu,
            "coverage_PACOPP": [None] * n_simu,
            "aveLength_PACOPP": [None] * n_simu,
            "coverage_COPPRS": [None] * n_simu,
            "aveLength_COPPRS": [None] * n_simu
        }

    # Run the simulation for n_simu times
    for i in tqdm(range(n_simu)):
        if results["torch.manual_seed"][i] is not None:
            continue

        args.seed = i
        print("\nWe are in the ",i,"th simulation:")
        res = main(args)
        results["torch.manual_seed"][i] = args.seed
        results["coverage_COPP"][i] = res[0].item()
        results["aveLength_COPP"][i] = res[1].item()
        results["coverage_PACOPP"][i] = res[2].tolist()
        results["aveLength_PACOPP"][i] = res[3].tolist()
        results["coverage_COPPRS"][i] = res[4].item()
        results["aveLength_COPPRS"][i] = res[5].item()

        with open(results_file, "w") as fp:
            json.dump(results, fp, indent=4)
            

    # Load the results
    with open(results_file, "r") as fp:
        results = json.load(fp)

    data = []
    methods = ["COPP", "COPP-RS"]+ [f"PAC-{d}" for d in delta]
    for i in range(n_simu):
        data.append({"Method": "COPP", "Coverage": results["coverage_COPP"][i], "AveLength": results["aveLength_COPP"][i]})
        data.append({"Method": "COPP-RS", "Coverage": results["coverage_COPPRS"][i], "AveLength": results["aveLength_COPPRS"][i]})

        for j, d in enumerate(delta):
            data.append({
                "Method": f"PAC-{d}",
                "Coverage": results["coverage_PACOPP"][i][j],
                "AveLength": results["aveLength_PACOPP"][i][j]
            })

    df = pd.DataFrame(data)

    # Plot the results
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,           
        "axes.labelsize": 14,     
        "xtick.labelsize": 12,    
        "ytick.labelsize": 12,     
        "figure.figsize": (8, 6), 
        "savefig.bbox": "tight", 
    })

    # plot for Coverage
    plt.figure()
    sns.boxplot(x="Method", y="Coverage", data=df, order=methods, palette="Blues_r", hue="Method", legend=False,
                flierprops={"marker": "o", "markersize": 2})
    plt.axhline(y=1-args.epsilon, color="red", linestyle="--", linewidth=1)  
    plt.ylabel("Coverage", fontsize=14)
    plt.ylim(0.66, 0.94)
    plt.xlabel("")
    plt.grid(True, linestyle="-", alpha=0.7)
    plt.savefig(f"{figure_path}/coverage.pdf", format="pdf")
    plt.show()

    # plot for Average length
    plt.figure()
    sns.boxplot(x="Method", y="AveLength", data=df, order=methods, palette="Blues_r", hue="Method", legend=False,
                flierprops={"marker": "o", "markersize": 2})
    plt.ylabel("Average length", fontsize=14)
    plt.ylim(7, 14.9)
    plt.xlabel("")
    plt.grid(True, linestyle="-", alpha=0.7)
    plt.savefig(f"{figure_path}/average_length.pdf", format="pdf")
    plt.show()
