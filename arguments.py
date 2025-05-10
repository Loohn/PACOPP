import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'y', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def probability(x):
    """make sure x in (0,1)"""
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: {x}")
    
    if x <= 0 or x >= 1:
        raise argparse.ArgumentTypeError(f"Value must be in the range (0,1), but got {x}")
    
    return x 


def set_args():
    parser = argparse.ArgumentParser(description="Parser for the OPE experiments")
    parser.add_argument(
        "--seed", type=int, default=1, help="seed for reproducing"
    )
    parser.add_argument(
        "--root", default=".", type=str, help="Root folder of repository"
    )
    parser.add_argument(
        "--n_train", type=int, default=1000, help="Number of training points for model"
    )
    parser.add_argument(
        "--n_cal", type=int, default=1000, help="Number of calibration points"
    )
    parser.add_argument(
        "--n_test", type=int, default=10000, help="Number of test points"
    )
    parser.add_argument(
        "--n_simu", type=int, default=1000, help="Number of simulations"
    )
    parser.add_argument(
        "--epsilon", type=probability, default=0.20, help="Nominal coverage error, value must be in (0,1)"
    )
    parser.add_argument(
        "--delta", type=probability, nargs="+", default=[0.5, 0.25, 0.1, 0.01], help="Outer probability for PAC validity, values must be in (0,1)"
    )

    ###############################################################################
    parser.add_argument(
        "--use_continuous_action_policy",
        type=str2bool,
        default=True,
        help="Use continuous action policy for behaviour and target policies",
    )
    parser.add_argument(
        "--x_std", type=float, default=2.0, help="Standard deviation of the context distribution"
    )
    parser.add_argument(
        "--behavior_std", type=float, default=2.0, help="Standard deviation of the behavior policy"
    )
    parser.add_argument(
        "--target_std", type=float, default=1.0, help="Standard deviation of the target policy"
    )
    parser.add_argument(
        "--reward_std", type=float, default=4.0,  help="Standard deviation of the conditional reward"
    )
    parser.add_argument(
        "--dim_x", type=int, default=1,  help="Dimention of the context x"
    )
    parser.add_argument(
        "--use_GMM_reward",
        type=str2bool,
        default=True,
        help="use mixture Gaussian reward",
    )
    parser.add_argument(
        "--mixture_weight", type=probability, default=0.2, help="Weight of the small variance Gaussian in the mixture"
    )
    parser.add_argument(
        "--mc_num",
        type=int,
        default=100,
        help="number MC samples for the cond no lop method",
    )

    args = parser.parse_args()
    return args