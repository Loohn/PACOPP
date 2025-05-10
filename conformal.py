import torch
from tqdm import tqdm

from scipy.stats import binom
import numpy as np
import math


def get_conformal_scores(
    x, y, model_quantile_lower, model_quantile_upper
):
    """Compute the conformal scores as max{q_lower(x) - y, y - q_upper(x)}
    Args:
        x, y (torch.Tensor) | size m x 1
        model_quantile_lower, model_quantile_upper | (torch model)
    Returns:
        scores (torch.Tensor) | size m x 1
    """
    with torch.no_grad():
        quantiles_lower = model_quantile_lower(x)
        quantiles_upper = model_quantile_upper(x)
        scores = torch.max(y - quantiles_upper, quantiles_lower - y)
    return scores



def get_threshold_scores(scores, weights,quantile):
    """Compute the empirical 1-alpha quantile of weighted conformal scores

    Args:
        scores (torch.Tensor): scores of the calibration data | size N x 1
        weights (torch.Tensor): weights for the calibratio data | size N x M: M=n_test sets of weights
        quantile (float): quantile we would like to achieve | = 1-alpha 

    Returns:
        Q(1-alpha, F): 1-alpha quantile for each weights | size M x 1
    """
    sorted_scores, sorted_indices = torch.sort(scores, dim=0)  # order of scores | (N, 1)
    sorted_weights = weights[sorted_indices.squeeze(), :]  # ordered weights | (N, M)
    cumsum_weights = torch.cumsum(sorted_weights, dim=0)  # cumcumulative sum of weights | (N, M)

    exceed_indices = cumsum_weights >= quantile  #  which >= 1-alpha | (N, M) Bool
    threshold_scores = torch.full((weights.shape[1], 1), torch.inf, dtype=scores.dtype)  # initialize: inf
    valid_columns = exceed_indices.any(dim=0)  # which column exceeds 1-alpha | (M,)

    if valid_columns.any():  # if at least 1 column
        # the first exceeds 1-alpha | (M,)
        first_exceed_positions = torch.argmax(exceed_indices[:, valid_columns].to(dtype=torch.int64), dim=0) 
        threshold_scores[valid_columns, 0] = sorted_scores[first_exceed_positions, 0]  # (M, 1)

    return threshold_scores  # (M, 1)



class ConditionalDensity_MC_est():
    """ 
    Monte Calro estimation of conditional density P(y|x) under a policy pi
    P(y|x) ~= sum_k hat{P}(y|x,a_k), a_k ~ pi(.|x) 
    """
    def __init__(self, policy, model_cond_reward, mc_num=200, batch_size=64):
        self.policy = policy
        self.model_cond_reward = model_cond_reward
        self.mc_num = mc_num
        self.batch_size = batch_size

    def __call__(self, x, y):
        # x: N*1, y: N*1
        batches = int(self.mc_num/self.batch_size)
        likelihood = torch.zeros_like(y)
        x = x.unsqueeze(1).repeat(1, self.batch_size, 1) # x: N,1 -> N,1,1 -> N,64,1 
        x = x.reshape(-1, x.shape[-1])  # N*64,1
        y = y.unsqueeze(1).repeat(1, self.batch_size, 1)
        y = y.reshape(-1, y.shape[-1])
        for batch in range(batches):
            a_sampled = self.policy.sample(x)
            with torch.no_grad():
                mean_pred, std_pred = self.model_cond_reward(torch.cat([x, a_sampled], dim=-1))

                normal_dist = torch.distributions.Normal(mean_pred.squeeze(-1), std_pred.squeeze(-1))
                likelihood_vals = torch.exp(normal_dist.log_prob(y.squeeze(-1)))
                
                likelihood_vals = likelihood_vals.reshape(-1, self.batch_size, y.shape[1])
                likelihood += likelihood_vals.sum(dim=1)  # sum over all actions

        return likelihood / (batches * self.batch_size)



def COPP(
        quantile,
        x_cal, 
        y_cal, 
        x_test,
        y_test,
        target_policy,
        behav_policy,
        model_quantile_lower, 
        model_quantile_upper,
        model_cond_reward,
        grid_points,
        mc_num
):
    # MC estimate of cond_densities under target and behavior policies
    target_cond_density = ConditionalDensity_MC_est(
        policy=target_policy,
        model_cond_reward=model_cond_reward,
        mc_num=mc_num
    )
    behav_cond_density = ConditionalDensity_MC_est(
        policy=behav_policy,
        model_cond_reward=model_cond_reward,
        mc_num=mc_num
    )
    
    # ratios of cond_densities for cal and test data 
    ratios_cal = target_cond_density(x_cal, y_cal) / (behav_cond_density(x_cal, y_cal) + 1e-12)  # (n_cal,1)
    ratios_test = target_cond_density(x_test, y_test) / (behav_cond_density(x_test, y_test) + 1e-12) # (n_test,1)
    # weights for the calibration scores
    weights = ratios_cal / (ratios_test.T + ratios_cal.sum())  # (n_cal, n_test)
    scores_cal = get_conformal_scores(x_cal, y_cal, model_quantile_lower, model_quantile_upper)  # (n_cal, 1)    
    # Quantile(F, 1-alpha)
    threshold_scores = get_threshold_scores(scores=scores_cal, weights=weights, quantile=quantile)
    # compute the coverage by checking s(x_test, y_test) <=  threshold_scores
    scores_test = get_conformal_scores(x_test, y_test, model_quantile_lower, model_quantile_upper)
    coverage = (scores_test <= threshold_scores).sum() / (scores_test.shape[0])
    # print(f'coverage={coverage}')

    # average length
    y_grid = torch.linspace(y_test.min(), y_test.max(), steps=grid_points) # (gp,)
    grid_len = ( y_test.max() - y_test.min() ) / (grid_points-1)
    counts_grid = 0 # count the points s.t. s(x_test, y_test_grid) <=  threshold_scores_grid
    for y_grid_value in tqdm(y_grid):
        y_test_grid = torch.ones_like(y_test) * y_grid_value
        ratios_test_grid = target_cond_density(x_test, y_test_grid) / (behav_cond_density(x_test, y_test_grid) + 1e-12) # (n_test,1)
        weights_grid = ratios_cal / (ratios_test_grid.T + ratios_cal.sum())  # (n_cal, n_test)
        threshold_scores_grid = get_threshold_scores(scores=scores_cal, weights=weights_grid, quantile=quantile)
        scores_test_grid = get_conformal_scores(x_test, y_test_grid, model_quantile_lower, model_quantile_upper)
        counts_grid += (scores_test_grid <= threshold_scores_grid).sum()
    
    aveLength = counts_grid / x_test.shape[0] * grid_len

    return coverage, aveLength


class RejectionSampling():
    """RS procedure"""
    def __init__(self, behav_policy, target_policy):
        self.behav_policy = behav_policy
        self.target_policy = target_policy

    def __call__(self, x: torch.Tensor, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weights = self.target_policy.density(x,a) / self.behav_policy.density(x,a)
        weights = weights.squeeze(1)
        V = torch.rand(x.shape[0]) * weights.max()
        return x[V <= weights].unsqueeze(1), y[V <= weights].unsqueeze(1)  # (N_rs,1)
    


def find_k(M, epsilon, delta):
    """ k = argmax k in [-1,...,M-1] s.t. Binom(M,epsilon)(k)<=delta 
    Return k | list, len(k)=len(delta)
    """
    binomcdf_values = binom.cdf(np.arange(-1, M), M, epsilon) 
    k = []
    for d in delta:
        k_d = np.max(np.where(binomcdf_values <= d)) - 1   # in {-1,...,M-1}
        k.append(int(k_d))
    return k


def PACOPP(
        epsilon,
        delta,
        x_cal,
        y_cal,
        x_test,
        y_test,
        model_quantile_lower, 
        model_quantile_upper,
):
    if not isinstance(delta, list):
        delta = [delta]
    
    scores_cal = get_conformal_scores(x_cal, y_cal, model_quantile_lower, model_quantile_upper)
    scores_cal_sorted= torch.sort(scores_cal,dim=0).values # (M,1)

    # get k(M,epsilon,delta) for each delta in delta[], M = x_cal.shape[0] 
    k = find_k(scores_cal.shape[0], epsilon, delta)  # list | len(k) = len(delta)

    threshold_scores = torch.full((len(delta),), torch.inf, dtype=scores_cal.dtype)
    for i in range(len(k)):
        if 0 <= k[i]:
            threshold_scores[i] = scores_cal_sorted[-k[i]-1, 0]  # the (M-k)-th smallest scores_cal

    with torch.no_grad():
        quantiles_lower_test = model_quantile_lower(x_test)
        quantiles_upper_test = model_quantile_upper(x_test)
    
    coverage_PACOPP = torch.zeros(len(delta))
    aveLength_PACOPP = torch.zeros(len(delta))
    for i in range(len(delta)):
        in_PI = (y_test <= quantiles_upper_test + threshold_scores[i]) & (y_test >= quantiles_lower_test - threshold_scores[i])
        coverage_PACOPP[i] =  torch.sum(in_PI) / x_test.shape[0]
        aveLength_PACOPP[i] = torch.mean(quantiles_upper_test - quantiles_lower_test) + 2 * threshold_scores[i]

    # COPPRS
    k_COPPRS = math.ceil((scores_cal.shape[0] + 1 ) * (1 - epsilon))
    if k_COPPRS <= scores_cal.shape[0]:
        threshold_score_COPPRS = scores_cal_sorted[k_COPPRS-1, 0]
    elif k_COPPRS == scores_cal.shape[0] + 1:
        threshold_score_COPPRS = float('inf')
    
    in_PI_COPPRS = (y_test <= quantiles_upper_test + threshold_score_COPPRS) & (y_test >= quantiles_lower_test - threshold_score_COPPRS)
    coverage_COPPRS = torch.sum(in_PI_COPPRS) / x_test.shape[0]
    aveLength_COPPRS = torch.mean(quantiles_upper_test - quantiles_lower_test) + 2 * threshold_score_COPPRS

    return coverage_PACOPP, aveLength_PACOPP, coverage_COPPRS, aveLength_COPPRS


