import torch
import torch.nn as nn
import torch.optim as optim
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class GaussianPolicy:
    """
    Defining a Gaussian policy ~ N(mean, std^2)
    Default: ~ N(x/4, std^2) if mean is not specified
    """
    def __init__(self, mean=None, std=1):
        # if std <= 0:
        #     raise ValueError("Standard deviation must be positive")
        self.std = std

        if mean is not None:
            if not isinstance(mean, torch.Tensor):
                try:
                    mean = torch.tensor(mean, dtype=torch.float32)
                except:
                    raise TypeError("Mean must be convertible to a torch.Tensor")
        self.mean = mean
    
    def __call__(self, x):   ## return the mean and std conditional on the given x
        if self.mean is not None:
            return self.mean, self.std
        else:
            return x/4, self.std  # default mean N(x/4, std^2)
    
    def density(self, x, a):
        mean, std = self.__call__(x)
        normal_dist = torch.distributions.Normal(mean, std)
        return torch.exp(normal_dist.log_prob(a)) 
    
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.__call__(x)
        return torch.normal(mean, std)

   
class MlpPolicy(GaussianPolicy):
    """GaussianPolicy using MLP to generate mean and std"""
    def __init__(self, MLP):
        self.MLP = MLP
    
    def __call__(self, x):
        self.MLP.eval()
        with torch.no_grad():
            return self.MLP(x)  # return the mean and std conditional on the given x
    

class Reward:
    """Defining the reward distributions given (s,a)
        if not GMM:  Y ~ N(x+a, std^2)
        else:Y ~ w*N(x+a, (std/4)^2) + (1-w)*N(x+a, std^2)
        where w is the mixture weight
    """
    def __init__(self, std=1, mixture_weight=None):
        if std <= 0:
            raise ValueError("Standard deviation must be positive")
        self.std = std
        self.mixture_weight = mixture_weight

    def __call__(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        mean = x + a 
        if self.mixture_weight is not None:
            normal_small_variance = torch.normal(mean, self.std/4)
            normal_large_variance = torch.normal(mean, self.std)
            mask = torch.rand_like(mean) < self.mixture_weight
            return torch.where(mask, normal_small_variance, normal_large_variance)
        else:
            return torch.normal(mean, self.std)


def generate_data(n, policy, reward_func, x_dim=1, x_std=2):
    x = torch.normal(0, x_std, size=(n, x_dim))  # X ~ N(0, x_std^2=4)
    a = policy.sample(x)
    y = reward_func(x, a)
    return x, a, y


class ConditionalGaussianNet(nn.Module):
    """Net for conditional Gaussian dist., one-hidden layer, 32nodes, ReLU activation function"""
    def __init__(self, input_dim, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_size, 1)     
        self.sigma_head = nn.Linear(hidden_size, 1)  

    def forward(self, x):
        features = self.net(x)
        mu = self.mu_head(features)   
        sigma = torch.exp(self.sigma_head(features)) + 1e-6  
        return mu, sigma


def train_policy_net(model, dataloader, epochs=100, lr=0.001):
    """train the behav policy"""

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, a_batch in dataloader:

            optimizer.zero_grad()
            mu, sigma = model(x_batch)

            loss = -torch.mean(-0.5 * ((a_batch - mu) / sigma) ** 2 - torch.log(sigma))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            logger.debug(
                f"Policy training: Epoch {epoch+1}/{epochs}, Loss {epoch_loss/len(dataloader):.4f}"
            )

    logger.info("Policy training finished!")


def train_cond_reward(model, dataloader, epochs=100, lr=0.001):
    """train the behav policy"""

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, a_batch, y_batch in dataloader:

            optimizer.zero_grad()
            inputs = torch.cat([x_batch, a_batch], dim=1)  # shape:(batch_size, d_x + d_a) 
            mu, sigma = model(inputs)
            #  NLL loss
            loss = -torch.mean(-0.5 * ((y_batch - mu) / sigma) ** 2 - torch.log(sigma))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            logger.debug(
                f"Reward training: Epoch {epoch+1}/{epochs}, Loss {epoch_loss/len(dataloader):.4f}"
            )

    logger.info("Reward training finished!")


class QuantileNet(nn.Module):
    def __init__(self, input_dim, hidden_size=32): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)


class PinballLoss():
    def __init__(self, quantile=0.10, reduction='mean'):
        self.quantile = quantile
        assert 0 < self.quantile
        assert self.quantile < 1
        self.reduction = reduction
    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


def train_quantile_net(model, dataloader, quantile, epochs=100, lr=0.001):
    """Quantile training"""


    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
    criterion = PinballLoss(quantile=quantile) 

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            logger.debug(
                f"Quantile training: Epoch {epoch+1}/{epochs}, Loss {epoch_loss/len(dataloader):.4f}"
            )

    logger.info("Quantile training finished!")



