import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from models.evaluating import eval
from logs.logger_file import logger_main, logger_trading_info

import argparse

parser = argparse.ArgumentParser(
    prog='DQN Trainer',
    description='DQN Training algorithm for Portfolio Management',
    epilog='Source code for Carlos Gustavo Salas Flores Signature Work at Duke University & Duke Kunshan University '
           'for the B.S. in Data Science undergrduate degree. '
)

parser.add_argument('-algorithm', type=str, default="Single_DQN", help='Training Algorithm to Use.')
parser.add_argument('-model', type=str, default="CNN", help='Q-approx model to use.')
parser.add_argument('-path', type=str, default=3, help="filepath of the model to test")

parser.add_argument('-portfolio', type=int, default=1, help="Choose portfolio to use")

parser.add_argument('-ic', type=int, default=100000, help="Set initial cash.")
parser.add_argument('-tp', type=int, default=2, help="Priority fee in gwei.")
parser.add_argument('-gl', type=int, default=21000, help="Gas limit in units.")

parser.add_argument('-e', type=float, default=0.01, help="Epsilon.")

parser.add_argument('-d', type=int, default=1000, help="Number of trading days.")
parser.add_argument('-lb', type=int, default=10, help="Lookback window.")

# ViT arguments
parser.add_argument('-dropout', type=float, default=0.2, help="Dropout rate for the model.")
parser.add_argument('-vsize', type=int, default=512, help="Embedding projection size.")
parser.add_argument('-nhead', type=int, default=8, help="Number of heads in Multi-Head Attention.")

parser.add_argument('-reward', type=str, default='roi', help="Reward metric to use in training.")
parser.add_argument('-episodes', type=int, default=500, help="Number of episodes to train.")

parser.add_argument('-use', type=int, default=3, help="2 to use covariance matrix. 3 to use snapshot of lookback days.")

parser.add_argument('-us', type=bool, default=False, help="Load or not a saved model")

args = parser.parse_args()


if __name__ == "__main__":
    data_file = "data/raw//ClosePriceData_2022-10-01_to_2022-08-21.csv"
    portfolios_json = "portfolios//portfolios.json"
    images_saving_path = "data/figures/trading_cycles"

    portfolio = args.portfolio

    # ViT Arguments
    dropout = args.dropout
    vector_size = args.vsize
    nhead = args.nhead

    device = torch.device("cuda:0") if torch.cuda.is_available() else None

    algorithm = args.algorithm
    model_name = args.model
    reward_metric = args.reward
    epsilon = args.e

    use = args.use
    episodes = args.episodes

    n_trading_days = args.d
    lookback = args.lb

    initial_cash = args.ic

    priority_fee = args.tp
    gas_limit = args.gl

    model_path = args.path

    training_info = f"""
    Training {model_name} with a {algorithm} in portfolio {portfolio} with
        data_file = {data_file}
        portfolios_json = {portfolios_json}
        model_path = {model_path}
        
        ViT arguments:
        
        dropout = {dropout}
        vector_size = {vector_size}
        nhead = {nhead}
        
        device = {"CPU" if not torch.cuda.is_available() else torch.cuda.get_device_name(device=device)}
        reward_metric = {reward_metric}
        epsilon = {epsilon}
        
        use = {use}
        
        episodes = {episodes}
        epsilon = {epsilon}
        
        n_trading_days = {n_trading_days}
        lookback = {lookback}
        
        initial_cash = {initial_cash}
        
        priority_fee = {priority_fee}
        gas_limit = {gas_limit}
    """
    print(training_info)

    logger_main.info(training_info)
    logger_trading_info.info(training_info)

    q, history_dqn = eval(
        algorithm=algorithm,
        model_name=model_name,
        images_saving_path=images_saving_path,
        token_prices_address=data_file,
        portfolio_json=portfolios_json,
        portfolio_to_use=portfolio,
        initial_cash=initial_cash,
        n_trading_days=n_trading_days,
        n_tokens=None,
        priority_fee=priority_fee,
        gas_limit=gas_limit,
        epsilon=epsilon,
        episodes=episodes,
        lookback=lookback,
        dropout=dropout,
        vector_size=vector_size,
        nhead=nhead,
        use=use,
        device=device,
        reward_metric=reward_metric,
        model_path=model_path
    )

    logger_main.info("Training Complete!")
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    plt.title("Total reward history")
    plt.plot(history_dqn["metric_history"], color="red")
    plt.savefig(f"data/figures/{model_name}_reward_{current_time}.png")

    plt.title("Total average loss")
    plt.plot(history_dqn["avg_loss"], color="blue")
    plt.savefig(f"data/figures/{model_name}_loss_{current_time}.png")