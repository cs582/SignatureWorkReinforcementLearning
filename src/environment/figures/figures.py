import matplotlib.pyplot as plt

from src.trading_environment.environment import Environment

data_file = "data/raw//ClosePriceData_2022-10-01_to_2022-08-21.csv"
portfolio_file = "portfolios//portfolios.json"


env = Environment(
    trading_days=800,
    token_prices_address=data_file,
    gas_address=None,
    gas_limit=1000,
    portfolio_json=portfolio_file,
    portfolio_to_use=1,
    initial_cash=1000,
    buy_limit=100,
    sell_limit=100,
    priority_fee=10,
    use_change=True,
    use_covariance=True,
    reward_metric='sharpe',
    device='cuda'
)

env.preload_prices()


def covariance_matrices(n_figures=10, save_dir="data/figures"):
    for i in range(0, n_figures):
        plt.imshow(env.database_train[i][0])
        plt.savefig(f"{save_dir}//figure_{i}.png")