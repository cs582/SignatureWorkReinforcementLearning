import matplotlib.pyplot as plt

from src.environment.trading_environment.environment import Environment

data_file = "data/raw//ClosePriceData_2022-10-01_to_2022-08-21.csv"
portfolio_file = "portfolios//portfolios.json"


def covariance_matrices(n_figures=10, save_dir="data/figures/covariance_matrices"):
    env = Environment(
        trading_days=800,
        token_prices_address=data_file,
        gas_address=None,
        gas_limit=1000,
        portfolio_json=portfolio_file,
        portfolio_to_use=1,
        initial_cash=1000,
        priority_fee=10,
        use=2,
        reward_metric='sharpe',
        device='cuda'
    )

    env.preload_prices()
    for i in range(0, n_figures):
        plt.imshow(env.database_train[i][0])
        plt.savefig(f"{save_dir}//covariance_at_timestep_{i+1}.png")


def shapshot_figures(n_figures=10, save_dir="data/figures/history_snapshot"):
    env = Environment(
        trading_days=800,
        token_prices_address=data_file,
        gas_address=None,
        gas_limit=1000,
        portfolio_json=portfolio_file,
        portfolio_to_use=1,
        initial_cash=1000,
        priority_fee=10,
        lookback=20,
        use=3,
        reward_metric='sharpe',
        device='cuda'
    )

    env.preload_prices()
    for i in range(0, n_figures):
        plt.imshow(env.database_train[i][0])
        plt.savefig(f"{save_dir}//snapshot_at_timestep_{i+1}.png")