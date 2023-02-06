# import unittest
# import torch
#
# from src.trading_environment.environment import Environment
#
#
# data_file = "data/raw//ClosePriceData_2022-10-01_to_2022-08-21.csv"
# portfolio_file = "portfolios//portfolios.json"
#
# class EnvironmentUnitTests(unittest.TestCase):
#     def test_loading_environment(self):
#         self.env = Environment(
#             trading_days=800,
#             token_prices_address=data_file,
#             gas_address=None,
#             gas_limit=1000,
#             portfolio_json=portfolio_file,
#             portfolio_to_use=1,
#             initial_cash=1000,
#             buy_limit=100,
#             sell_limit=100,
#             priority_fee=10,
#             use_change=True,
#             use_covariance=True,
#             reward_metric='sharpe',
#             device=torch.device("cuda:0")
#         )
#         self.assertRaises(, self.env, self.env)
#
# def test_load_prices(n_figures=10):
#     env.preload_prices()
#     return True
#
# def test_dqn_model(n_classes, kernel, inplace, bias):
#     env.preload_prices()
#     model = DQN(n_classes=n_classes, kernel=kernel, inplace=inplace, bias=bias).double()
#
#     y_output = model(torch.from_numpy(env.database_train[0][0]).double())
#     y_target = (y_output > 0.0).double()
#     return True

