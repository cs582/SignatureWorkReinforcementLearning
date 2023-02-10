import json
import pandas as pd

from src.preprocessing.data_preprocessing.action_space_viz import visualize_action_space


def create(portfolio, portfolios_json, data_file):
    df = pd.read_csv(data_file, index_col=0).sort_index(inplace=True)

    with open(portfolios_json, "r") as file:
        columns = json.load(file)

    df = df[columns]


