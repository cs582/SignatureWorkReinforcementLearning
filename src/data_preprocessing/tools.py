def get_dfs(df):
    df.drop(columns=["TUSD", "BUSD"], inplace=True)

    df.sort_index(inplace=True)

    df_change = df/df.iloc[0]-1
    df_daily_roi = df.pct_change().dropna()

    print("df shape:", df.shape)
    print("df_change shape:", df_change.shape)
    print("df_daily_roi:", df_daily_roi.shape)

    return df_change, df_daily_roi