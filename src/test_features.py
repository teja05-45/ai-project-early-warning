import pandas as pd
from feature_engineering import engineer_features

df = pd.read_csv("data/raw/project_delivery_risk.csv")
df_fe = engineer_features(df)

print(df_fe.head())
