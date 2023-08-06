import pandas as pd
from supervised.automl import AutoML

df = pd.read_csv('../data_file/fix_holiday_feature_train.csv')

for i in range(1, 11):
    building_df = df[df['building_number'] == 1].copy()
    building_df.drop(['building_number','date_time', 'building_type', 'total_area', 'cooling_area', 'upper_outliers', 'lower_outliers', 'outliers', 'anomaly', 'solar_power_capacity'], axis=1 , inplace=True)
    X = building_df.drop('power_consumption', axis=1)
    y = building_df.power_consumption
    automl = AutoML(mode = 'Optuna',
                    algorithms=["CatBoost", "Xgboost", "LightGBM"],
                    results_path=f"building_train{i}",
                    optuna_time_budget=3600)
    automl.fit(X, y)