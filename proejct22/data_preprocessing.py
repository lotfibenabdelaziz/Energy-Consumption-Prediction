# data_preprocessing.py

import pandas as pd
import numpy as np
import os

# Function: Convert timestamp in milliseconds to datetime

def parse_timestamp(timestamp):
    timestamp = pd.to_datetime(timestamp, unit='ms', utc=False)
    return timestamp + pd.Timedelta(hours=1)

# Function: Load CSV with standard formatting
def load_csv(filepath, col_names):
    return pd.read_csv(filepath, index_col=0, parse_dates=False, skiprows=1, names=col_names)

# Function: Interpolate hourly data
def interpolate_hourly(df):
    df = df.asfreq('1h')
    return df.interpolate(method='time')

# Function: Preprocess energy data
def preprocess_energy(energy_df1, energy_df2):
    energy_df1 = interpolate_hourly(energy_df1)
    energy_df2 = interpolate_hourly(energy_df2)
    
    energy_df1 = energy_df1.diff().dropna()
    energy_df2 = energy_df2.diff().dropna()

    energy_df2 = energy_df2.loc[energy_df2.index >= '2022-03-26 14:00:00']
    return energy_df1.add(energy_df2)

# Function: Preprocess lab loads
def preprocess_lab_data(lab_name, lab_df, threshold=180):
    power_col = f"{lab_name}_power"
    binary_col = f"{lab_name}_binary"
    lab_df = lab_df.rename(columns={"power": power_col})
    lab_df[binary_col] = (lab_df[power_col] >= threshold).astype(int)
    return lab_df[[power_col, binary_col]]

# Function: Create cyclical features
def cyclical_encoding(X, variable, max_value):
    X[f"{variable}_sin"] = np.sin(X[variable] * 2.0 * np.pi / max_value)
    X[f"{variable}_cos"] = np.cos(X[variable] * 2.0 * np.pi / max_value)
    return X

# Function: Check school days in TU Graz academic calendar
def get_is_schoolday(date_arg):
    date_arg = date_arg.date()
    school_periods = [
        ('2022-03-26', '2022-04-08'),
        ('2022-04-25', '2022-05-26'),
        ('2022-05-30', '2022-06-03'),
        ('2022-06-08', '2022-06-30'),
        ('2022-10-03', '2022-11-01'),
        ('2022-11-03', '2022-12-20'),
        ('2023-01-09', '2023-01-31'),
        ('2023-03-01', '2023-03-31'),
    ]
    class_periods = [(pd.Timestamp(start).date(), pd.Timestamp(end).date()) for start, end in school_periods]
    if date_arg.weekday() >= 5:
        return 0
    for start, end in class_periods:
        if start <= date_arg <= end:
            return 1
    return 0

# Main loader

def load_and_preprocess_data():
    # Load all data
    energy_data_1 = load_csv('../data/raw/energy/20_000100-*.csv', ['time', 'energy'])
    energy_data_2 = load_csv('../data/raw/energy/20_999100-*.csv', ['time', 'energy'])

    labs = {
        'lab_P051': load_csv('../data/raw/special_loads/P051.csv', ['time', 'power']),
        'lab_P015': load_csv('../data/raw/special_loads/P015.csv', ['time', 'power']),
        'lab_P011': load_csv('../data/raw/special_loads/P011.csv', ['time', 'power']),
        'lab_251102': load_csv('../data/raw/special_loads/20_251102.csv', ['time', 'power']),
    }

    weather_files = {
        'glob_irrad_in_diffuse': '../data/raw/weather/*GlobIrradInDiffuse*.csv',
        'glob_irrad_total_disc': '../data/raw/weather/*GlobIrradTotal*.csv',
        'rel_hum': '../data/raw/weather/*RelHum*.csv',
        'dew_point': '../data/raw/weather/*DewPoint*.csv',
        'enth': '../data/raw/weather/*Enth*.csv',
        'temp': '../data/raw/weather/*Temp*.csv',
    }

    weather_dfs = {k: load_csv(v, ['time', k]) for k, v in weather_files.items()}

    # Parse timestamps
    for df in [energy_data_1, energy_data_2, *labs.values(), *weather_dfs.values()]:
        df.index = parse_timestamp(df.index)

    # Preprocess
    energy_data = preprocess_energy(energy_data_1, energy_data_2)

    combined = energy_data.rename(columns={"energy": "total_energy_consumption"})

    # Merge labs
    for name, df in labs.items():
        lab_clean = preprocess_lab_data(name, interpolate_hourly(df))
        combined = combined.join(lab_clean, how='left')

    # Join weather
    for name, df in weather_dfs.items():
        df = df.resample('h').mean().loc['2022-03-26 13:00:00':'2023-04-29 09:00:00']
        combined = combined.join(df, how='left')

    # Filter rel_hum outliers
    q1 = combined['rel_hum'].quantile(0.25)
    q3 = combined['rel_hum'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    combined = combined[(combined['rel_hum'] >= lower) & (combined['rel_hum'] <= upper)]

    # Add time-based features
    combined['WorkDay'] = combined.index.map(get_is_schoolday)
    combined['day_of_week'] = combined.index.day_of_week
    combined['month'] = combined.index.month
    combined['year'] = combined.index.year
    combined['hour'] = combined.index.hour
    combined['day'] = combined.index.day
    combined['day_of_year'] = combined.index.day_of_year

    for cyclical_col, max_val in [('day_of_week', 7), ('hour', 24), ('month', 12)]:
        combined = cyclical_encoding(combined, cyclical_col, max_val)

    return combined
