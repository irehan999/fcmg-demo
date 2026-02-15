"""
Preprocessing functions from Kaggle notebook
Replicates exact feature engineering pipeline
"""
import pandas as pd
import numpy as np
from datetime import datetime

def string_processing(data, columns):
    """Strip and uppercase string columns"""
    for i in columns:
        data[i] = data[i].str.strip().str.upper()
    return data

def Holiday_func(tbl):
    """Process holiday events table"""
    Local = tbl[tbl["locale"] == 'Local']
    Regional = tbl[tbl["locale"] == 'Regional']
    National = tbl[tbl["locale_name"].str.contains('Ecuador')]
    prefix = ["Local_","Regional_","National_"]
    diff_type = [Local, Regional, National]
    
    for i in range(3):
        diff_type[i] = diff_type[i][["date", "locale_name","type"]]
        diff_type[i]["values"] = 1
        diff_type[i]["type"] = prefix[i] + diff_type[i]["type"]
        diff_type[i].drop_duplicates(inplace=True)
        diff_type[i] = diff_type[i].pivot_table(
            index=["date", 'locale_name'], 
            columns=["type"], 
            values="values",
            aggfunc='mean', 
            fill_value=0
        ).reset_index()
        diff_type[i] = string_processing(diff_type[i], ['locale_name'])
    
    return diff_type[0], diff_type[1], diff_type[2]

def merge_tables(data, transactions, Oil, stores, National, Local, Regional):
    """Merge all auxiliary tables"""
    input1 = pd.merge(data, transactions, left_on=["date", "store_nbr"], right_on=["date","store_nbr"], how="left")
    input2 = pd.merge(input1, Oil, on=["date"], how="left")
    input3 = pd.merge(input2, stores, on=["store_nbr"], how="left")
    input4 = pd.merge(input3, National, left_on='date', right_on='date', how='left')
    input4 = pd.merge(input4, Local, left_on=["date", "city"], right_on=["date","locale_name"], how="left")
    input4 = pd.merge(input4, Regional, left_on=["date", "state"], right_on=["date","locale_name"], how="left")
    input4["dcoilwtico"] = input4["dcoilwtico"].ffill().bfill()
    input4.drop(columns=["transactions"], inplace=True, errors='ignore')
    input4.fillna(0, inplace=True)
    
    return input4

def date_treatment(data):
    """Extract date features"""
    data["week_num"] = data["date"].dt.strftime("%V")
    data["Day_Name"] = data["date"].dt.day_name()
    data["Month"] = data["date"].dt.month
    data["Year"] = data["date"].dt.year
    data["Month_End"] = data["date"] + pd.offsets.MonthEnd(0)
    data["Month_Begin"] = data["date"] - pd.offsets.MonthBegin(0)
    data.loc[data["date"] != data["Month_Begin"], "Month_Begin"] = 0
    data.loc[data["date"] == data["Month_Begin"], "Month_Begin"] = 1
    data.loc[data["date"] != data["Month_End"], "Month_End"] = 0
    data.loc[data["date"] == data["Month_End"], "Month_End"] = 1
    Day_Name_Dummies = pd.get_dummies(data["Day_Name"], dtype='int')
    data["week_num"] = data["week_num"].astype('int')
    data["Month_End"] = data["Month_End"].astype('int')
    data["Month_Begin"] = data["Month_Begin"].astype('int')
    data = pd.concat([data, Day_Name_Dummies], axis=1)
    
    return data

def fourier_features(index, freq, order):
    """Calculate Fourier seasonality features"""
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1/freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i*k),
            f"cos_{freq}_{i}": np.cos(i*k)
        })
    return pd.DataFrame(features, index=index)

def add_seasonality(period, freq, order):
    """Add seasonality features"""
    seasonality = pd.DataFrame()
    
    for i in freq:
        freq_seasonality = fourier_features(period, freq=i, order=order)
        seasonality = pd.concat([seasonality, freq_seasonality], axis=1)
    
    seasonality.reset_index(inplace=True)
    seasonality.rename(columns={"index":"date"}, inplace=True)
    seasonality["date"] = seasonality["date"].dt.to_timestamp()
    
    return seasonality

def add_trend(tbl):
    """Add trend line"""
    num_list = np.array([i+1 for i in range(len(tbl))])
    sqr_list = np.square(num_list)
    
    trend_line = pd.DataFrame({"Trend":num_list, "Trend_Square":sqr_list})
    trend_line["date"] = tbl["date"]
    
    return trend_line

def add_lag_features(df):
    """Add lag features"""
    for lag in range(1, 8):
        df[f'lag_{lag}'] = df['sales'].shift(lag*1782)
    return df.dropna()

def make_lags(data, clm, lags, idx):
    """Make lag features for prediction"""
    lag_data = pd.DataFrame()
    for i in range(1, lags+1):
        lag_1 = data[[clm]].shift(i)
        lag_1.rename(columns={clm: f"lag_{i}"}, inplace=True)
        lag_data = pd.concat([lag_data, lag_1], axis=1)
    lag_data.index = idx
    lag_data.dropna(inplace=True)
    return lag_data

def make_multistep_target(ts, steps):
    """Make multi-step targets"""
    return pd.concat(
        {f'y_step_{i + 1}': ts.shift(-i) for i in range(steps)},
        axis=1
    )

def preprocess_for_prediction(test, train, transactions, Oil, stores, Holiday_events):
    """
    Full preprocessing pipeline for test data
    Replicates notebook's data_preprocessing_2 function
    """
    # Process stores
    stores = string_processing(stores, ['city', 'state'])
    
    # Process holidays
    Local, Regional, National = Holiday_func(Holiday_events)
    
    # Merge test data
    test_data = merge_tables(test, transactions, Oil, stores, National, Local, Regional)
    
    # Calculate rankings from training data
    family_rank = train.groupby(by=["family"], as_index=False)["sales"].mean()\
        .sort_values(by=["sales"], ascending=False)\
        .reset_index(drop=True).reset_index()\
        .drop(columns=["sales"])\
        .rename(columns={"index":"family_rank"})
    
    store_rank = train.groupby(by=["store_nbr"], as_index=False)["sales"].mean()\
        .sort_values(by=["sales"], ascending=False)\
        .reset_index(drop=True).reset_index()\
        .drop(columns=["sales"])\
        .rename(columns={"index":"store_rank"})
    
    # Merge rankings
    test_data = pd.merge(test_data, family_rank, left_on=["family"], right_on=["family"])
    test_data = pd.merge(test_data, store_rank, left_on=["store_nbr"], right_on=["store_nbr"])
    
    # Date features
    test_data = date_treatment(test_data)
    
    # Calculate lags from last 7 days of training data
    train_lags = train[train['date'] >= '2017-08-08'].copy()
    train_lags = train_lags.sort_values(['store_nbr', 'family', 'date'])
    
    # Get last 7 values for each store-family combination
    lag_dict = {}
    for (store, fam), group in train_lags.groupby(['store_nbr', 'family']):
        last_7 = group['sales'].tail(7).values
        if len(last_7) == 7:
            for i in range(7):
                lag_dict[(store, fam, i+1)] = last_7[-(i+1)]
    
    # Add lag features to test data
    for i in range(1, 8):
        test_data[f'lag_{i}'] = test_data.apply(
            lambda row: lag_dict.get((row['store_nbr'], row['family'], i), 0), 
            axis=1
        )
    
    # Seasonality from start to end
    seasonality = add_seasonality(
        pd.period_range('2013-01-01', '2017-08-31'), 
        freq=[3.5, 7, 30, 365], 
        order=1
    )
    trend_line = add_trend(seasonality[["date"]])
    
    # Merge seasonality and trend
    test_data_seas = pd.merge(test_data, seasonality, left_on="date", right_on="date", how="left")
    test_data_seas = pd.merge(test_data_seas, trend_line, left_on="date", right_on="date", how="left")
    
    # Drop unnecessary columns
    drop_cols = ['state', 'family', 'store_nbr', 'city', 'Day_Name', 'type', 'cluster']
    drop_cols += [c for c in test_data_seas.columns if 'locale_name' in c]
    test_data_seas.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    return test_data_seas
