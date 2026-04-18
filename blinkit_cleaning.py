
import pandas as pd
from datetime import datetime

def clean_column_names(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

def handle_missing(df):
    if 'item_weight' in df.columns:
        df['item_weight'] = df['item_weight'].fillna(df['item_weight'].mean())
    return df

def fix_invalid(df):
    if 'item_visibility' in df.columns:
        df['item_visibility'] = df['item_visibility'].replace(0, pd.NA)
        df['item_visibility'] = df['item_visibility'].fillna(df['item_visibility'].mean())
    return df

def standardize_categories(df):
    if 'item_fat_content' in df.columns:
        df['item_fat_content'] = df['item_fat_content'].str.lower().replace({
            'lf': 'low fat',
            'low fat': 'low fat',
            'low_fat': 'low fat',
            'reg': 'regular',
            'regular': 'regular'
        })
    return df

def feature_engineering(df):
    if 'outlet_establishment_year' in df.columns:
        current_year = datetime.now().year
        df['outlet_age'] = current_year - df['outlet_establishment_year']
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def clean_data(df):
    df = clean_column_names(df)
    df = handle_missing(df)
    df = fix_invalid(df)
    df = standardize_categories(df)
    df = feature_engineering(df)
    df = remove_duplicates(df)
    return df
