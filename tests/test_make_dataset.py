import pandas as pd
import numpy as np
from src.data.make_dataset import handle_missing_values, cap_outliers_iqr

def test_handle_missing_values():
    # Create mock dataframe with missing values
    df = pd.DataFrame({
        'high_missing': [1, np.nan, np.nan, np.nan, 5],
        'low_missing_num': [10, 20, np.nan, 40, 50],
        'low_missing_cat': ['A', 'B', 'A', np.nan, 'B']
    })
    
    # 60% missing in 'high_missing' -> should be dropped
    # 20% missing in num -> imputed with median (30)
    # 20% missing in cat -> imputed with mode ('A') [A and B are both 2, so mode picks first alphabetical usually, or we just ensure it fills]
    
    clean_df = handle_missing_values(df, drop_threshold=0.50)
    
    assert 'high_missing' not in clean_df.columns
    assert 'low_missing_num' in clean_df.columns
    assert clean_df['low_missing_num'].isna().sum() == 0
    assert clean_df['low_missing_num'].iloc[2] == 30.0 # Median of 10,20,40,50
    assert clean_df['low_missing_cat'].isna().sum() == 0

def test_cap_outliers_iqr():
    # Create mock dataframe with extreme outliers
    df = pd.DataFrame({
        'normal': [10, 12, 11, 13, 12, 10, 11, 12, 10, 11],
        'with_outlier': [10, 12, 11, 13, 12, 10, 11, 12, 1000, -1000],  # 1000 and -1000 are huge outliers
        'categorical_encoded': [0, 1, 0, 1, 2, 0, 1, 2, 0, 1] # < 10 unique -> shouldn't be capped
    })
    
    clean_df = cap_outliers_iqr(df)
    
    assert clean_df['normal'].max() <= 13
    assert clean_df['with_outlier'].max() < 1000  # Should be clipped
    assert clean_df['with_outlier'].min() > -1000 # Should be clipped
    
    # Categorical encoded shouldn't be touched by clipping
    assert clean_df['categorical_encoded'].max() == 2
