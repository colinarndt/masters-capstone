import re
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

def extract_score(text):
    """
    Extract score from model response
    """
    if pd.isna(text):  # Handle NaN/None values
        return None
    
    pattern = r'(?i)#?\s*score\s*[:=]\s*(-?\d*\.?\d+)?'
    
    # Search for the pattern in the text
    match = re.search(pattern, str(text))
    
    # If found and there's a number group, return it as float, otherwise return None
    return float(match.group(1)) if match and match.group(1) is not None else None

def extract_explanation(text):
    """
    Extract explanation from model response
    """
    if pd.isna(text):  # Handle NaN/None values
        return None
    
    # Pattern to match 'REASON:' followed by any text
    pattern = r'(?i)#?\s*reason\s*[:=]\s*(.+)'
    
    # Search for the pattern in the text
    match = re.search(pattern, str(text))
    
    # If found, return the description text, otherwise return None
    return match.group(1) if match and match.group(1) is not None else None

def extract_keyword(text):
    """
    Extract the first word if it's YES, NO, or UNKNOWN.
    """
    # Strip leading/trailing whitespace and split by whitespace
    words = text.strip().split()
    
    # Check if there's at least one word and if it matches our criteria
    if words and words[0].upper() in ["YES", "NO", "UNKNOWN"]:
        return words[0]
    else:
        return ""

def extract_remaining_text(text):
    """
    Get the remaining text after removing the first word if it's YES, NO, or UNKNOWN.
    """
    # Strip leading/trailing whitespace
    cleaned_text = text.strip()
    words = cleaned_text.split()
    
    # Check if there's at least one word and if it matches our criteria
    if words and words[0].upper() in ["YES", "NO", "UNKNOWN"]:
        # Get the remaining text by removing the first word
        first_word = words[0]
        remaining = cleaned_text[len(first_word):].strip()
        return remaining
    else:
        return cleaned_text

def process_headline(row, query_function, **kwargs):
    """
    Process headline with given query function
    """
    result = query_function(row['title'], row['company_name'], **kwargs)
    
    print(row['title'])
    print()
    print(result)
    
    return result.replace('\n', '')

def process_financial_data(df):
    """
    Process financial data to calculate trade scores and returns
    """
    data = df.copy()
    
    initial_rows = len(data)
    print(f"Input data contains {initial_rows} headlines")
    
    # Drop rows with no trade prediction
    data = data.dropna(subset=['trade_score'])
    dropped_rows = initial_rows - len(data)
    print(f"Dropping {dropped_rows} rows not containing a trade score")
    
    # Group stocks in same headline period and calculate mean trade_score
    grouped = data.groupby(
        ['stock', 'entry_price', 'exit_price', 'return_pct']
    ).agg({
        'date': 'first',
        'trade_score': 'mean',
        'title': 'count'
    }).reset_index()
    
    # Calculate long-only strategy return
    grouped['long_return_pct'] = grouped.apply(
        lambda row: row['return_pct'] if row['trade_score'] > 0
        else 0,
        axis=1
    )
    
    # Calculate short-only strategy return
    grouped['short_return_pct'] = grouped.apply(
        lambda row: -row['return_pct'] if row['trade_score'] < 0
        else 0,
        axis=1
    )
    
    # Calculate long-short strategy return
    grouped['long_short_return_pct'] = grouped.apply(
        lambda row: 0 if row['trade_score'] == 0
        else abs(row['return_pct']) if np.sign(row['trade_score']) == np.sign(row['return_pct'])
        else -abs(row['return_pct']),
        axis=1
    )
    
    return grouped

def calculate_returns(df):
    """
    Calculate daily weighted returns across different strategies.
    Returns are weighted equally based on number of stocks traded each day.
    
    Finally calculate cumulative returns for different strategies across the trading period.
    """
    
    #Group by date and count number of stocks per day
    daily_counts = df.groupby('date').agg({
        'stock': 'count',
        'trade_score': 'mean'
    }).reset_index()
    daily_counts.columns = ['date', 'num_stocks', 'avg_trade_score']
    
    # Calculate weights for each stock (1/n where n is number of stocks that day)
    df = df.merge(daily_counts, on='date')
    df['weight'] = 1 / df['num_stocks']
    
    # Calculate weighted returns for each strategy
    weighted_returns = df.groupby('date').agg({
        'long_return_pct': lambda x: (x * df.loc[x.index, 'weight']).sum(),
        'short_return_pct': lambda x: (x * df.loc[x.index, 'weight']).sum(),
        'long_short_return_pct': lambda x: (x * df.loc[x.index, 'weight']).sum(),
        'num_stocks': 'first',
        'avg_trade_score': 'mean'
    }).reset_index()
    
    # Round returns to 2 decimal places
    return_columns = ['long_return_pct', 'short_return_pct', 'long_short_return_pct']
    weighted_returns[return_columns] = weighted_returns[return_columns].round(2)
    
    # Calculate cumulative returns
    weighted_returns['long_total_return'] = (
        (1 + weighted_returns['long_return_pct'] / 100).cumprod() - 1
    ) * 100
    weighted_returns['short_total_return'] = (
        (1 + weighted_returns['short_return_pct'] / 100).cumprod() - 1
    ) * 100
    weighted_returns['long_short_total_return'] = \
        np.maximum(
            weighted_returns['long_total_return'] + weighted_returns['short_total_return'],
            -100)

    return weighted_returns

def analyze_trading_strategies(df):
    """
    Analyze trading strategy data with comprehensive statistics.
    """
    # Convert date column to datetime if it's not already
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])
    
    # Make a copy of the data
    df_copy = df.copy()
    df_copy.set_index('date', inplace=True)
    
    # Get strategy columns
    strategy_cols = ['long_return_pct', 'short_return_pct', 'long_short_return_pct']
    
    # Basic summary statistics
    summary_stats = df_copy[strategy_cols].describe()
    
    # Calculate additional statistics
    results = {
        'summary': summary_stats,
        'strategies': {}
    }
    
    for col in strategy_cols:
        strategy_data = df_copy[col]
        
        # Calculate performance metrics
        total_return = ((1 + strategy_data/100).prod() - 1) * 100
        annualized_return = ((1 + total_return/100)**(252/len(strategy_data)) - 1) * 100
        daily_vol = strategy_data.std()
        annualized_vol = daily_vol * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
        
        
        # Store results
        results['strategies'][col] = {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'daily_volatility_pct': daily_vol,
            'annualized_volatility_pct': annualized_vol,
            'sharpe_ratio': sharpe_ratio
        }
    
    # Calculate correlations
    results['correlation_matrix'] = df_copy[strategy_cols].corr()
    
    # Monthly performance
    df_copy['month'] = df_copy.index.to_period('M')
    monthly_returns = df_copy.groupby('month')[strategy_cols].apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    results['monthly_returns'] = monthly_returns
    
    # Quarterly performance
    df_copy['quarter'] = df_copy.index.to_period('Q')
    quarterly_returns = df_copy.groupby('quarter')[strategy_cols].apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    results['quarterly_returns'] = quarterly_returns
    
    # Calculate cumulative returns for plotting
    for col in strategy_cols:
        df_copy[f'{col}_cumulative'] = (1 + df_copy[col]/100).cumprod()
    
    results['data'] = df_copy
    
    return results
