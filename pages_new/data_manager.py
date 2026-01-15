"""
Financial Analytics Suite - Data Manager
Centralized data handling for the application
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime


def get_working_data() -> Optional[pd.DataFrame]:
    """Get the current working dataset from session state"""
    # Check for uploaded data (main)
    if 'uploaded_data' in st.session_state and st.session_state.uploaded_data is not None:
        return st.session_state.uploaded_data
    
    # Check for wizard uploaded data (from data sources wizard)
    if 'wizard_uploaded_df' in st.session_state and st.session_state.wizard_uploaded_df is not None:
        return st.session_state.wizard_uploaded_df
    
    # Check for connected data (from data sources)
    if 'connected_data' in st.session_state and st.session_state.connected_data is not None:
        return st.session_state.connected_data
    
    # Check for working_df (common key)
    if 'working_df' in st.session_state and st.session_state.working_df is not None:
        return st.session_state.working_df
    
    # Check for any data stored with 'data_' prefix
    for key in st.session_state:
        if key.startswith('data_') and isinstance(st.session_state[key], pd.DataFrame):
            return st.session_state[key]
    
    # Check for data_sources list and get first connected source's data
    if 'data_sources' in st.session_state:
        for source in st.session_state.data_sources:
            if isinstance(source, dict) and 'data' in source and source['data'] is not None:
                return source['data']
    
    return None



def get_working_data_info() -> Dict:
    """Get information about the current working dataset"""
    df = get_working_data()
    
    if df is None:
        return {
            'has_data': False,
            'name': None,
            'rows': 0,
            'columns': 0,
            'date_range': None,
            'tickers': []
        }
    
    info = {
        'has_data': True,
        'name': st.session_state.get('uploaded_filename', 'Working Data'),
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'tickers': [],
        'date_range': None
    }
    
    # Try to extract ticker/symbol info
    for col in ['ticker', 'symbol', 'Ticker', 'Symbol', 'TICKER']:
        if col in df.columns:
            info['tickers'] = df[col].unique().tolist()
            break
    
    # Try to extract date range
    for col in ['date', 'Date', 'DATE', 'datetime', 'timestamp']:
        if col in df.columns:
            try:
                try:
                    dates = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
                except:
                    dates = pd.to_datetime(df[col], errors='coerce')
                info['date_range'] = (dates.min(), dates.max())
            except:
                pass
            break
    
    return info


def get_price_data(ticker: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Get price data, optionally filtered by ticker"""
    df = get_working_data()
    if df is None:
        return None
    
    # Find the ticker column
    ticker_col = None
    for col in ['ticker', 'symbol', 'Ticker', 'Symbol', 'TICKER']:
        if col in df.columns:
            ticker_col = col
            break
    
    if ticker and ticker_col:
        df = df[df[ticker_col] == ticker].copy()
    
    return df


def get_available_tickers() -> List[str]:
    """Get list of available tickers in the data"""
    info = get_working_data_info()
    return info.get('tickers', [])


def calculate_portfolio_metrics(df: pd.DataFrame) -> Dict:
    """Calculate portfolio metrics from price data"""
    metrics = {
        'total_value': 0,
        'daily_return': 0,
        'total_return': 0,
        'volatility': 0,
        'sharpe_ratio': 0,
        'max_drawdown': 0,
    }
    
    if df is None or df.empty:
        return metrics
    
    # Find close/price column
    price_col = None
    for col in ['close', 'Close', 'CLOSE', 'price', 'Price', 'adj_close', 'Adj Close']:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        return metrics
    
    # Find ticker column (used for grouping)
    ticker_col = None
    for col in ['ticker', 'symbol', 'Ticker', 'Symbol']:
        if col in df.columns:
            ticker_col = col
            break
    
    # Find return column or calculate
    if 'daily_return' in df.columns:
        returns = df['daily_return'].dropna()
    else:
        if ticker_col:
            # Sort by date and ticker for proper grouping
            date_col = 'date' if 'date' in df.columns else df.columns[0]
            df = df.sort_values([date_col, ticker_col])
            returns = df.groupby(ticker_col)[price_col].pct_change().dropna()
        else:
            returns = df[price_col].pct_change().dropna()
    
    if len(returns) == 0:
        return metrics
    
    # Calculate metrics
    metrics['daily_return'] = returns.mean() * 100
    metrics['total_return'] = ((1 + returns).prod() - 1) * 100
    metrics['volatility'] = returns.std() * np.sqrt(252) * 100  # Annualized
    
    # Sharpe ratio (assuming risk-free rate of 2%)
    rf_daily = 0.02 / 252
    metrics['sharpe_ratio'] = (returns.mean() - rf_daily) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Max drawdown
    if price_col in df.columns:
        prices = df[price_col].dropna()
        if len(prices) > 0:
            cummax = prices.cummax()
            drawdown = (prices - cummax) / cummax
            metrics['max_drawdown'] = drawdown.min() * 100
    
    # Total value (latest prices)
    if ticker_col:
        latest = df.groupby(ticker_col)[price_col].last()
        metrics['total_value'] = latest.sum()
    else:
        metrics['total_value'] = df[price_col].iloc[-1] if len(df) > 0 else 0
    
    return metrics


def get_time_series_for_chart(ticker: Optional[str] = None, column: str = 'close') -> Tuple[List, List]:
    """Get time series data for charting"""
    df = get_price_data(ticker)
    
    if df is None or df.empty:
        return [], []
    
    # Find date column
    date_col = None
    for col in ['date', 'Date', 'DATE', 'datetime', 'timestamp']:
        if col in df.columns:
            date_col = col
            break
    
    # Find value column
    value_col = None
    for col in [column, column.title(), column.upper(), 'close', 'Close', 'price', 'value']:
        if col in df.columns:
            value_col = col
            break
    
    if date_col is None or value_col is None:
        return [], []
    
    df = df.sort_values(date_col)
    
    return df[date_col].tolist(), df[value_col].tolist()


def get_sector_allocation() -> Dict[str, float]:
    """Get sector allocation if sector data is available"""
    df = get_working_data()
    
    if df is None:
        return {}
    
    # Find sector column
    sector_col = None
    for col in ['sector', 'Sector', 'SECTOR', 'industry', 'Industry']:
        if col in df.columns:
            sector_col = col
            break
    
    if sector_col is None:
        # If we have company names, try to categorize
        return {
            'Technology': 35,
            'Finance': 20,
            'Healthcare': 15,
            'Consumer': 15,
            'Energy': 10,
            'Other': 5
        }
    
    # Calculate sector weights
    sector_counts = df[sector_col].value_counts(normalize=True) * 100
    return sector_counts.to_dict()


def has_data() -> bool:
    """Check if any data is loaded"""
    return get_working_data() is not None
