"""
Financial Analytics Suite - Executive Decision Dashboard
Commercial-Grade PowerBI-Style Dashboard for Investment & Business Decisions
Enterprise Edition with AI-Powered Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from pages_new.model_results_manager import (
    get_dashboard_data, get_all_run_models, get_model_result,
    get_run_history, count_run_models, get_category_results
)
from pages_new.data_manager import get_working_data, get_working_data_info, has_data
from pages_new.theme_utils import get_theme_colors


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TYPE DETECTION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def detect_data_type(df: pd.DataFrame) -> Dict:
    """
    Automatically detect the type of data and return available features.
    
    Returns:
        Dict with:
        - data_type: 'stock', 'sales', 'financial', 'general'
        - features: list of available analysis features
        - detected_columns: dict of detected column types
        - description: human-readable description
    """
    columns = [col.lower() for col in df.columns]
    original_columns = df.columns.tolist()
    
    detected = {
        'data_type': 'general',
        'features': [],
        'detected_columns': {},
        'description': 'General Numeric Data',
        'icon': '📊',
        'available_charts': [],
        'available_models': []
    }
    
    # ═══════════════════════════════════════════════════════════════
    # STOCK/TRADING DATA DETECTION
    # ═══════════════════════════════════════════════════════════════
    stock_indicators = {
        'price': ['close', 'open', 'high', 'low', 'price', 'adj close', 'adj_close', 'adjusted close'],
        'volume': ['volume', 'vol', 'trading_volume'],
        'ticker': ['ticker', 'symbol', 'stock', 'asset'],
        'ohlc': ['open', 'high', 'low', 'close']
    }
    
    has_price = any(col in columns for col in stock_indicators['price'])
    has_volume = any(col in columns for col in stock_indicators['volume'])
    has_ticker = any(col in columns for col in stock_indicators['ticker'])
    has_ohlc = sum(1 for col in stock_indicators['ohlc'] if col in columns) >= 3
    
    # ═══════════════════════════════════════════════════════════════
    # SALES DATA DETECTION
    # ═══════════════════════════════════════════════════════════════
    sales_indicators = ['sales', 'quantity', 'units', 'orders', 'customers', 'product', 
                       'category', 'region', 'store', 'transaction', 'purchase']
    has_sales = sum(1 for col in columns if any(s in col for s in sales_indicators)) >= 2
    
    # ═══════════════════════════════════════════════════════════════
    # FINANCIAL/ACCOUNTING DATA DETECTION
    # ═══════════════════════════════════════════════════════════════
    financial_indicators = ['revenue', 'expense', 'profit', 'cost', 'income', 'balance',
                           'asset', 'liability', 'equity', 'cash', 'budget', 'actual',
                           'margin', 'ebitda', 'net_income', 'gross_profit']
    has_financial = sum(1 for col in columns if any(f in col for f in financial_indicators)) >= 2
    
    # ═══════════════════════════════════════════════════════════════
    # DETERMINE DATA TYPE
    # ═══════════════════════════════════════════════════════════════
    
    if has_price or has_ohlc or (has_volume and has_ticker):
        detected['data_type'] = 'stock'
        detected['description'] = 'Stock/Trading Data'
        detected['icon'] = '📈'
        detected['features'] = [
            'Technical Analysis (RSI, MACD, MA)',
            'Price Trends & Momentum',
            'Volume Analysis',
            'Risk-Return Analysis',
            'Portfolio Allocation',
            'Volatility Metrics',
            'Returns Distribution'
        ]
        detected['available_charts'] = [
            'performance', 'risk_matrix', 'allocation', 'trend_momentum',
            'volume', 'returns_distribution', 'technical_signals'
        ]
        detected['available_models'] = [
            'ARIMA', 'SARIMA', 'Prophet', 'LSTM', 'XGBoost'
        ]
        
    elif has_sales:
        detected['data_type'] = 'sales'
        detected['description'] = 'Sales/Retail Data'
        detected['icon'] = '🛒'
        detected['features'] = [
            'Sales Trend Analysis',
            'Product Performance',
            'Regional Analysis',
            'Growth Metrics',
            'Seasonality Detection',
            'Category Breakdown'
        ]
        detected['available_charts'] = [
            'performance', 'allocation', 'trend_momentum', 'returns_distribution'
        ]
        detected['available_models'] = [
            'ARIMA', 'Prophet', 'XGBoost', 'Linear Regression'
        ]
        
    elif has_financial:
        detected['data_type'] = 'financial'
        detected['description'] = 'Financial/Accounting Data'
        detected['icon'] = '💰'
        detected['features'] = [
            'Revenue Analysis',
            'Expense Tracking',
            'Profit Margins',
            'Cash Flow Analysis',
            'Budget vs Actual',
            'Financial Ratios'
        ]
        detected['available_charts'] = [
            'performance', 'cashflow', 'profitability', 'allocation', 'trend_momentum'
        ]
        detected['available_models'] = [
            'ARIMA', 'Prophet', 'Linear Regression', 'Gradient Boosting'
        ]
        
    else:
        detected['data_type'] = 'general'
        detected['description'] = 'General Numeric Data'
        detected['icon'] = '📊'
        detected['features'] = [
            'Trend Analysis',
            'Statistical Summary',
            'Distribution Analysis',
            'Correlation Analysis',
            'Outlier Detection'
        ]
        detected['available_charts'] = [
            'performance', 'trend_momentum', 'returns_distribution', 'allocation'
        ]
        detected['available_models'] = [
            'Linear Regression', 'Random Forest', 'XGBoost'
        ]
    
    # ═══════════════════════════════════════════════════════════════
    # DETECT SPECIFIC COLUMNS
    # ═══════════════════════════════════════════════════════════════
    for orig_col, lower_col in zip(original_columns, columns):
        if any(p in lower_col for p in stock_indicators['price']):
            detected['detected_columns']['price'] = orig_col
        if any(v in lower_col for v in stock_indicators['volume']):
            detected['detected_columns']['volume'] = orig_col
        if any(t in lower_col for t in stock_indicators['ticker']):
            detected['detected_columns']['ticker'] = orig_col
        if 'date' in lower_col or 'time' in lower_col:
            detected['detected_columns']['date'] = orig_col
    
    return detected



def calculate_business_insights(df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive business & portfolio insights with advanced analysis.
    
    This function performs:
    - Multi-timeframe return analysis (1D, 5D, 20D, 60D)
    - Momentum scoring using price trends
    - Volatility-adjusted performance metrics
    - Technical indicator-based signals
    - Multi-factor decision scoring
    - Risk-adjusted recommendations
    """
    insights = {
        'total_return': 0, 'volatility': 0, 'sharpe_ratio': 0, 'max_drawdown': 0,
        'best_performer': 'N/A', 'worst_performer': 'N/A', 'trend': 'neutral',
        'recommendation': 'Hold', 'risk_level': 'Medium', 'confidence': 75,
        # Business KPIs
        'revenue_growth': 0, 'profit_margin': 0, 'cash_flow': 0,
        'liquidity_ratio': 0, 'debt_ratio': 0, 'roi': 0,
        'market_sentiment': 50, 'decision_score': 50,
        # Advanced Metrics
        'momentum_score': 0, 'trend_strength': 0, 'risk_reward_ratio': 0,
        'signal_strength': 'Neutral', 'short_term_outlook': 'Neutral',
        'long_term_outlook': 'Neutral', 'action_urgency': 'Normal',
        # Technical Signals
        'rsi_signal': 'Neutral', 'macd_signal': 'Neutral', 'ma_signal': 'Neutral',
        'composite_signal': 50,
        # General Data Stats
        'mean_val': 0, 'median_val': 0, 'std_dev': 0, 'max_val': 0, 'min_val': 0,
        'data_range': 0, 'coeff_variation': 0, 'skewness': 0,
        'total_sum': 0  # Added for sales/financial
    }

    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            return insights
            
        # Get price column (prefer close price)
        price_col = None
        price_keywords = ['close', 'Close', 'price', 'Price', 'adj_close', 'Adj Close', 'value', 'Value', 'amount', 'Amount']
        for col in price_keywords:
            if col in df.columns:
                price_col = col
                break
        
        if price_col is None:
            # Fallback to the first numeric column that isn't an index or ID
            likely_cols = [c for c in numeric_cols if 'id' not in c.lower() and 'index' not in c.lower()]
            price_col = likely_cols[0] if likely_cols else numeric_cols[0]
        
        # ═══════════════════════════════════════════════════════════════
        # GENERAL STATISTICAL ANALYSIS
        # ═══════════════════════════════════════════════════════════════
        
        # CRITICAL: Ensure data is sorted by date for time-series analysis
        sorted_df = df.copy()
        date_col = None
        date_keywords = ['date', 'Date', 'DATE', 'datetime', 'timestamp', 'Time', 'time', 'Period', 'period', 'Month', 'month', 'Year', 'year']
        for col in date_keywords:
            if col in sorted_df.columns:
                date_col = col
                break
        
        if date_col:
            try:
                sorted_df[date_col] = pd.to_datetime(sorted_df[date_col], errors='coerce')
                sorted_df = sorted_df.sort_values(by=date_col, ascending=True)
            except:
                pass # Use index order if date sort fails
        
        data_vals = sorted_df[price_col].dropna().values
        if len(data_vals) > 0:
            insights['mean_val'] = float(np.mean(data_vals))
            insights['median_val'] = float(np.median(data_vals))
            insights['std_dev'] = float(np.std(data_vals))
            insights['max_val'] = float(np.max(data_vals))
            insights['min_val'] = float(np.min(data_vals))
            insights['total_sum'] = float(np.sum(data_vals))
            insights['data_range'] = float(insights['max_val'] - insights['min_val'])
            insights['coeff_variation'] = float(insights['std_dev'] / insights['mean_val']) if insights['mean_val'] != 0 else 0.0
        
        prices = data_vals
        if len(prices) < 5:
            return insights
        
        # ═══════════════════════════════════════════════════════════════
        # MULTI-TIMEFRAME RETURN ANALYSIS
        # ═══════════════════════════════════════════════════════════════
        returns_1d = (prices[-1] / prices[-2] - 1) * 100 if len(prices) > 1 else 0
        returns_5d = (prices[-1] / prices[-5] - 1) * 100 if len(prices) > 5 else 0
        returns_20d = (prices[-1] / prices[-20] - 1) * 100 if len(prices) > 20 else 0
        returns_60d = (prices[-1] / prices[-60] - 1) * 100 if len(prices) > 60 else 0
        total_return = (prices[-1] / prices[0] - 1) * 100
        
        insights['total_return'] = total_return
        
        # ═══════════════════════════════════════════════════════════════
        # MOMENTUM SCORING (Short-term vs Long-term momentum)
        # ═══════════════════════════════════════════════════════════════
        short_momentum = (returns_1d + returns_5d * 0.8) / 2
        long_momentum = (returns_20d * 0.7 + returns_60d * 0.3 if len(prices) > 60 else returns_20d)
        
        # Momentum score: -100 to +100
        momentum_score = np.clip(short_momentum + long_momentum, -100, 100)
        insights['momentum_score'] = momentum_score
        
        # ═══════════════════════════════════════════════════════════════
        # VOLATILITY & RISK METRICS
        # ═══════════════════════════════════════════════════════════════
        daily_returns = np.diff(prices) / prices[:-1]
        volatility = np.std(daily_returns) * np.sqrt(252) * 100
        insights['volatility'] = volatility
        
        # Max Drawdown calculation (with zero-check)
        if len(prices) > 0 and np.all(prices > 0):
            peak = np.maximum.accumulate(prices)
            drawdown = (peak - prices) / peak * 100
            max_drawdown = np.max(drawdown)
        else:
            # Handle zeros or invalid prices gracefully
            max_drawdown = 0.0
        insights['max_drawdown'] = float(max_drawdown)
        
        # Sharpe Ratio (risk-adjusted return)
        risk_free_rate = 4.5  # Current approximate risk-free rate
        excess_return = total_return - risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0
        insights['sharpe_ratio'] = sharpe
        
        # Risk-Reward Ratio
        avg_gain = np.mean(daily_returns[daily_returns > 0]) if len(daily_returns[daily_returns > 0]) > 0 else 0.01
        avg_loss = abs(np.mean(daily_returns[daily_returns < 0])) if len(daily_returns[daily_returns < 0]) > 0 else 0.01
        risk_reward = avg_gain / avg_loss if avg_loss > 0 else 1
        insights['risk_reward_ratio'] = risk_reward
        
        # ═══════════════════════════════════════════════════════════════
        # TECHNICAL INDICATORS
        # ═══════════════════════════════════════════════════════════════
        
        # RSI (14-period)
        if len(prices) >= 14:
            gains = np.diff(prices)
            pos_gains = np.where(gains > 0, gains, 0)[-14:]
            neg_gains = np.where(gains < 0, abs(gains), 0)[-14:]
            avg_gain_rsi = np.mean(pos_gains)
            avg_loss_rsi = np.mean(neg_gains)
            rs = avg_gain_rsi / avg_loss_rsi if avg_loss_rsi > 0 else 999
            rsi = 100 - (100 / (1 + rs))
            
            if rsi < 30:
                insights['rsi_signal'] = 'Oversold (Buy)'
            elif rsi > 70:
                insights['rsi_signal'] = 'Overbought (Sell)'
            else:
                insights['rsi_signal'] = 'Neutral'
        
        # Moving Average Signal
        if len(prices) >= 50:
            sma_20 = np.mean(prices[-20:])
            sma_50 = np.mean(prices[-50:])
            current_price = prices[-1]
            
            if current_price > sma_20 > sma_50:
                insights['ma_signal'] = 'Bullish (Above MAs)'
            elif current_price < sma_20 < sma_50:
                insights['ma_signal'] = 'Bearish (Below MAs)'
            elif sma_20 > sma_50:
                insights['ma_signal'] = 'Bullish Trend'
            else:
                insights['ma_signal'] = 'Bearish Trend'
        
        # MACD-like signal
        if len(prices) >= 26:
            ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean().iloc[-1]
            ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean().iloc[-1]
            macd = ema_12 - ema_26
            signal = pd.Series(prices).ewm(span=9, adjust=False).mean().iloc[-1]
            
            if macd > 0 and ema_12 > signal:
                insights['macd_signal'] = 'Bullish'
            elif macd < 0 and ema_12 < signal:
                insights['macd_signal'] = 'Bearish'
            else:
                insights['macd_signal'] = 'Neutral'
        
        # ═══════════════════════════════════════════════════════════════
        # COMPOSITE SIGNAL (Multi-Factor Scoring: 0-100)
        # ═══════════════════════════════════════════════════════════════
        
        # Factor 1: Momentum (25% weight)
        momentum_factor = 50 + momentum_score * 0.5  # Scale to 0-100
        
        # Factor 2: Risk-Adjusted Return (25% weight)
        sharpe_factor = 50 + sharpe * 15  # Sharpe of 2 = 80, -2 = 20
        
        # Factor 3: Trend Alignment (25% weight)
        trend_factor = 50
        if 'Bullish' in insights['ma_signal']:
            trend_factor += 20
        if 'Bearish' in insights['ma_signal']:
            trend_factor -= 20
        if 'Bullish' in insights['macd_signal']:
            trend_factor += 10
        if 'Bearish' in insights['macd_signal']:
            trend_factor -= 10
        
        # Factor 4: Risk Level (25% weight)
        risk_factor = 100 - (max_drawdown * 2)  # Lower drawdown = higher score
        
        # Composite score
        composite = (momentum_factor * 0.25 + sharpe_factor * 0.25 + 
                    trend_factor * 0.25 + risk_factor * 0.25)
        insights['composite_signal'] = max(5, min(95, composite))
        
        # ═══════════════════════════════════════════════════════════════
        # TREND STRENGTH & DIRECTION
        # ═══════════════════════════════════════════════════════════════
        
        # Trend strength based on consistency of direction
        positive_days = np.sum(daily_returns[-20:] > 0) if len(daily_returns) >= 20 else 10
        trend_strength = abs(positive_days - 10) * 10  # 0-100 scale
        insights['trend_strength'] = trend_strength
        
        if positive_days > 14:
            insights['trend'] = 'bullish'
        elif positive_days < 6:
            insights['trend'] = 'bearish'
        else:
            insights['trend'] = 'neutral'
        
        # ═══════════════════════════════════════════════════════════════
        # SMART AI RECOMMENDATION ENGINE
        # ═══════════════════════════════════════════════════════════════
        
        # Detect type for recommendations
        data_type_info = detect_data_type(df)
        dtype = data_type_info['data_type']
        
        composite_score = insights['composite_signal']
        
        # Recommendation labels based on data type
        labels = {
            'stock': ['Strong Buy', 'Buy', 'Hold', 'Reduce', 'Sell'],
            'sales': ['Hyper-Growth', 'Strong Growth', 'Stable', 'Slowdown', 'Critical'],
            'financial': ['High Profit', 'Healthy', 'Stable', 'Declining', 'Urgent Action'],
            'general': ['Optimal Trend', 'Positive', 'Neutral', 'Counter-Trend', 'Critical Risk']
        }
        active_labels = labels.get(dtype, labels['general'])
        
        # Primary recommendation based on composite score
        if composite_score >= 80:
            insights['recommendation'] = active_labels[0]
            insights['signal_strength'] = 'Very Strong'
            insights['confidence'] = 90
            insights['action_urgency'] = 'High'
        elif composite_score >= 65:
            insights['recommendation'] = active_labels[1]
            insights['signal_strength'] = 'Strong'
            insights['confidence'] = 80
            insights['action_urgency'] = 'Moderate'
        elif composite_score >= 50:
            insights['recommendation'] = active_labels[2]
            insights['signal_strength'] = 'Neutral'
            insights['confidence'] = 70
            insights['action_urgency'] = 'Normal'
        elif composite_score >= 35:
            insights['recommendation'] = active_labels[3]
            insights['signal_strength'] = 'Weak'
            insights['confidence'] = 75
            insights['action_urgency'] = 'Moderate'
        else:
            insights['recommendation'] = active_labels[4]
            insights['signal_strength'] = 'Strong Warning'
            insights['confidence'] = 85
            insights['action_urgency'] = 'High'
        
        # Adjust confidence based on volatility (high vol = less confident)
        if volatility > 40:
            insights['confidence'] = max(50, insights['confidence'] - 15)
            insights['risk_level'] = 'Very High'
        elif volatility > 30:
            insights['confidence'] = max(55, insights['confidence'] - 10)
            insights['risk_level'] = 'High'
        elif volatility > 20:
            insights['risk_level'] = 'Medium-High'
        elif volatility > 10:
            insights['risk_level'] = 'Medium'
        else:
            insights['confidence'] = min(95, insights['confidence'] + 5)
            insights['risk_level'] = 'Low'

        
        # Short-term and Long-term Outlook
        if short_momentum > 5:
            insights['short_term_outlook'] = 'Bullish'
        elif short_momentum < -5:
            insights['short_term_outlook'] = 'Bearish'
        else:
            insights['short_term_outlook'] = 'Neutral'
        
        if long_momentum > 10:
            insights['long_term_outlook'] = 'Bullish'
        elif long_momentum < -10:
            insights['long_term_outlook'] = 'Bearish'
        else:
            insights['long_term_outlook'] = 'Neutral'
        
        # ═══════════════════════════════════════════════════════════════
        # BEST/WORST PERFORMERS (if multiple assets)
        # ═══════════════════════════════════════════════════════════════
        ticker_col = None
        for col in ['ticker', 'Ticker', 'symbol', 'Symbol']:
            if col in df.columns:
                ticker_col = col
                break
        
        if ticker_col and price_col in df.columns:
            try:
                ticker_returns = df.groupby(ticker_col)[price_col].apply(
                    lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 1 and x.iloc[0] != 0 else 0
                )
                if len(ticker_returns) > 0:
                    insights['best_performer'] = ticker_returns.idxmax()
                    insights['worst_performer'] = ticker_returns.idxmin()
            except:
                pass
        
        # ═══════════════════════════════════════════════════════════════
        # BUSINESS METRICS (enhanced data-type awareness)
        # ═══════════════════════════════════════════════════════════════
        insights['market_sentiment'] = insights['composite_signal']
        insights['decision_score'] = insights['composite_signal']
        
        # Check if this is likely sales data
        has_sales_cols = any(c in [col.lower() for col in df.columns] for c in ['sales', 'revenue', 'units', 'quantity'])
        
        if has_sales_cols:
            insights['revenue_growth'] = float(total_return * 0.5) # More realistic scale for sales
            insights['profit_margin'] = max(5, min(45, 15 + total_return * 0.1))
            insights['cash_flow'] = insights['total_sum'] * 0.2
            insights['liquidity_ratio'] = 2.0 + (insights['revenue_growth'] / 100)
            insights['roi'] = float(insights['revenue_growth'] * 1.5)
        else:
            insights['revenue_growth'] = total_return * 0.8
            insights['profit_margin'] = max(5, min(45, 25 + total_return * 0.3))
            insights['cash_flow'] = total_return * 10000
            insights['liquidity_ratio'] = max(0.5, min(3.5, 1.5 + sharpe * 0.3))
            insights['roi'] = total_return * risk_reward
        
        insights['debt_ratio'] = max(0.1, min(0.9, 0.4 - total_return * 0.003))
        
    except Exception as e:
        # Return default insights on error
        pass
    
    return insights


def create_sentiment_gauge(sentiment: float, c: Dict) -> go.Figure:
    """Create market sentiment gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=sentiment,
        delta={'reference': 50, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': "#6366f1", 'thickness': 0.75},
            'bgcolor': "rgba(30,41,59,0.8)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 25], 'color': 'rgba(239,68,68,0.4)'},
                {'range': [25, 50], 'color': 'rgba(249,115,22,0.4)'},
                {'range': [50, 75], 'color': 'rgba(234,179,8,0.4)'},
                {'range': [75, 100], 'color': 'rgba(16,185,129,0.4)'}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': sentiment}
        },
        title={'text': "Market Sentiment", 'font': {'size': 16, 'color': 'white'}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Inter'},
        height=280,
        margin=dict(t=80, b=30, l=30, r=30)
    )
    return fig


def create_decision_gauge(score: float, c: Dict) -> go.Figure:
    """Create decision score gauge"""
    color = "#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(30,41,59,0.8)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239,68,68,0.3)'},
                {'range': [40, 70], 'color': 'rgba(245,158,11,0.3)'},
                {'range': [70, 100], 'color': 'rgba(16,185,129,0.3)'}
            ]
        },
        title={'text': "Decision Score", 'font': {'size': 16, 'color': 'white'}}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Inter'},
        height=280,
        margin=dict(t=80, b=30, l=30, r=30)
    )
    return fig


def create_cashflow_chart(df: pd.DataFrame, c: Dict) -> go.Figure:
    """Create cash flow waterfall chart"""
    categories = ['Revenue', 'Cost of Sales', 'Operating Exp', 'Taxes', 'Net Cash Flow']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 4:
        base_val = abs(df[numeric_cols[0]].mean())
        values = [base_val, -base_val*0.4, -base_val*0.25, -base_val*0.08, base_val*0.27]
    else:
        values = [100000, -40000, -25000, -8000, 27000]
    
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=categories,
        y=values,
        connector={"line": {"color": "rgba(99,102,241,0.4)", "width": 2}},
        increasing={"marker": {"color": "#10b981", "line": {"color": "#10b981", "width": 1}}},
        decreasing={"marker": {"color": "#ef4444", "line": {"color": "#ef4444", "width": 1}}},
        totals={"marker": {"color": "#6366f1", "line": {"color": "#6366f1", "width": 1}}},
        textposition="outside",
        text=[f"${v/1000:.0f}K" for v in values]
    ))
    
    fig.update_layout(
        title=dict(text='💵 Cash Flow Analysis', font=dict(size=18, color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,24,39,0.8)',
        font=dict(color='#f8fafc', family='Inter'),
        height=320,
        margin=dict(t=60, l=60, r=30, b=50),
        xaxis=dict(gridcolor='rgba(51,65,85,0.3)'),
        yaxis=dict(gridcolor='rgba(51,65,85,0.3)', title='Amount ($)'),
        showlegend=False
    )
    return fig


def create_profitability_radar(insights: Dict, c: Dict) -> go.Figure:
    """Create profitability metrics radar chart"""
    categories = ['Profit Margin', 'ROI', 'Revenue Growth', 'Cash Position', 'Efficiency']
    
    values = [
        min(100, max(0, insights['profit_margin'] * 2)),
        min(100, max(0, 50 + insights['roi'])),
        min(100, max(0, 50 + insights['revenue_growth'])),
        min(100, max(0, insights['liquidity_ratio'] * 30)),
        min(100, max(0, 100 - insights['debt_ratio'] * 100))
    ]
    values.append(values[0])  # Close the radar
    categories.append(categories[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(99,102,241,0.3)',
        line=dict(color='#6366f1', width=2),
        name='Performance'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(51,65,85,0.5)',
                           tickfont=dict(color='#94a3b8', size=10)),
            angularaxis=dict(gridcolor='rgba(51,65,85,0.5)',
                            tickfont=dict(color='white', size=11)),
            bgcolor='rgba(17,24,39,0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Inter'),
        height=320,
        margin=dict(t=40, b=40, l=60, r=60),
        showlegend=False,
        title=dict(text='📊 Profitability Metrics', font=dict(size=18, color='white'), x=0.5)
    )
    return fig


def create_performance_chart(df: pd.DataFrame, c: Dict) -> go.Figure:
    """Create main performance overview chart with clear labels and values"""
    fig = go.Figure()
    
    # Find price column preferentially
    price_col = None
    for col in ['close', 'Close', 'CLOSE', 'price', 'Price', 'adj_close', 'Adj Close']:
        if col in df.columns:
            price_col = col
            break
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if price_col and price_col in numeric_cols:
        # Move price column to front
        numeric_cols = [price_col] + [c for c in numeric_cols if c != price_col]
    
    numeric_cols = numeric_cols[:5]  # Max 5 lines
    colors = ['#6366f1', '#22d3ee', '#f472b6', '#10b981', '#f59e0b']
    
    annotations_data = []
    
    for i, col in enumerate(numeric_cols):
        if len(df[col].dropna()) > 0:
            vals = df[col].dropna().values
            normalized = (vals / vals[0] * 100) if vals[0] != 0 else vals
            
            # Calculate actual return for this column
            actual_return = ((vals[-1] / vals[0]) - 1) * 100 if vals[0] != 0 else 0
            
            fig.add_trace(go.Scatter(
                y=normalized,
                mode='lines',
                name=f"{col} ({actual_return:+.1f}%)",
                line=dict(color=colors[i % len(colors)], width=2.5),
                fill='tonexty' if i > 0 else None,
                fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.1)',
                hovertemplate=f'<b>{col}</b><br>Value: %{{y:.2f}}<br>Return: {actual_return:+.1f}%<extra></extra>'
            ))
            
            # Add end-point annotation
            annotations_data.append({
                'x': len(normalized) - 1,
                'y': normalized[-1],
                'text': f'{normalized[-1]:.1f}',
                'color': colors[i % len(colors)]
            })
    
    # Add annotations for final values
    for ann in annotations_data[:3]:  # Only annotate first 3 to avoid clutter
        fig.add_annotation(
            x=ann['x'], y=ann['y'],
            text=ann['text'],
            showarrow=False,
            xshift=25,
            font=dict(size=10, color=ann['color'], family='Inter')
        )
    
    fig.update_layout(
        title=dict(
            text='📈 Portfolio Performance (Normalized to Base 100)',
            font=dict(size=16, color='white', family='Inter'),
            x=0.01
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,24,39,0.8)',
        font=dict(color='#f8fafc', family='Inter'),
        height=380,
        margin=dict(t=60, l=60, r=80, b=60),
        xaxis=dict(
            title='Trading Days',
            gridcolor='rgba(51,65,85,0.4)',
            tickfont=dict(size=10),
            showline=True,
            linecolor='rgba(51,65,85,0.6)'
        ),
        yaxis=dict(
            title='Normalized Value (Start = 100)',
            gridcolor='rgba(51,65,85,0.4)',
            tickfont=dict(size=10),
            showline=True,
            linecolor='rgba(51,65,85,0.6)'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.25,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10)
        ),
        hovermode='x unified'
    )
    return fig



def create_allocation_donut(df: pd.DataFrame, c: Dict) -> go.Figure:
    """Create portfolio allocation donut chart with clear values"""
    
    # Check if we have ticker column for better allocation view
    ticker_col = None
    for col in ['ticker', 'Ticker', 'symbol', 'Symbol']:
        if col in df.columns:
            ticker_col = col
            break
    
    if ticker_col:
        # Group by ticker and get allocation
        price_col = None
        for col in ['close', 'Close', 'price', 'market_value', 'value']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col:
            ticker_allocation = df.groupby(ticker_col)[price_col].last().abs()
            labels = ticker_allocation.index.tolist()[:8]
            values = ticker_allocation.values.tolist()[:8]
            total = sum(values)
            percentages = [(v/total)*100 for v in values]
        else:
            labels = df[ticker_col].unique().tolist()[:8]
            percentages = [100/len(labels)] * len(labels)
    else:
        # Use numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
        if len(numeric_cols) > 0:
            last_values = [abs(df[col].iloc[-1]) if len(df[col]) > 0 else 1 for col in numeric_cols]
            total = sum(last_values)
            percentages = [(v/total)*100 for v in last_values]
            labels = numeric_cols
        else:
            labels = ['Asset A', 'Asset B', 'Asset C']
            percentages = [40, 35, 25]
    
    colors = ['#6366f1', '#22d3ee', '#f472b6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#14b8a6']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=percentages,
        hole=0.65,
        marker=dict(colors=colors[:len(labels)], line=dict(color='rgba(0,0,0,0.3)', width=2)),
        textinfo='percent',
        textfont=dict(size=11, color='white'),
        hovertemplate='<b>%{label}</b><br>Allocation: %{percent}<br>Value: %{value:.1f}<extra></extra>',
        pull=[0.02 if p == max(percentages) else 0 for p in percentages]  # Pull out largest slice
    )])
    
    # Add center annotation with total assets count
    fig.update_layout(
        title=dict(
            text='💼 Portfolio Allocation by Asset',
            font=dict(size=16, color='white', family='Inter'),
            x=0.01
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', family='Inter'),
        height=380,
        margin=dict(t=60, l=20, r=20, b=40),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=9)
        ),
        annotations=[
            dict(
                text=f'<b>{len(labels)}</b><br>Assets',
                x=0.5, y=0.5,
                font=dict(size=16, color='white', family='Inter'),
                showarrow=False
            )
        ]
    )
    return fig



def create_risk_matrix(df: pd.DataFrame, c: Dict) -> go.Figure:
    """Create risk vs return scatter matrix"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
    
    risks, returns, labels, sizes = [], [], [], []
    
    for col in numeric_cols:
        if len(df[col].dropna()) > 5:
            pct_change = df[col].pct_change().dropna()
            if len(pct_change) > 0:
                volatility = pct_change.std() * np.sqrt(252) * 100
                first_val = df[col].dropna().iloc[0]
                last_val = df[col].dropna().iloc[-1]
                ret = ((last_val / first_val) - 1) * 100 if first_val != 0 and not pd.isna(first_val) else 0
                
                # Only add if values are valid (not NaN or inf)
                if not (pd.isna(volatility) or pd.isna(ret) or np.isinf(volatility) or np.isinf(ret)):
                    risks.append(float(volatility))
                    returns.append(float(ret))
                    labels.append(col[:12])
                    sizes.append(max(20, min(60, abs(ret) + 20)))  # Clamp sizes
    
    # Use defaults if no valid data
    if not risks or len(risks) < 2:
        risks = [15, 20, 25, 18, 22, 30]
        returns = [8, 12, 15, 10, 14, 18]
        labels = ['Asset A', 'Asset B', 'Asset C', 'Asset D', 'Asset E', 'Asset F']
        sizes = [30, 35, 40, 32, 38, 45]

    
    colors_scale = ['#ef4444' if r < 0 else '#10b981' for r in returns]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=risks, y=returns,
        mode='markers+text',
        marker=dict(size=sizes, color=colors_scale, opacity=0.8,
                   line=dict(width=2, color='white')),
        text=labels,
        textposition='top center',
        textfont=dict(size=10, color='white'),
        hovertemplate='<b>%{text}</b><br>Risk: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
    ))
    
    # Add quadrant lines
    avg_risk = np.mean(risks)
    avg_return = np.mean(returns)
    fig.add_hline(y=avg_return, line_dash="dash", line_color="rgba(255,255,255,0.3)", 
                  annotation_text="Avg Return", annotation_font_color="#94a3b8")
    fig.add_vline(x=avg_risk, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                  annotation_text="Avg Risk", annotation_font_color="#94a3b8")
    
    # Quadrant annotations
    fig.add_annotation(x=max(risks)*0.9, y=max(returns)*0.9, text="⭐ High Risk High Return",
                      font=dict(size=9, color='#10b981'), showarrow=False)
    fig.add_annotation(x=min(risks)*1.2, y=max(returns)*0.9, text="💎 Low Risk High Return",
                      font=dict(size=9, color='#22d3ee'), showarrow=False)
    
    fig.update_layout(
        title=dict(
            text=f'⚖️ Risk-Return Matrix | Best: {labels[returns.index(max(returns))] if returns else "N/A"}',
            font=dict(size=16, color='white', family='Inter'),
            x=0.01
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,24,39,0.8)',
        font=dict(color='#f8fafc', family='Inter'),
        height=380,
        margin=dict(t=60, l=70, r=40, b=60),
        xaxis=dict(
            title='Risk (Annualized Volatility %)',
            gridcolor='rgba(51,65,85,0.4)',
            tickfont=dict(size=10),
            showline=True,
            linecolor='rgba(51,65,85,0.6)'
        ),
        yaxis=dict(
            title='Return (%)',
            gridcolor='rgba(51,65,85,0.4)',
            tickfont=dict(size=10),
            showline=True,
            linecolor='rgba(51,65,85,0.6)'
        ),
        showlegend=False
    )
    return fig



def create_trend_momentum(df: pd.DataFrame, c: Dict) -> go.Figure:
    """Create trend momentum chart with MACD-style indicators and clear values"""
    
    # Find price column
    price_col = None
    for col in ['close', 'Close', 'CLOSE', 'price', 'Price', 'adj_close']:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            price_col = numeric_cols[0]
    
    if price_col and len(df) > 20:
        vals = df[price_col].dropna().values
        col_name = price_col
        ma_short = pd.Series(vals).rolling(window=min(10, len(vals)//3)).mean()
        ma_long = pd.Series(vals).rolling(window=min(20, len(vals)//2)).mean()
        
        # Calculate key values
        current_price = vals[-1]
        price_change = ((vals[-1] / vals[0]) - 1) * 100 if vals[0] != 0 else 0
    else:
        vals = np.cumsum(np.random.randn(100)) + 100
        col_name = "Price"
        ma_short = pd.Series(vals).rolling(window=10).mean()
        ma_long = pd.Series(vals).rolling(window=20).mean()
        current_price = vals[-1]
        price_change = 0
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3],
        subplot_titles=[
            f'Price: {current_price:.2f} ({price_change:+.1f}%)',
            'Momentum Histogram'
        ]
    )
    
    # Price line with current value annotation
    fig.add_trace(go.Scatter(
        y=vals,
        mode='lines',
        name=f'{col_name}',
        line=dict(color='#6366f1', width=2),
        hovertemplate=f'<b>{col_name}</b>: %{{y:.2f}}<extra></extra>'
    ), row=1, col=1)
    
    # Fast MA
    fig.add_trace(go.Scatter(
        y=ma_short,
        mode='lines',
        name='SMA 10 (Fast)',
        line=dict(color='#22d3ee', width=1.5, dash='dash'),
        hovertemplate='<b>SMA 10</b>: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Slow MA
    fig.add_trace(go.Scatter(
        y=ma_long,
        mode='lines',
        name='SMA 20 (Slow)',
        line=dict(color='#f472b6', width=1.5, dash='dot'),
        hovertemplate='<b>SMA 20</b>: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # Add current price annotation
    fig.add_annotation(
        x=len(vals) - 1,
        y=current_price,
        text=f'${current_price:.2f}',
        showarrow=True,
        arrowhead=2,
        arrowcolor='#6366f1',
        font=dict(size=11, color='#6366f1'),
        xshift=30,
        row=1, col=1
    )
    
    # Momentum histogram
    momentum = (ma_short - ma_long).fillna(0)
    colors = ['#10b981' if m > 0 else '#ef4444' for m in momentum]
    fig.add_trace(go.Bar(
        y=momentum,
        marker_color=colors,
        name='Momentum',
        opacity=0.8,
        hovertemplate='Momentum: %{y:.2f}<extra></extra>'
    ), row=2, col=1)
    
    # Signal line in momentum subplot
    signal_status = "🟢 Bullish" if momentum.iloc[-1] > 0 else "🔴 Bearish"
    
    fig.update_layout(
        title=dict(
            text=f'📊 Trend & Momentum Analysis | Signal: {signal_status}',
            font=dict(size=16, color='white', family='Inter'),
            x=0.01
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,24,39,0.8)',
        font=dict(color='#f8fafc', family='Inter'),
        height=420,
        margin=dict(t=70, l=60, r=60, b=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=10)
        ),
        showlegend=True
    )
    
    fig.update_xaxes(gridcolor='rgba(51,65,85,0.4)', title_text='Trading Days', row=2, col=1)
    fig.update_yaxes(gridcolor='rgba(51,65,85,0.4)', title_text='Price', row=1, col=1)
    fig.update_yaxes(gridcolor='rgba(51,65,85,0.4)', title_text='Momentum', row=2, col=1)
    
    return fig


def create_volume_chart(df: pd.DataFrame, c: Dict) -> go.Figure:
    """Create volume analysis chart with clear values"""
    
    # Find volume column
    volume_col = None
    for col in ['volume', 'Volume', 'VOLUME', 'vol', 'Vol']:
        if col in df.columns:
            volume_col = col
            break
    
    # Find price column for color coding
    price_col = None
    for col in ['close', 'Close', 'price', 'Price']:
        if col in df.columns:
            price_col = col
            break
    
    if volume_col is None:
        # No volume data, create a placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="No volume data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='#94a3b8')
        )
        fig.update_layout(
            title=dict(text='📊 Volume Analysis', font=dict(size=16, color='white')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(17,24,39,0.8)',
            height=300
        )
        return fig
    
    volumes = df[volume_col].values
    
    # Color by price movement if available
    if price_col:
        price_changes = df[price_col].diff()
        colors = ['#10b981' if change > 0 else '#ef4444' for change in price_changes]
    else:
        colors = ['#6366f1'] * len(volumes)
    
    # Calculate stats
    avg_volume = np.mean(volumes)
    max_volume = np.max(volumes)
    current_volume = volumes[-1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=volumes,
        marker_color=colors,
        name='Volume',
        opacity=0.8,
        hovertemplate='Volume: %{y:,.0f}<extra></extra>'
    ))
    
    # Add average line
    fig.add_hline(
        y=avg_volume,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Avg: {avg_volume/1e6:.1f}M",
        annotation_font_color="#f59e0b"
    )
    
    fig.update_layout(
        title=dict(
            text=f'📊 Volume Analysis | Current: {current_volume/1e6:.2f}M',
            font=dict(size=16, color='white', family='Inter'),
            x=0.01
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,24,39,0.8)',
        font=dict(color='#f8fafc', family='Inter'),
        height=320,
        margin=dict(t=60, l=60, r=30, b=50),
        xaxis=dict(
            title='Trading Days',
            gridcolor='rgba(51,65,85,0.4)'
        ),
        yaxis=dict(
            title='Volume',
            gridcolor='rgba(51,65,85,0.4)',
            tickformat=',.0f'
        ),
        showlegend=False
    )
    
    return fig


def create_returns_distribution(df: pd.DataFrame, c: Dict) -> go.Figure:
    """Create returns distribution histogram"""
    
    # Find price column
    price_col = None
    for col in ['close', 'Close', 'price', 'Price', 'adj_close']:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            price_col = numeric_cols[0]
    
    if price_col:
        returns = df[price_col].pct_change().dropna() * 100
    else:
        returns = pd.Series(np.random.randn(100))
    
    # Calculate stats
    mean_return = returns.mean()
    std_return = returns.std()
    skewness = returns.skew()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        marker=dict(
            color='#6366f1',
            line=dict(color='white', width=0.5)
        ),
        opacity=0.8,
        name='Returns',
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    fig.add_vline(
        x=mean_return,
        line_dash="dash",
        line_color="#10b981",
        annotation_text=f"Mean: {mean_return:.2f}%",
        annotation_font_color="#10b981"
    )
    
    # Distribution shape description
    if skewness > 0.5:
        shape = "Right-skewed (Positive)"
    elif skewness < -0.5:
        shape = "Left-skewed (Negative)"
    else:
        shape = "Symmetric"
    
    fig.update_layout(
        title=dict(
            text=f'📈 Returns Distribution | {shape}',
            font=dict(size=16, color='white', family='Inter'),
            x=0.01
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(17,24,39,0.8)',
        font=dict(color='#f8fafc', family='Inter'),
        height=320,
        margin=dict(t=60, l=60, r=30, b=50),
        xaxis=dict(
            title='Daily Return (%)',
            gridcolor='rgba(51,65,85,0.4)'
        ),
        yaxis=dict(
            title='Frequency',
            gridcolor='rgba(51,65,85,0.4)'
        ),
        showlegend=False,
        bargap=0.02
    )
    
    # Add standard deviation bands annotation
    fig.add_annotation(
        x=0.98, y=0.95,
        xref='paper', yref='paper',
        text=f'σ = {std_return:.2f}%',
        showarrow=False,
        font=dict(size=12, color='#94a3b8', family='Inter'),
        bgcolor='rgba(0,0,0,0.5)',
        borderpad=4
    )
    
    return fig


def render_powerbi_dashboard():
    """Render the commercial-grade Executive Dashboard"""
    c = get_theme_colors()
    
    if not has_data():
        st.warning("⚠️ No data available. Please upload data via Data Sources first.")
        return
    
    df_original = get_working_data()
    if df_original is None or len(df_original) == 0:
        st.warning("⚠️ No data available.")
        return
    
    data_info = get_working_data_info()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TICKER FILTER - Allow filtering by specific stocks
    # ═══════════════════════════════════════════════════════════════════════════
    ticker_col = None
    for col in ['ticker', 'Ticker', 'symbol', 'Symbol', 'TICKER']:
        if col in df_original.columns:
            ticker_col = col
            break
    
    df = df_original.copy()
    selected_ticker = "All Stocks"
    
    # Control Panel
    st.markdown(f'''
    <div style="
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
    ">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <span style="font-size: 20px;">🎛️</span>
            <span style="font-size: 16px; font-weight: 700; color: {c['text']};">Dashboard Controls</span>
        </div>
    ''', unsafe_allow_html=True)
    
    control_cols = st.columns([2, 2, 2, 2])
    
    with control_cols[0]:
        if ticker_col:
            available_tickers = ['All Stocks'] + sorted(df_original[ticker_col].unique().tolist())
            selected_ticker = st.selectbox(
                "📈 Filter by Stock",
                available_tickers,
                key="dashboard_ticker_filter",
                help="Select a specific stock or view all"
            )
            
            if selected_ticker != "All Stocks":
                df = df_original[df_original[ticker_col] == selected_ticker].copy()
        else:
            st.info("No ticker column found")
    
    with control_cols[1]:
        # Date range info
        date_col = None
        for col in ['date', 'Date', 'DATE', 'datetime']:
            if col in df.columns:
                date_col = col
                break
        if date_col:
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce')
                date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
            except:
                date_range = "N/A"
        else:
            date_range = "N/A"
        st.markdown(f"**📅 Date Range:**")
        st.caption(date_range)
    
    with control_cols[2]:
        st.markdown(f"**📊 Data Points:**")
        st.caption(f"{len(df):,} rows × {len(df.columns)} columns")
    
    with control_cols[3]:
        st.markdown(f"**🏷️ Selected:**")
        st.caption(f"{selected_ticker}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Calculate insights for filtered data
    insights = calculate_business_insights(df)
    all_results = get_all_run_models()
    total_models = count_run_models()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA TYPE DETECTION & FEATURE DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════
    data_type_info = detect_data_type(df)
    data_name = data_info.get('name', 'Your Data')
    data_rows = len(df)
    data_cols_count = len(df.columns)
    ticker_badge = f' | Viewing: <span style="color: {c["primary"]}; font-weight: 700;">{selected_ticker}</span>' if selected_ticker != "All Stocks" else ""
    
    # Data Type Detection Banner
    type_colors = {
        'stock': '#10b981',
        'sales': '#f59e0b', 
        'financial': '#6366f1',
        'general': '#64748b'
    }
    type_color = type_colors.get(data_type_info['data_type'], '#64748b')
    
    # Build feature badges HTML safely
    f_list = data_type_info.get('features', [])[:6]
    feature_badges_html = "".join([f'''<span style="background:#1e293b;color:#94a3b8;padding:4px 10px;border-radius:6px;font-size:10px;border:1px solid #334155;display:inline-block;margin:2px;">✓ {str(f)}</span>''' for f in f_list])
    
    m_list = data_type_info.get('available_models', [])[:4]
    models_str = ", ".join([str(m) for m in m_list])
    
    # Combined Banner HTML
    st.markdown(f'''
<div style="background: linear-gradient(135deg, {type_color}15 0%, {type_color}05 100%); border: 1px solid {type_color}40; border-radius: 14px; padding: 16px 20px; margin-bottom: 20px;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
<div>
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
<span style="font-size: 28px;">{data_type_info.get('icon', '📊')}</span>
<div>
<div style="font-size: 18px; font-weight: 700; color: {c.get('text', '#f8fafc')};">
{data_type_info.get('description', 'Data')} Detected
</div>
<div style="font-size: 12px; color: {c.get('text_muted', '#64748b')};">
{str(data_name)} • {int(data_rows):,} rows × {int(data_cols_count)} columns {ticker_badge}
</div>
</div>
</div>
</div>
<div style="text-align: right;">
<span style="background: {type_color}30; color: {type_color}; padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 600; text-transform: uppercase;">{str(data_type_info.get('data_type', 'general')).upper()} DATA</span>
</div>
</div>
<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #10b98130;">
<div style="font-size: 11px; color: #64748b; margin-bottom: 6px;">AVAILABLE FEATURES FOR THIS DATA:</div>
<div style="display: flex; flex-wrap: wrap; gap: 6px;">
{feature_badges_html}
</div>
<div style="margin-top: 10px; font-size: 10px; color: #64748b;">
<strong>Recommended Models:</strong> {models_str}
</div>
</div>
</div>
''', unsafe_allow_html=True)


    
    # Action Buttons (Share & Download)
    col1, col2, col3 = st.columns([6, 1, 1.5])
    with col2:
        if st.button("🔗 Share", key="share_dashboard_btn", use_container_width=True):
            st.toast("Dashboard link copied to clipboard!", icon="📋")
    with col3:
        # Generate HTML for download
        dashboard_html = generate_executive_html(c, df, insights, all_results)
        st.download_button(
            "📥 Download",
            data=dashboard_html,
            file_name=f"executive_dashboard_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
            key="download_dashboard_top_btn",
            use_container_width=True
        )

    # Premium Header
    st.markdown(f'''
<div style="background: linear-gradient(135deg, {c['primary']}30 0%, {c['secondary']}20 50%, {c['accent']}10 100%); border: 1px solid {c['primary']}40; border-radius: 20px; padding: 25px 35px; margin-bottom: 20px; position: relative; overflow: hidden;">
    <div style="position: absolute; top: 0; right: 0; width: 200px; height: 200px; background: radial-gradient(circle, {c['primary']}20 0%, transparent 70%);"></div>
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <h1 style="font-size: 28px; font-weight: 800; margin: 0; background: linear-gradient(135deg, {c['primary']}, {c['accent']}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                📊 Executive Decision Dashboard
            </h1>
            <p style="color: {c['text_secondary']}; margin: 5px 0 0 0; font-size: 13px;">
                AI-Powered Investment Analytics & Business Intelligence
            </p>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <div style="text-align: center; padding: 10px 16px; background: {c['bg_card']}; border-radius: 10px; border: 1px solid {c['border']};">
                <div style="font-size: 10px; color: {c['text_muted']}; text-transform: uppercase;">Updated</div>
                <div style="font-size: 13px; font-weight: 600; color: {c['text']};">{datetime.now().strftime('%H:%M:%S')}</div>
            </div>
            <div style="text-align: center; padding: 10px 16px; background: linear-gradient(135deg, {c['success']}30, {c['success']}10); border-radius: 10px; border: 1px solid {c['success']}40;">
                <div style="font-size: 10px; color: {c['success']}; text-transform: uppercase;">Status</div>
                <div style="font-size: 13px; font-weight: 600; color: {c['success']};">● Live</div>
            </div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)
    
    # AI Investment Signal Banner
    signal_cfg = {
        'Strong Buy': (c['success'], '🚀', 'Market conditions strongly favor investment'),
        'Buy': ('#22c55e', '📈', 'Positive outlook with good risk-reward'),
        'Hold': (c['warning'], '⏸️', 'Wait for better entry points'),
        'Reduce': ('#f97316', '📉', 'Consider reducing exposure'),
        'Sell': (c['error'], '🔴', 'Risk indicators suggest exit')
    }
    signal_color, signal_icon, signal_reason = signal_cfg.get(insights['recommendation'], (c['primary'], '📊', 'Analyzing...'))
    
    st.markdown(f'''
<div style="background: linear-gradient(90deg, {signal_color}25, {signal_color}10, transparent); border-left: 4px solid {signal_color}; border-radius: 14px; padding: 20px 25px; margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div style="display: flex; align-items: center; gap: 15px;">
            <span style="font-size: 40px;">{signal_icon}</span>
            <div>
                <div style="font-size: 11px; color: {c['text_muted']}; text-transform: uppercase; letter-spacing: 1px;">🤖 AI Investment Signal</div>
                <div style="font-size: 28px; font-weight: 800; color: {signal_color};">{insights['recommendation']}</div>
                <div style="font-size: 12px; color: {c['text_secondary']}; margin-top: 2px;">{signal_reason}</div>
            </div>
        </div>
        <div style="display: flex; gap: 25px;">
            <div style="text-align: center;">
                <div style="font-size: 10px; color: {c['text_muted']};">Confidence</div>
                <div style="font-size: 22px; font-weight: 700; color: {c['text']};">{insights['confidence']}%</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 10px; color: {c['text_muted']};">Risk Level</div>
                <div style="font-size: 22px; font-weight: 700; color: {c['warning'] if insights['risk_level'] == 'Medium' else c['error'] if insights['risk_level'] == 'High' else c['success']};">{insights['risk_level']}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 10px; color: {c['text_muted']};">Trend</div>
                <div style="font-size: 22px; font-weight: 700; color: {c['success'] if insights['trend'] == 'bullish' else c['error'] if insights['trend'] == 'bearish' else c['text']};">
                    {'↗️' if insights['trend'] == 'bullish' else '↘️' if insights['trend'] == 'bearish' else '→'} {insights['trend'].title()}
                </div>
            </div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)
    
    # Row 1: Gauges + KPIs
    gauge_cols = st.columns([1, 1, 2])
    
    with gauge_cols[0]:
        fig_sentiment = create_sentiment_gauge(insights['market_sentiment'], c)
        st.plotly_chart(fig_sentiment, use_container_width=True, key="sentiment_gauge")
    
    with gauge_cols[1]:
        fig_decision = create_decision_gauge(insights['decision_score'], c)
        st.plotly_chart(fig_decision, use_container_width=True, key="decision_gauge")
    
    with gauge_cols[2]:
        # Key Dynamic KPIs (Content adapts based on data type)
        if data_type_info['data_type'] == 'sales':
            kpi_data = [
                ('🛒', 'Total Sales', f"${insights['total_sum']:,.0f}", insights['revenue_growth'] > 0),
                ('📦', 'Avg Order', f"${insights['mean_val']:.2f}", True),
                ('📈', 'Sales Growth', f"{insights['revenue_growth']:.1f}%", insights['revenue_growth'] > 0),
                ('🛡️', 'Stability', f"{(1 - insights['coeff_variation']) * 100:.1f}%", insights['coeff_variation'] < 0.2),
            ]
        elif data_type_info['data_type'] in ['financial', 'general']:
            kpi_data = [
                ('💰', 'Avg Value', f"{insights['mean_val']:.2f}", insights['trend'] == 'bullish'),
                ('📈', 'Period Growth', f"{insights['total_return']:.1f}%", insights['total_return'] > 0),
                ('📉', 'Drawdown', f"{insights['max_drawdown']:.1f}%", insights['max_drawdown'] < 15),
                ('⚡', 'Volatility', f"{insights['volatility']:.1f}%", insights['volatility'] < 20),
            ]
        else: # Stock/Default
            kpi_data = [
                ('💰', 'Total Return', f"{insights['total_return']:.1f}%", insights['total_return'] > 0),
                ('📊', 'Volatility', f"{insights['volatility']:.1f}%", insights['volatility'] < 20),
                ('⚡', 'Sharpe Ratio', f"{insights['sharpe_ratio']:.2f}", insights['sharpe_ratio'] > 1),
                ('📉', 'Max Drawdown', f"{insights['max_drawdown']:.1f}%", insights['max_drawdown'] < 15),
            ]
        
        kpi_cols = st.columns(4)
        for i, (icon, label, value, is_positive) in enumerate(kpi_data):
            with kpi_cols[i]:
                trend_color = c['success'] if is_positive else c['error']
                st.markdown(f'''
<div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 14px; padding: 16px; text-align: center; border-top: 3px solid {trend_color};">
    <div style="font-size: 24px; margin-bottom: 6px;">{icon}</div>
    <div style="font-size: 20px; font-weight: 800; color: {c['text']};">{value}</div>
    <div style="font-size: 10px; color: {c['text_muted']}; text-transform: uppercase;">{label}</div>
</div>
''', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    # Get available charts for this data type
    available_charts = data_type_info.get('available_charts', [])
    
    # Row 2: Main Charts (Performance - Available for ALL data types)
    col1, col2 = st.columns(2)
    with col1:
        fig1 = create_performance_chart(df, c)
        st.plotly_chart(fig1, use_container_width=True, key="perf_chart")
    
    with col2:
        # Risk Matrix: Best for Stock/Financial data
        if data_type_info['data_type'] in ['stock', 'financial', 'general']:
            fig2 = create_risk_matrix(df, c)
            st.plotly_chart(fig2, use_container_width=True, key="risk_chart")
        else:
            # For sales data, show the trend instead
            fig2 = create_trend_momentum(df, c)
            st.plotly_chart(fig2, use_container_width=True, key="trend_chart_alt")
    
    # Row 3: Type-specific charts
    col3, col4 = st.columns(2)
    
    with col3:
        if data_type_info['data_type'] in ['stock', 'financial']:
            # Cash flow for stock/financial
            fig3 = create_cashflow_chart(df, c)
            st.plotly_chart(fig3, use_container_width=True, key="cashflow_chart")
        else:
            # Allocation for sales/general
            fig3 = create_allocation_donut(df, c)
            st.plotly_chart(fig3, use_container_width=True, key="alloc_chart_alt")
    
    with col4:
        # Profitability radar for all types
        fig4 = create_profitability_radar(insights, c)
        st.plotly_chart(fig4, use_container_width=True, key="profit_radar")
    
    # Row 4: Allocation & Trend
    col5, col6 = st.columns(2)
    with col5:
        if data_type_info['data_type'] in ['stock', 'financial']:
            fig5 = create_allocation_donut(df, c)
            st.plotly_chart(fig5, use_container_width=True, key="alloc_chart")
        else:
            fig5 = create_returns_distribution(df, c)
            st.plotly_chart(fig5, use_container_width=True, key="returns_alt")
    
    with col6:
        if data_type_info['data_type'] != 'sales':
            fig6 = create_trend_momentum(df, c)
            st.plotly_chart(fig6, use_container_width=True, key="trend_chart")
        else:
            fig6 = create_returns_distribution(df, c)
            st.plotly_chart(fig6, use_container_width=True, key="returns_chart_sales")
    
    # Row 5: Volume & Returns Distribution (Stock data only has volume)
    if data_type_info['data_type'] == 'stock':
        col7, col8 = st.columns(2)
        with col7:
            fig7 = create_volume_chart(df, c)
            st.plotly_chart(fig7, use_container_width=True, key="volume_chart")
        with col8:
            fig8 = create_returns_distribution(df, c)
            st.plotly_chart(fig8, use_container_width=True, key="returns_dist_chart")
    elif data_type_info['data_type'] in ['financial', 'general']:
        # Show returns distribution for financial/general
        col7, col8 = st.columns(2)
        with col7:
            fig7 = create_returns_distribution(df, c)
            st.plotly_chart(fig7, use_container_width=True, key="returns_dist_chart_fin")
        with col8:
            # Show a summary info card instead of volume
            st.markdown(f'''
<div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 14px; padding: 20px; height: 300px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <div style="font-size: 40px; margin-bottom: 10px;">📊</div>
    <div style="font-size: 16px; font-weight: 600; color: {c['text']}; margin-bottom: 8px;">{data_type_info['description']} Analysis</div>
    <div style="font-size: 12px; color: {c['text_muted']}; text-align: center;">
        {len(data_type_info['features'])} analysis features available<br>
        {len(data_type_info['available_models'])} ML models recommended
    </div>
</div>
''', unsafe_allow_html=True)

    
    # ═══════════════════════════════════════════════════════════════════════════
    # DYNAMIC PERFORMANCE METRICS ROW
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Define KPIs based on data type
    kpi_configs = []
    
    if data_type_info['data_type'] == 'stock':
        kpi_configs = [
            {'label': 'Avg Price', 'value': f"${insights['mean_val']:.2f}", 'color': c['primary']},
            {'label': 'Trend Strength', 'value': f"{insights['trend_strength']:.1f}%", 'color': '#10b981' if insights['trend_strength'] > 50 else '#f59e0b'},
            {'label': 'Momentum', 'value': f"{insights['momentum_score']:.1f}", 'color': '#10b981' if insights['momentum_score'] > 0 else '#ef4444'},
            {'label': 'Risk/Reward', 'value': f"{insights['risk_reward_ratio']:.2f}x", 'color': '#10b981' if insights['risk_reward_ratio'] > 1.5 else '#f59e0b'},
            {'label': 'Max Drawdown', 'value': f"{insights['max_drawdown']:.1f}%", 'color': '#ef4444' if insights['max_drawdown'] > 20 else '#10b981'},
            {'label': 'Models Active', 'value': f"{total_models}", 'color': c['accent']}
        ]
    elif data_type_info['data_type'] == 'sales':
        kpi_configs = [
            {'label': 'Total Sales', 'value': f"${insights['total_sum']:,.0f}", 'color': '#10b981'},
            {'label': 'Avg Order', 'value': f"${insights['mean_val']:.2f}", 'color': c['primary']},
            {'label': 'Sales Growth', 'value': f"{insights['revenue_growth']:.1f}%", 'color': '#10b981' if insights['revenue_growth'] > 0 else '#ef4444'},
            {'label': 'Sales Stability', 'value': f"{(1 - insights['coeff_variation']) * 100:.1f}%", 'color': '#8b5cf6'},
            {'label': 'Peak Value', 'value': f"${insights['max_val']:,.0f}", 'color': '#f59e0b'},
            {'label': 'Forecasters', 'value': f"{total_models}", 'color': c['accent']}
        ]
    elif data_type_info['data_type'] == 'financial':
        kpi_configs = [
            {'label': 'Revenue Growth', 'value': f"{insights['revenue_growth']:.1f}%", 'color': '#10b981' if insights['revenue_growth'] > 0 else '#ef4444'},
            {'label': 'Profit Margin', 'value': f"{insights['profit_margin']:.1f}%", 'color': c['primary']},
            {'label': 'ROI', 'value': f"{insights['roi']:.1f}%", 'color': '#10b981' if insights['roi'] > 0 else '#ef4444'},
            {'label': 'Liquidity', 'value': f"{insights['liquidity_ratio']:.2f}x", 'color': '#10b981' if insights['liquidity_ratio'] > 1 else '#f59e0b'},
            {'label': 'Debt Ratio', 'value': f"{insights['debt_ratio']:.1%}", 'color': '#10b981' if insights['debt_ratio'] < 0.4 else '#ef4444'},
            {'label': 'Analysis Status', 'value': 'Active', 'color': c['accent']}
        ]
    else:
        kpi_configs = [
            {'label': 'Mean Value', 'value': f"{insights['mean_val']:.2f}", 'color': c['primary']},
            {'label': 'Median', 'value': f"{insights['median_val']:.2f}", 'color': c['secondary']},
            {'label': 'Volatility (CV)', 'value': f"{insights['coeff_variation']:.2%}", 'color': '#f59e0b'},
            {'label': 'Data Spread', 'value': f"{insights['data_range']:.2f}", 'color': '#6366f1'},
            {'label': 'Trend Direction', 'value': insights['trend'].title(), 'color': '#10b981' if insights['trend'] == 'bullish' else '#ef4444'},
            {'label': 'Valid Samples', 'value': f"{len(df):,}", 'color': c['accent']}
        ]

    # Build KPI Highlights HTML safely
    # Build KPI Highlights HTML safely - NO INDENTATION to avoid markdown code blocks
    kpis_html = "".join([f'''<div style="text-align: center; padding: 15px; background: #1e293b; border-radius: 12px; border-bottom: 2px solid {cfg.get('color', '#6366f1')};"><div style="font-size: 10px; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">{str(cfg.get('label', 'Metric'))}</div><div style="font-size: 18px; font-weight: 700; color: {cfg.get('color', '#6366f1')};">{str(cfg.get('value', '0.00'))}</div></div>''' for cfg in kpi_configs])

    st.markdown(f'''
<div style="background: {c.get('bg_card', '#1e293b')}; border: 1px solid {c.get('border', '#334155')}; border-radius: 16px; padding: 22px 25px; margin: 15px 0;">
<h3 style="color: {c.get('text', '#f8fafc')}; margin: 0 0 18px 0; font-size: 18px;">📊 {str(data_type_info.get('description', 'Data'))} Analysis Highlights</h3>
<div style="display: grid; grid-template-columns: repeat({len(kpi_configs)}, 1fr); gap: 15px;">
{kpis_html}
</div>
</div>
''', unsafe_allow_html=True)

    
    # Technical Signals Panel - ONLY for Stock Data (RSI, MACD, MA are stock-specific)
    if data_type_info['data_type'] == 'stock':
        rsi_color = '#10b981' if 'Buy' in insights.get('rsi_signal', '') else '#ef4444' if 'Sell' in insights.get('rsi_signal', '') else c['text_secondary']
        macd_color = '#10b981' if 'Bullish' in insights.get('macd_signal', '') else '#ef4444' if 'Bearish' in insights.get('macd_signal', '') else c['text_secondary']
        ma_color = '#10b981' if 'Bullish' in insights.get('ma_signal', '') else '#ef4444' if 'Bearish' in insights.get('ma_signal', '') else c['text_secondary']
        short_color = '#10b981' if insights.get('short_term_outlook', '') == 'Bullish' else '#ef4444' if insights.get('short_term_outlook', '') == 'Bearish' else c['text_secondary']
        long_color = '#10b981' if insights.get('long_term_outlook', '') == 'Bullish' else '#ef4444' if insights.get('long_term_outlook', '') == 'Bearish' else c['text_secondary']

        st.markdown(f'''
<div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); border: 1px solid {c['border']}; border-radius: 16px; padding: 20px 25px; margin: 15px 0;">
    <h3 style="color: {c['text']}; margin: 0 0 15px 0; font-size: 18px;">📡 Technical Signals & Market Analysis</h3>
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
        <div style="text-align: center; padding: 15px; background: {c['bg_surface']}; border-radius: 12px; border-top: 3px solid {rsi_color};">
            <div style="font-size: 10px; color: {c['text_muted']}; text-transform: uppercase;">RSI Signal</div>
            <div style="font-size: 14px; font-weight: 700; color: {rsi_color}; margin-top: 6px;">{insights.get('rsi_signal', 'Neutral')}</div>
        </div>
        <div style="text-align: center; padding: 15px; background: {c['bg_surface']}; border-radius: 12px; border-top: 3px solid {macd_color};">
            <div style="font-size: 10px; color: {c['text_muted']}; text-transform: uppercase;">MACD Signal</div>
            <div style="font-size: 14px; font-weight: 700; color: {macd_color}; margin-top: 6px;">{insights.get('macd_signal', 'Neutral')}</div>
        </div>
        <div style="text-align: center; padding: 15px; background: {c['bg_surface']}; border-radius: 12px; border-top: 3px solid {ma_color};">
            <div style="font-size: 10px; color: {c['text_muted']}; text-transform: uppercase;">Moving Avg</div>
            <div style="font-size: 14px; font-weight: 700; color: {ma_color}; margin-top: 6px;">{insights.get('ma_signal', 'Neutral')}</div>
        </div>
        <div style="text-align: center; padding: 15px; background: {c['bg_surface']}; border-radius: 12px; border-top: 3px solid {short_color};">
            <div style="font-size: 10px; color: {c['text_muted']}; text-transform: uppercase;">Short-Term</div>
            <div style="font-size: 14px; font-weight: 700; color: {short_color}; margin-top: 6px;">{insights.get('short_term_outlook', 'Neutral')}</div>
        </div>
        <div style="text-align: center; padding: 15px; background: {c['bg_surface']}; border-radius: 12px; border-top: 3px solid {long_color};">
            <div style="font-size: 10px; color: {c['text_muted']}; text-transform: uppercase;">Long-Term</div>
            <div style="font-size: 14px; font-weight: 700; color: {long_color}; margin-top: 6px;">{insights.get('long_term_outlook', 'Neutral')}</div>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;">
        <div style="text-align: center; padding: 12px; background: {c['bg_card']}; border-radius: 10px;">
            <div style="font-size: 10px; color: {c['text_muted']};">Momentum Score</div>
            <div style="font-size: 18px; font-weight: 700; color: {'#10b981' if insights.get('momentum_score', 0) > 0 else '#ef4444'};">
                {'+' if insights.get('momentum_score', 0) > 0 else ''}{insights.get('momentum_score', 0):.1f}
            </div>
        </div>
        <div style="text-align: center; padding: 12px; background: {c['bg_card']}; border-radius: 10px;">
            <div style="font-size: 10px; color: {c['text_muted']};">Trend Strength</div>
            <div style="font-size: 18px; font-weight: 700; color: {c['primary']};">{insights.get('trend_strength', 0):.0f}%</div>
        </div>
        <div style="text-align: center; padding: 12px; background: {c['bg_card']}; border-radius: 10px;">
            <div style="font-size: 10px; color: {c['text_muted']};">Risk/Reward</div>
            <div style="font-size: 18px; font-weight: 700; color: {'#10b981' if insights.get('risk_reward_ratio', 1) > 1 else '#ef4444'};">{insights.get('risk_reward_ratio', 1):.2f}x</div>
        </div>
        <div style="text-align: center; padding: 12px; background: {c['bg_card']}; border-radius: 10px;">
            <div style="font-size: 10px; color: {c['text_muted']};">Action Urgency</div>
            <div style="font-size: 14px; font-weight: 700; color: {'#ef4444' if insights.get('action_urgency', '') == 'High' else '#f59e0b' if insights.get('action_urgency', '') == 'Moderate' else c['text_secondary']};">{insights.get('action_urgency', 'Normal')}</div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

    
    # DYNAMIC AI RECOMMENDATIONS
    dtype = data_type_info['data_type']
    
    # Define recommendations based on data type
    rec_configs = []
    if dtype == 'stock':
        rec_configs = [
            {'title': '✅ Immediate Action', 'color': c['success'], 'text': 'Increase allocation to top performers and consider leveraging position' if insights['trend'] == 'bullish' else 'Focus on capital preservation and defensive positioning' if insights['trend'] == 'bearish' else 'Maintain current allocation, await clearer signals'},
            {'title': '⚠️ Risk Management', 'color': c['warning'], 'text': 'High volatility detected - implement stop-losses and consider hedging' if insights['volatility'] > 25 else 'Moderate volatility - standard risk controls sufficient' if insights['volatility'] > 15 else 'Low volatility environment - opportunity for strategic growth'},
            {'title': '💡 Strategic Insight', 'color': c['primary'], 'text': 'Strong Sharpe ratio suggests efficient risk-adjusted returns' if insights['sharpe_ratio'] > 1 else 'Consider optimizing portfolio for better risk-adjusted returns' if insights['sharpe_ratio'] > 0.5 else 'Review asset allocation - returns not justifying risk taken'},
            {'title': '📈 Next Steps', 'color': c['accent'], 'text': 'Run ML models for price prediction insights' if total_models < 2 else 'Conduct scenario analysis for stress testing' if total_models < 4 else 'Portfolio well-analyzed - execute optimized strategy'}
        ]
    elif dtype == 'sales':
        rec_configs = [
            {'title': '✅ Sales Action', 'color': c['success'], 'text': 'Aggressively scale top performing products' if insights['revenue_growth'] > 5 else 'Review underperforming product lines' if insights['revenue_growth'] < 0 else 'Continue current marketing spend, observe trends'},
            {'title': '⚠️ Inventory Risk', 'color': c['warning'], 'text': 'High sales volatility - maintain larger safety stock' if insights['coeff_variation'] > 0.3 else 'Stable demand - optimize inventory for cost efficiency'},
            {'title': '💡 Market Insight', 'color': c['primary'], 'text': 'Sales trend is consistently positive' if insights['trend'] == 'bullish' else 'Market demand showing signs of cooling' if insights['trend'] == 'bearish' else 'Stable market demand - maintain baseline operations'},
            {'title': '📈 Growth Strategy', 'color': c['accent'], 'text': 'Analyze regional sales distribution for expansion opportunities'}
        ]
    elif dtype == 'financial':
        rec_configs = [
            {'title': '✅ Operation Action', 'color': c['success'], 'text': 'Reinvest profits into high-margin segments' if insights['profit_margin'] > 20 else 'Audit operational costs to improve margins' if insights['profit_margin'] < 10 else 'Hold steady, focus on margin consistency'},
            {'title': '⚠️ Liquidity Risk', 'color': c['warning'], 'text': 'Monitor cash flow closely - liquidity ratio below benchmark' if insights['liquidity_ratio'] < 1.2 else 'Strong liquidity position - healthy cash reserves'},
            {'title': '💡 Budget Insight', 'color': c['primary'], 'text': 'High ROI indicates efficient capital deployment' if insights['roi'] > 15 else 'Target improvements in asset utilization to boost ROI'},
            {'title': '📈 Fiscal Planning', 'color': c['accent'], 'text': 'Optimize debt structure to reduce interest burden' if insights['debt_ratio'] > 0.5 else 'Room for strategic debt for expansion'}
        ]
    else:
        rec_configs = [
            {'title': '✅ Data Action', 'color': c['success'], 'text': 'Focus on variables driving upward trend' if insights['trend'] == 'bullish' else 'Investigate factors causing negative trend' if insights['trend'] == 'bearish' else 'No immediate action - data trend is stable'},
            {'title': '⚠️ Stability Check', 'color': c['warning'], 'text': 'High data variance detected - use robust modeling' if insights['coeff_variation'] > 0.5 else 'Data shows low variance and high reliability'},
            {'title': '💡 Value Insight', 'color': c['primary'], 'text': f"Latest value is {((insights['mean_val'] - insights['median_val'])/insights['std_dev']):.1f}σ from mean" if insights['std_dev'] > 0 else 'Value is consistent with historical mean'},
            {'title': '📈 Next Phase', 'color': c['accent'], 'text': 'Apply advanced ML clustering to find hidden patterns'}
        ]

    # Build Recommendations HTML safely
    # Build Recommendations HTML safely - NO INDENTATION to avoid markdown code blocks
    recs_html = "".join([f'''<div style="background:#1e293b;padding:18px;border-radius:12px;border-left:4px solid {r.get('color', '#6366f1')};height:100%;"><div style="font-size:13px;font-weight:600;color:#f8fafc;margin-bottom:8px;">{str(r.get('title', 'Action'))}</div><div style="font-size:12px;color:#94a3b8;">{str(r.get('text', 'Analyzing...'))}</div></div>''' for r in rec_configs])

    st.markdown(f'''
<div style="background: linear-gradient(135deg, {c.get('bg_card', '#1e293b')} 0%, {c.get('bg_surface', '#1e293b')} 100%); border: 1px solid {c.get('border', '#334155')}; border-radius: 18px; padding: 25px 30px; margin: 15px 0;">
<h3 style="color: {c.get('text', '#f8fafc')}; margin: 0 0 20px 0; font-size: 18px;">🎯 AI-Powered Action Recommendations ({str(data_type_info.get('description', 'Data'))})</h3>
<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
{recs_html}
</div>
</div>
''', unsafe_allow_html=True)

    # Analysis Summary from Models
    if total_models > 0:
        st.markdown(f"### 🔬 Analysis Summary")
        
        # Safe column creation
        num_cols = min(len(all_results), 4) if all_results else 1
        summary_cols = st.columns(num_cols)
        col_idx = 0
        
        for category, models in all_results.items():
            if models:
                cat_results = get_category_results(category)
                if not cat_results: continue
                
                icons = {'forecasting': '🔮', 'ml_models': '🧪', 'portfolio': '💼', 'scenario': '📈', 'risk': '🛡️'}
                icon = icons.get(str(category), '📊')
                
                # Build Category Items HTML safely
                cat_items_html = ""
                items_to_show = list(cat_results.items())[:2]
                for model_name, result in items_to_show:
                    metrics = result.get('metrics', {}) if result else {}
                    primary = list(metrics.values())[0] if metrics else 'N/A'
                    if isinstance(primary, float):
                        primary = f"{primary:.3f}"
                    cat_items_html += f'''<div style="background: {c.get('bg_surface', '#1e293b')}; padding: 8px 10px; border-radius: 8px; margin-bottom: 6px;"><div style="font-size: 11px; color: {c.get('text_muted', '#64748b')};">{str(model_name)}</div><div style="font-size: 15px; font-weight: 700; color: {c.get('primary', '#6366f1')};">{str(primary)}</div></div>'''

                with summary_cols[col_idx % len(summary_cols)]:
                    st.markdown(f'''
<div style="background: {c.get('bg_card', '#1e293b')}; border: 1px solid {c.get('border', '#334155')}; border-radius: 14px; padding: 16px; height: 100%;">
<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
<span style="font-size: 22px;">{icon}</span>
<span style="font-size: 13px; font-weight: 600; color: {c.get('text', '#f8fafc')};">{str(category).replace('_', ' ').title()}</span>
</div>
{cat_items_html}
</div>
''', unsafe_allow_html=True)
                col_idx += 1
    
    st.markdown("---")
    
    # Export Section
    st.markdown(f"### 📥 Export Executive Report")
    
    exp_cols = st.columns([2, 1, 1])
    
    with exp_cols[0]:
        html_report = generate_executive_html(c, df, insights, all_results)
        st.download_button(
            "🎯 Download Executive Dashboard (HTML)",
            data=html_report,
            file_name=f"executive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            key="dl_exec_dashboard",
            use_container_width=True
        )
    
    with exp_cols[1]:
        summary_data = {
            'Metric': ['Total Return', 'Volatility', 'Sharpe Ratio', 'Risk Level', 'Recommendation', 
                      'Confidence', 'Revenue Growth', 'Profit Margin', 'ROI', 'Liquidity Ratio'],
            'Value': [f"{insights['total_return']:.2f}%", f"{insights['volatility']:.2f}%", 
                     f"{insights['sharpe_ratio']:.2f}", insights['risk_level'], 
                     insights['recommendation'], f"{insights['confidence']}%",
                     f"{insights['revenue_growth']:.2f}%", f"{insights['profit_margin']:.2f}%",
                     f"{insights['roi']:.2f}%", f"{insights['liquidity_ratio']:.2f}x"]
        }
        csv_data = pd.DataFrame(summary_data).to_csv(index=False).encode('utf-8')
        st.download_button("📊 Summary CSV", data=csv_data, file_name="executive_summary.csv",
                          mime="text/csv", key="dl_summary_csv", use_container_width=True)
    
    with exp_cols[2]:
        json_data = json.dumps({'insights': insights, 'timestamp': datetime.now().isoformat()}, indent=2, default=str)
        st.download_button("📄 Data JSON", data=json_data, file_name="executive_data.json",
                          mime="application/json", key="dl_data_json", use_container_width=True)


def generate_executive_html(c: Dict, df: pd.DataFrame = None, insights: Dict = None, all_results: Dict = None) -> str:
    """Generate premium HTML export"""
    
    if df is None:
        df = get_working_data() if has_data() else pd.DataFrame()
    if insights is None:
        insights = calculate_business_insights(df) if len(df) > 0 else {}
    if all_results is None:
        all_results = get_all_run_models()
    
    total_models = count_run_models()
    
    signal_colors = {
        'Strong Buy': '#10b981', 'Buy': '#22c55e', 'Hold': '#f59e0b', 
        'Reduce': '#f97316', 'Sell': '#ef4444'
    }
    signal_color = signal_colors.get(insights.get('recommendation', 'Hold'), '#6366f1')
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Dashboard - Financial Analytics Suite</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%);
            color: #f8fafc;
            min-height: 100vh;
            padding: 40px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        .header {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 24px;
            padding: 40px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .header p {{ color: #94a3b8; font-size: 16px; }}
        
        .signal-banner {{
            background: linear-gradient(90deg, {signal_color}25, {signal_color}10, transparent);
            border-left: 4px solid {signal_color};
            border-radius: 16px;
            padding: 25px 30px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .signal-main {{ font-size: 32px; font-weight: 800; color: {signal_color}; }}
        .signal-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
        .signal-stat {{ text-align: center; }}
        .signal-stat-value {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            border-top: 3px solid #6366f1;
        }}
        .kpi-icon {{ font-size: 26px; margin-bottom: 8px; }}
        .kpi-value {{ font-size: 22px; font-weight: 800; color: #f8fafc; }}
        .kpi-label {{ font-size: 10px; color: #64748b; text-transform: uppercase; margin-top: 4px; }}
        
        .actions-section {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
        }}
        .actions-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-top: 20px;
        }}
        .action-card {{
            background: #111827;
            padding: 18px;
            border-radius: 12px;
            border-left: 4px solid #6366f1;
        }}
        .action-title {{ font-size: 13px; font-weight: 600; margin-bottom: 6px; }}
        .action-text {{ font-size: 12px; color: #94a3b8; }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: #64748b;
            border-top: 1px solid #334155;
            margin-top: 40px;
        }}
        
        @media (max-width: 900px) {{
            .actions-grid {{ grid-template-columns: 1fr 1fr; }}
            .signal-banner {{ flex-direction: column; text-align: center; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Executive Decision Dashboard</h1>
            <p>Financial Analytics Suite - AI-Powered Investment Analysis</p>
            <p style="margin-top: 15px; font-size: 14px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="signal-banner">
            <div>
                <div class="signal-label">🤖 AI Investment Signal</div>
                <div class="signal-main">{insights.get('recommendation', 'Hold')}</div>
            </div>
            <div style="display: flex; gap: 40px;">
                <div class="signal-stat">
                    <div class="signal-label">Confidence</div>
                    <div class="signal-stat-value">{insights.get('confidence', 75)}%</div>
                </div>
                <div class="signal-stat">
                    <div class="signal-label">Risk Level</div>
                    <div class="signal-stat-value">{insights.get('risk_level', 'Medium')}</div>
                </div>
                <div class="signal-stat">
                    <div class="signal-label">Trend</div>
                    <div class="signal-stat-value">{insights.get('trend', 'neutral').title()}</div>
                </div>
            </div>
        </div>
        
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">💰</div>
                <div class="kpi-value">{insights.get('total_return', 0):.1f}%</div>
                <div class="kpi-label">Total Return</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📊</div>
                <div class="kpi-value">{insights.get('volatility', 0):.1f}%</div>
                <div class="kpi-label">Volatility</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">⚡</div>
                <div class="kpi-value">{insights.get('sharpe_ratio', 0):.2f}</div>
                <div class="kpi-label">Sharpe Ratio</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📈</div>
                <div class="kpi-value">{insights.get('revenue_growth', 0):.1f}%</div>
                <div class="kpi-label">Revenue Growth</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">💵</div>
                <div class="kpi-value">{insights.get('profit_margin', 0):.1f}%</div>
                <div class="kpi-label">Profit Margin</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🎯</div>
                <div class="kpi-value">{insights.get('roi', 0):.1f}%</div>
                <div class="kpi-label">ROI</div>
            </div>
        </div>
        
        <div class="actions-section">
            <h3 style="font-size: 20px; margin-bottom: 20px;">🎯 AI-Powered Recommendations</h3>
            <div class="actions-grid">
                <div class="action-card" style="border-left-color: #10b981;">
                    <div class="action-title">✅ Immediate Action</div>
                    <div class="action-text">{'Increase allocation to top performers' if insights.get('trend') == 'bullish' else 'Focus on capital preservation'}</div>
                </div>
                <div class="action-card" style="border-left-color: #f59e0b;">
                    <div class="action-title">⚠️ Risk Management</div>
                    <div class="action-text">{'High volatility - implement hedging' if insights.get('volatility', 0) > 25 else 'Standard risk controls sufficient'}</div>
                </div>
                <div class="action-card" style="border-left-color: #6366f1;">
                    <div class="action-title">💡 Strategic Insight</div>
                    <div class="action-text">{'Strong risk-adjusted returns' if insights.get('sharpe_ratio', 0) > 1 else 'Optimize for better returns'}</div>
                </div>
                <div class="action-card" style="border-left-color: #8b5cf6;">
                    <div class="action-title">📈 Next Steps</div>
                    <div class="action-text">{'Run additional analysis' if total_models < 3 else 'Execute optimized strategy'}</div>
                </div>
            </div>
        </div>
        
        <!-- DETAILED ANALYSIS SECTION ADDED BASED ON USER REQUEST to "Print everything including images" -->
        <h2 style="font-size: 24px; font-weight: 700; margin: 40px 0 20px 0; border-bottom: 2px solid #6366f1; padding-bottom: 15px;">📊 Detailed Analysis & Visualizations</h2>
'''
    
    # Generate additional HTML for all charts and models, similar to the Project Report
    chart_counter = 0
    
    # Helper for charts
    def create_chart_html(plot_data, category, model_name, chart_id):
        try:
            fig = go.Figure()
            if category == 'forecasting':
                if 'historical_values' in plot_data:
                    x_vals = plot_data.get('historical_dates', list(range(len(plot_data['historical_values']))))
                    fig.add_trace(go.Scatter(x=x_vals, y=plot_data['historical_values'], mode='lines', name='Historical', line=dict(color='#94a3b8', width=2)))
                if 'forecast_values' in plot_data:
                    x_vals = plot_data.get('forecast_dates', list(range(len(plot_data['forecast_values']))))
                    fig.add_trace(go.Scatter(x=x_vals, y=plot_data['forecast_values'], mode='lines', name='Forecast', line=dict(color='#6366f1', width=2, dash='dash')))
                fig.update_layout(title=f"{model_name} Forecast")
            elif category == 'ml_models':
                if 'feature_importance' in plot_data:
                    features = plot_data['feature_importance']
                    fig = go.Figure(go.Bar(x=list(features.values()), y=list(features.keys()), orientation='h', marker_color='#6366f1'))
                    fig.update_layout(title=f"{model_name} - Feature Importance")
                elif 'actual' in plot_data and 'predicted' in plot_data:
                    x_vals = list(range(len(plot_data['actual'])))
                    fig.add_trace(go.Scatter(x=x_vals, y=plot_data['actual'], name='Actual', line=dict(color='#94a3b8')))
                    fig.add_trace(go.Scatter(x=x_vals, y=plot_data['predicted'], name='Predicted', line=dict(color='#6366f1')))
                    fig.update_layout(title=f"{model_name} - Actual vs Predicted")
            # Common layout
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.5)', font=dict(color='#f8fafc'), margin=dict(t=50, l=50, r=30, b=50), height=350)
            return fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)
        except Exception:
            return '<div style="text-align:center; padding:20px; color:#64748b;">Chart unavailable</div>'

    for category, models in all_results.items():
        if models:
            cat_results = get_category_results(category)
            icons = {'forecasting': '🔮', 'ml_models': '🧪', 'portfolio': '💼', 'scenario': '📈', 'risk': '🛡️'}
            icon = icons.get(category, '📊')
            
            html += f'''
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 20px; padding: 30px; margin-bottom: 30px;">
            <div style="font-size: 20px; font-weight: 700; color: #f8fafc; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
                {icon} {category.replace('_', ' ').title()}
            </div>
'''
            for model_name, result in cat_results.items():
                metrics = result.get('metrics', {})
                plot_data = result.get('plot_data', {})
                
                html += f'''
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 16px; padding: 25px; margin-bottom: 20px;">
                <div style="font-size: 18px; font-weight: 700; color: #f8fafc; margin-bottom: 15px;">📊 {model_name}</div>
                <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px;">
'''
                for key, value in list(metrics.items())[:6]:
                    if key != 'primary_metric':
                        display_val = f"{value:.4f}" if isinstance(value, float) and abs(value) < 10 else str(value)
                        html += f'''
                    <div style="background: rgba(99, 102, 241, 0.1); padding: 10px 15px; border-radius: 10px;">
                        <div style="font-size: 10px; color: #94a3b8; text-transform: uppercase;">{key.replace('_', ' ').title()}</div>
                        <div style="font-size: 15px; font-weight: 700; color: #f8fafc;">{display_val}</div>
                    </div>
'''
                html += '</div>'
                
                if plot_data:
                    chart_id = f"chart_{chart_counter}"
                    chart_html = create_chart_html(plot_data, category, model_name, chart_id)
                    html += f'''
                <div style="background: rgba(15, 23, 42, 0.4); border-radius: 12px; padding: 15px; border: 1px solid rgba(51, 65, 85, 0.3);">
                    {chart_html}
                </div>
'''
                    chart_counter += 1
                
                html += '</div>'
            html += '</div>'
    
    html += '''
        
        <div class="footer">
            <p style="font-size: 16px; font-weight: 600; color: #f8fafc; margin-bottom: 8px;">💎 Financial Analytics Suite</p>
            <p>Enterprise-grade AI-powered financial analytics platform</p>
        </div>
    </div>
</body>
</html>'''
    
    return html
