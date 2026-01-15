"""
Financial Analytics Suite - Overview Page
General data overview with in-depth descriptive statistics
This is DIFFERENT from Dashboard - Overview shows DATA stats, Dashboard shows MODEL results
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict

# Import data manager
from pages_new.data_manager import (
    get_working_data, get_working_data_info, calculate_portfolio_metrics,
    get_time_series_for_chart, get_available_tickers, has_data, get_sector_allocation
)

# Import theme utilities
from pages_new.theme_utils import get_theme_colors, inject_premium_styles


def render_no_data_message(c: Dict):
    """Render a message when no data is available"""
    st.markdown(f'''
    <div style="background: {c['glass_bg']}; border: 2px dashed {c['border']}; border-radius: 20px; padding: 60px 40px; text-align: center; margin: 20px 0;">
        <div style="font-size: 64px; margin-bottom: 16px;">📊</div>
        <h2 style="color: {c['text']}; font-size: 24px; margin-bottom: 12px;">No Data Loaded</h2>
        <p style="color: {c['text_secondary']}; font-size: 14px; max-width: 400px; margin: 0 auto 24px;">
            Upload your financial data to see portfolio metrics, charts, and analysis.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📥 Go to Data Sources", type="primary", use_container_width=True):
            st.session_state.current_page = 'data_sources'
            st.rerun()


def render_metric_card(title: str, value: str, delta: str, delta_positive: bool, c: Dict):
    """Render a metric card"""
    delta_color = c['success'] if delta_positive else c['error']
    delta_icon = '↑' if delta_positive else '↓'
    
    card_html = f'<div style="background: {c["glass_bg"]}; backdrop-filter: blur(16px); border: 1px solid {c["glass_border"]}; border-radius: 16px; padding: 20px; text-align: center;"><div style="font-size: 11px; color: {c["text_muted"]}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">{title}</div><div style="font-size: 28px; font-weight: 700; color: {c["text"]}; margin-bottom: 8px;">{value}</div><div style="font-size: 12px; color: {delta_color}; font-weight: 600;">{delta_icon} {delta}</div></div>'
    st.markdown(card_html, unsafe_allow_html=True)


def calculate_descriptive_stats(df: pd.DataFrame) -> Dict:
    """Calculate comprehensive descriptive statistics for the data"""
    stats = {
        'basic': {},
        'numeric': {},
        'categorical': {},
        'missing': {},
        'correlations': None
    }
    
    # Basic info
    stats['basic'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'duplicates': df.duplicated().sum()
    }
    
    # Numeric column stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats['numeric'][col] = {
                    'count': len(col_data),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'q1': col_data.quantile(0.25),
                    'median': col_data.median(),
                    'q3': col_data.quantile(0.75),
                    'max': col_data.max(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                }
    
    # Categorical column stats
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        for col in cat_cols[:5]:  # Limit to first 5 categorical columns
            stats['categorical'][col] = {
                'unique': df[col].nunique(),
                'top': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'N/A',
                'top_freq': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            }
    
    # Missing data analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    stats['missing'] = {
        'total_missing': missing.sum(),
        'columns_with_missing': (missing > 0).sum(),
        'missing_pct': missing_pct.to_dict()
    }
    
    # Correlation matrix for numeric columns
    if len(numeric_cols) >= 2:
        stats['correlations'] = df[numeric_cols[:10]].corr()
    
    return stats


def render_descriptive_stats_section(df: pd.DataFrame, c: Dict):
    """Render the descriptive statistics section"""
    st.markdown("## 📊 Descriptive Statistics")
    st.caption("Comprehensive statistical analysis of your data")
    
    stats = calculate_descriptive_stats(df)
    
    # Basic info cards
    st.markdown("### 📋 Data Summary")
    basic_cols = st.columns(4)
    
    with basic_cols[0]:
        st.markdown(f'''
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 11px; color: {c['text_muted']}; text-transform: uppercase;">Total Rows</div>
            <div style="font-size: 28px; font-weight: 700; color: {c['primary']};">{stats['basic']['rows']:,}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with basic_cols[1]:
        st.markdown(f'''
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 11px; color: {c['text_muted']}; text-transform: uppercase;">Total Columns</div>
            <div style="font-size: 28px; font-weight: 700; color: {c['accent']};">{stats['basic']['columns']}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with basic_cols[2]:
        st.markdown(f'''
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 11px; color: {c['text_muted']}; text-transform: uppercase;">Memory Usage</div>
            <div style="font-size: 28px; font-weight: 700; color: {c['secondary']};">{stats['basic']['memory_mb']:.2f} MB</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with basic_cols[3]:
        dup_color = c['success'] if stats['basic']['duplicates'] == 0 else c['warning']
        st.markdown(f'''
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 12px; padding: 20px; text-align: center;">
            <div style="font-size: 11px; color: {c['text_muted']}; text-transform: uppercase;">Duplicates</div>
            <div style="font-size: 28px; font-weight: 700; color: {dup_color};">{stats['basic']['duplicates']}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # Numeric variables statistics
    if stats['numeric']:
        st.markdown("### 📈 Numeric Variables")
        
        # Create a nice table
        table_data = []
        for col, col_stats in stats['numeric'].items():
            table_data.append({
                'Column': col,
                'Count': f"{col_stats['count']:,}",
                'Mean': f"{col_stats['mean']:.4f}",
                'Std': f"{col_stats['std']:.4f}",
                'Min': f"{col_stats['min']:.4f}",
                'Q1': f"{col_stats['q1']:.4f}",
                'Median': f"{col_stats['median']:.4f}",
                'Q3': f"{col_stats['q3']:.4f}",
                'Max': f"{col_stats['max']:.4f}",
                'Skewness': f"{col_stats['skewness']:.2f}",
                'Kurtosis': f"{col_stats['kurtosis']:.2f}"
            })
        
        stats_df = pd.DataFrame(table_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Distribution plots
        st.markdown("#### 📊 Distribution Plots")
        
        numeric_cols = list(stats['numeric'].keys())
        selected_col = st.selectbox("Select column for distribution", numeric_cols, key="dist_col")
        
        if selected_col:
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_col, nbins=30, 
                                   title=f"Distribution of {selected_col}")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color=c['text']),
                    xaxis=dict(showgrid=True, gridcolor=c['border']),
                    yaxis=dict(showgrid=True, gridcolor=c['border']),
                    height=300
                )
                fig.update_traces(marker_color=c['primary'])
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                # Box plot
                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color=c['text']),
                    height=300
                )
                fig.update_traces(marker_color=c['primary'], line_color=c['primary'])
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # Categorical variables
    if stats['categorical']:
        st.markdown("### 🏷️ Categorical Variables")
        
        cat_data = []
        for col, col_stats in stats['categorical'].items():
            cat_data.append({
                'Column': col,
                'Unique Values': col_stats['unique'],
                'Most Common': str(col_stats['top']),
                'Frequency': col_stats['top_freq']
            })
        
        cat_df = pd.DataFrame(cat_data)
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # Missing data analysis
    st.markdown("### ⚠️ Missing Data Analysis")
    
    missing_cols = st.columns([1, 2])
    
    with missing_cols[0]:
        total_cells = stats['basic']['rows'] * stats['basic']['columns']
        missing_pct = (stats['missing']['total_missing'] / total_cells * 100) if total_cells > 0 else 0
        
        st.markdown(f'''
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 12px; padding: 20px;">
            <div style="margin-bottom: 12px;">
                <span style="font-size: 14px; font-weight: 600; color: {c['text']};">Missing Values Summary</span>
            </div>
            <div style="margin-bottom: 8px;">
                <span style="color: {c['text_muted']};">Total Missing:</span>
                <span style="font-weight: 600; color: {c['warning']}; margin-left: 8px;">{stats['missing']['total_missing']:,}</span>
            </div>
            <div style="margin-bottom: 8px;">
                <span style="color: {c['text_muted']};">Columns with Missing:</span>
                <span style="font-weight: 600; color: {c['text']}; margin-left: 8px;">{stats['missing']['columns_with_missing']}</span>
            </div>
            <div>
                <span style="color: {c['text_muted']};">Overall Missing %:</span>
                <span style="font-weight: 600; color: {c['error'] if missing_pct > 5 else c['success']}; margin-left: 8px;">{missing_pct:.2f}%</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    with missing_cols[1]:
        # Bar chart of missing by column
        missing_by_col = {k: v for k, v in stats['missing']['missing_pct'].items() if v > 0}
        
        if missing_by_col:
            fig = go.Figure(go.Bar(
                x=list(missing_by_col.keys()),
                y=list(missing_by_col.values()),
                marker_color=c['warning']
            ))
            fig.update_layout(
                title="Missing Values by Column (%)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color=c['text']),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor=c['border']),
                height=250
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.success("✅ No missing values in the dataset!")
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # Correlation matrix
    if stats['correlations'] is not None and len(stats['correlations']) >= 2:
        st.markdown("### 🔗 Correlation Matrix")
        
        corr_matrix = stats['correlations']
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=10),
            hovertemplate='%{x} ↔ %{y}<br>Correlation: %{z:.3f}<extra></extra>',
            colorbar=dict(title='Correlation', thickness=15)
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color=c['text']),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_performance_chart(c: Dict):
    """Render the main performance chart using real data"""
    df = get_working_data()
    
    if df is None or df.empty:
        st.info("Upload data to see the performance chart")
        return
    
    # Get available tickers
    tickers = get_available_tickers()
    
    fig = go.Figure()
    
    # Convert primary hex to rgba
    primary_hex = c['primary'].lstrip('#')
    primary_rgb = tuple(int(primary_hex[i:i+2], 16) for i in (0, 2, 4))
    fill_rgba = f'rgba({primary_rgb[0]}, {primary_rgb[1]}, {primary_rgb[2]}, 0.1)'
    
    if tickers:
        # Plot first 5 tickers
        colors = [c['primary'], c['accent'], c['secondary'], c['success'], c['warning']]
        
        for i, ticker in enumerate(tickers[:5]):
            dates, values = get_time_series_for_chart(ticker, 'close')
            if dates and values:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines',
                    name=ticker,
                    line=dict(color=colors[i % len(colors)], width=2),
                ))
    else:
        # Try to plot any numeric column
        dates, values = get_time_series_for_chart(None, 'close')
        if dates and values:
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name='Price',
                fill='tozeroy',
                fillcolor=fill_rgba,
                line=dict(color=c['primary'], width=2),
            ))
    
    fig.update_layout(
        margin=dict(t=20, l=10, r=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=c['text']),
        xaxis=dict(showgrid=True, gridcolor=c['border'], tickfont=dict(size=10, color=c['text_muted'])),
        yaxis=dict(showgrid=True, gridcolor=c['border'], tickfont=dict(size=10, color=c['text_muted']), title='Price'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
        hovermode='x unified',
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_allocation_chart(c: Dict):
    """Render allocation pie chart"""
    allocation = get_sector_allocation()
    
    if not allocation:
        st.info("No sector data available")
        return
    
    labels = list(allocation.keys())
    values = list(allocation.values())
    
    colors = [c['primary'], c['accent'], c['secondary'], c['success'], c['warning'], c['error']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='percent',
        textposition='outside',
        textfont=dict(size=10, color=c['text'])
    )])
    
    fig.update_layout(
        margin=dict(t=20, l=10, r=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=c['text']),
        showlegend=True,
        legend=dict(font=dict(size=10, color=c['text_secondary'])),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_top_performers(c: Dict):
    """Render top performing assets"""
    df = get_working_data()
    
    if df is None or df.empty:
        st.info("Upload data to see top performers")
        return
    
    tickers = get_available_tickers()
    
    if not tickers:
        st.info("No ticker data available")
        return
    
    # Calculate returns for each ticker
    performances = []
    
    ticker_col = None
    for col in ['ticker', 'symbol', 'Ticker', 'Symbol']:
        if col in df.columns:
            ticker_col = col
            break
    
    price_col = None
    for col in ['close', 'Close', 'price', 'Price']:
        if col in df.columns:
            price_col = col
            break
    
    if ticker_col and price_col:
        for ticker in tickers[:10]:
            ticker_data = df[df[ticker_col] == ticker]
            if len(ticker_data) > 1:
                first_price = ticker_data[price_col].iloc[0]
                last_price = ticker_data[price_col].iloc[-1]
                if first_price > 0:
                    pct_change = ((last_price - first_price) / first_price) * 100
                    performances.append({
                        'ticker': ticker,
                        'price': last_price,
                        'change': pct_change
                    })
    
    if not performances:
        st.info("Unable to calculate performance")
        return
    
    # Sort by performance
    performances.sort(key=lambda x: x['change'], reverse=True)
    
    # Display top 5
    for perf in performances[:5]:
        change_color = c['success'] if perf['change'] >= 0 else c['error']
        change_icon = '↑' if perf['change'] >= 0 else '↓'
        
        row_html = f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: {c["bg_card"]}; border: 1px solid {c["border"]}; border-radius: 10px; margin-bottom: 8px;"><div style="font-weight: 600; color: {c["text"]};">{perf["ticker"]}</div><div style="color: {c["text_muted"]};">${perf["price"]:.2f}</div><div style="color: {change_color}; font-weight: 600;">{change_icon} {abs(perf["change"]):.1f}%</div></div>'
        st.markdown(row_html, unsafe_allow_html=True)


def render():
    """Render the Overview page"""
    c = get_theme_colors()
    inject_premium_styles()
    
    # Header
    st.title("📈 Data Overview")
    st.markdown(f"<p style='color: {c['text_secondary']};'>General overview and descriptive statistics of your data</p>", unsafe_allow_html=True)
    
    # Check if we have data
    if not has_data():
        render_no_data_message(c)
        return
    
    # Get data info and metrics
    data_info = get_working_data_info()
    df = get_working_data()
    metrics = calculate_portfolio_metrics(df)
    
    # Data info bar
    st.markdown(f'<div style="background: {c["primary"]}15; border: 1px solid {c["primary"]}30; border-radius: 12px; padding: 12px 20px; margin-bottom: 20px; display: flex; align-items: center; gap: 20px;"><span style="font-size: 16px;">📊</span><div><span style="color: {c["text"]}; font-weight: 600;">Working Data:</span> <span style="color: {c["text_secondary"]};">{data_info["name"]}</span></div><div style="color: {c["text_muted"]};">|</div><div><span style="color: {c["text_muted"]};">{data_info["rows"]:,} rows</span></div><div style="color: {c["text_muted"]};">|</div><div><span style="color: {c["text_muted"]};">{len(data_info["tickers"])} assets</span></div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "📈 Descriptive Statistics", "📋 Data Preview"])
    
    with tab1:
        # Summary tab - quick metrics
        st.markdown("### 💹 Portfolio Metrics")
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            if metrics['total_value'] > 0:
                value_str = f"${metrics['total_value']:,.0f}" if metrics['total_value'] > 1000 else f"${metrics['total_value']:.2f}"
            else:
                value_str = "-"
            render_metric_card("Portfolio Value", value_str, f"{metrics['total_return']:.1f}%", metrics['total_return'] >= 0, c)
        
        with metric_cols[1]:
            render_metric_card("Daily Return", f"{metrics['daily_return']:.2f}%", "avg", metrics['daily_return'] >= 0, c)
        
        with metric_cols[2]:
            render_metric_card("Volatility", f"{metrics['volatility']:.1f}%", "annualized", True, c)
        
        with metric_cols[3]:
            render_metric_card("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}", "risk-adj", metrics['sharpe_ratio'] >= 0, c)
        
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        # Charts row
        chart_cols = st.columns([2, 1])
        
        with chart_cols[0]:
            st.markdown("#### 📈 Price Performance")
            card_html = f'<div style="background: {c["glass_bg"]}; border: 1px solid {c["glass_border"]}; border-radius: 16px; padding: 20px;">'
            st.markdown(card_html, unsafe_allow_html=True)
            render_performance_chart(c)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with chart_cols[1]:
            st.markdown("#### 🥧 Sector Allocation")
            card_html = f'<div style="background: {c["glass_bg"]}; border: 1px solid {c["glass_border"]}; border-radius: 16px; padding: 20px;">'
            st.markdown(card_html, unsafe_allow_html=True)
            render_allocation_chart(c)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        # Bottom row
        bottom_cols = st.columns([1, 1])
        
        with bottom_cols[0]:
            st.markdown("#### 🏆 Top Performers")
            render_top_performers(c)
        
        with bottom_cols[1]:
            st.markdown("#### 📋 Data Summary")
            
            summary_html = f'<div style="background: {c["glass_bg"]}; border: 1px solid {c["glass_border"]}; border-radius: 16px; padding: 20px;">'
            st.markdown(summary_html, unsafe_allow_html=True)
            
            st.markdown(f"**Dataset:** {data_info['name']}")
            st.markdown(f"**Records:** {data_info['rows']:,}")
            st.markdown(f"**Columns:** {data_info['columns']}")
            st.markdown(f"**Assets:** {len(data_info['tickers'])}")
            
            if data_info['date_range']:
                st.markdown(f"**Date Range:** {data_info['date_range'][0].strftime('%Y-%m-%d')} to {data_info['date_range'][1].strftime('%Y-%m-%d')}")
            
            if data_info['tickers']:
                st.markdown(f"**Tickers:** {', '.join(data_info['tickers'][:10])}{'...' if len(data_info['tickers']) > 10 else ''}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        # Descriptive Statistics tab
        render_descriptive_stats_section(df, c)
    
    with tab3:
        # Data Preview tab
        st.markdown("### 📋 Data Preview")
        
        # Column info
        st.markdown("#### Column Information")
        col_info = []
        for col in df.columns:
            col_info.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null': f"{df[col].notna().sum():,}",
                'Null': f"{df[col].isna().sum():,}",
                'Unique': f"{df[col].nunique():,}"
            })
        
        col_df = pd.DataFrame(col_info)
        st.dataframe(col_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Data preview options
        preview_cols = st.columns([1, 1, 2])
        
        with preview_cols[0]:
            preview_type = st.selectbox("View", ["First Rows", "Last Rows", "Random Sample"])
        
        with preview_cols[1]:
            num_rows = st.number_input("Number of rows", min_value=5, max_value=100, value=20)
        
        st.markdown("#### Data")
        
        if preview_type == "First Rows":
            st.dataframe(df.head(int(num_rows)), use_container_width=True, hide_index=True)
        elif preview_type == "Last Rows":
            st.dataframe(df.tail(int(num_rows)), use_container_width=True, hide_index=True)
        else:
            st.dataframe(df.sample(min(int(num_rows), len(df))), use_container_width=True, hide_index=True)
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # HTML Export Section
    st.markdown("---")
    st.markdown("#### 📥 Export Overview Report")
    
    # Generate HTML report for overview
    def generate_overview_html():
        stats = calculate_descriptive_stats(df)
        
        html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Overview Report - Financial Analytics Suite</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            color: #f8fafc;
            min-height: 100vh;
            padding: 40px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            text-align: center;
            padding: 50px 40px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.1));
            border-radius: 24px;
            margin-bottom: 40px;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }}
        .header h1 {{
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }}
        .header p {{ color: #94a3b8; font-size: 16px; }}
        .section {{
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(51, 65, 85, 0.5);
            border-radius: 20px;
            padding: 28px;
            margin-bottom: 28px;
        }}
        .section-title {{
            font-size: 22px;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background: linear-gradient(145deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.05));
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 14px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
        .kpi-value {{ font-size: 28px; font-weight: 800; color: #6366f1; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid rgba(51, 65, 85, 0.5); }}
        th {{ background: rgba(99, 102, 241, 0.1); font-weight: 600; font-size: 11px; text-transform: uppercase; color: #a5b4fc; }}
        .footer {{ text-align: center; padding: 30px; color: #64748b; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Data Overview Report</h1>
            <p>Comprehensive data statistics and analysis</p>
            <p style="margin-top: 12px; font-size: 14px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <div class="section">
            <div class="section-title">📋 Data Summary</div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Rows</div>
                    <div class="kpi-value">{stats['basic']['rows']:,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total Columns</div>
                    <div class="kpi-value">{stats['basic']['columns']}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Memory Usage</div>
                    <div class="kpi-value">{stats['basic']['memory_mb']:.2f} MB</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Duplicates</div>
                    <div class="kpi-value">{stats['basic']['duplicates']}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📊 Numeric Variables Statistics</div>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Count</th>
                        <th>Mean</th>
                        <th>Std</th>
                        <th>Min</th>
                        <th>Median</th>
                        <th>Max</th>
                    </tr>
                </thead>
                <tbody>
'''
        for col, col_stats in stats['numeric'].items():
            html += f'''
                    <tr>
                        <td>{col}</td>
                        <td>{col_stats['count']:,}</td>
                        <td>{col_stats['mean']:.4f}</td>
                        <td>{col_stats['std']:.4f}</td>
                        <td>{col_stats['min']:.4f}</td>
                        <td>{col_stats['median']:.4f}</td>
                        <td>{col_stats['max']:.4f}</td>
                    </tr>
'''
        
        html += '''
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">📉 Missing Data Analysis</div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Missing Values</div>
'''
        html += f'<div class="kpi-value">{stats["missing"]["total_missing"]:,}</div>'
        html += '''
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Columns With Missing</div>
'''
        html += f'<div class="kpi-value">{stats["missing"]["columns_with_missing"]}</div>'
        html += '''
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by <strong>Financial Analytics Suite</strong></p>
        </div>
    </div>
</body>
</html>
'''
        return html
    
    export_cols = st.columns(2)
    with export_cols[0]:
        html_report = generate_overview_html()
        st.download_button(
            "📥 Download Overview Report (HTML)",
            data=html_report,
            file_name=f"data_overview_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            key="dl_overview_html",
            use_container_width=True
        )
    
    with export_cols[1]:
        # CSV export of stats
        stats = calculate_descriptive_stats(df)
        if stats['numeric']:
            table_data = []
            for col, col_stats in stats['numeric'].items():
                table_data.append({
                    'Column': col,
                    'Count': col_stats['count'],
                    'Mean': col_stats['mean'],
                    'Std': col_stats['std'],
                    'Min': col_stats['min'],
                    'Max': col_stats['max']
                })
            stats_df = pd.DataFrame(table_data)
            csv_data = stats_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Statistics (CSV)",
                data=csv_data,
                file_name="data_statistics.csv",
                mime="text/csv",
                key="dl_overview_csv",
                use_container_width=True
            )
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("#### 🚀 Quick Actions")
    action_cols = st.columns(4)
    
    with action_cols[0]:
        if st.button("🔮 Run Forecast", use_container_width=True):
            st.session_state.current_page = 'forecasting'
            st.rerun()
    
    with action_cols[1]:
        if st.button("🧪 ML Analysis", use_container_width=True):
            st.session_state.current_page = 'ml_lab'
            st.rerun()
    
    with action_cols[2]:
        if st.button("📊 View Dashboard", use_container_width=True):
            st.session_state.current_page = 'dashboard_builder'
            st.rerun()
    
    with action_cols[3]:
        if st.button("📄 Generate Report", use_container_width=True):
            st.session_state.current_page = 'reports'
            st.rerun()


if __name__ == "__main__":
    render()
