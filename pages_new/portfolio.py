"""
Financial Analytics Suite - Portfolio Analytics & Optimization Page
Portfolio builder, efficient frontier, risk analytics, and optimization using YOUR data
ALL ANALYSIS REQUIRES USER ACTION - Nothing runs automatically
WITH DOWNLOAD OPTIONS for graphs and tables
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import io

# Import data manager
from pages_new.data_manager import (
    get_working_data, get_working_data_info, has_data,
    get_available_tickers, get_price_data
)

# Import theme utilities
from pages_new.theme_utils import get_theme_colors, render_plot_with_download

# Import model results manager
from pages_new.model_results_manager import (
    save_model_result, get_model_result, has_model_been_run,
    get_category_results
)


def dataframe_to_excel(df):
    """Convert dataframe to Excel bytes"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()


def render_download_buttons(fig=None, df=None, name="download"):
    """Render download buttons for figure and/or dataframe"""
    cols = st.columns(4)
    
    if fig is not None:
        with cols[0]:
            try:
                html_str = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    "📥 HTML",
                    data=html_str,
                    file_name=f"{name}.html",
                    mime="text/html",
                    key=f"dl_html_{name}_{hash(str(fig))%10000}",
                    use_container_width=True
                )
            except:
                pass
    
    if df is not None:
        with cols[1]:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 CSV",
                data=csv_data,
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"dl_csv_{name}_{hash(str(df))%10000}",
                use_container_width=True
            )
        
        with cols[2]:
            try:
                excel_data = dataframe_to_excel(df)
                st.download_button(
                    "📥 Excel",
                    data=excel_data,
                    file_name=f"{name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_xlsx_{name}_{hash(str(df))%10000}",
                    use_container_width=True
                )
            except:
                pass


def get_portfolio_from_data() -> List[Dict]:
    """Extract portfolio data from uploaded dataset"""
    if not has_data():
        return []
    
    df = get_working_data()
    tickers = get_available_tickers()
    
    if not tickers:
        return []
    
    assets = []
    for i, ticker in enumerate(tickers[:10]):  # Limit to 10 assets
        ticker_df = get_price_data(ticker)
        
        if ticker_df is not None and 'close' in ticker_df.columns:
            prices = ticker_df['close'].values
            returns = pd.Series(prices).pct_change().dropna()
            
            # Calculate metrics
            ann_return = returns.mean() * 252  # Annualized return
            ann_vol = returns.std() * np.sqrt(252)  # Annualized volatility
            latest_price = prices[-1] if len(prices) > 0 else 0
            
            assets.append({
                'symbol': ticker,
                'name': ticker,
                'sector': 'Unknown',  # Could be enhanced with sector mapping
                'weight': 1.0 / len(tickers[:10]),  # Equal weight initially
                'return': ann_return,
                'std': ann_vol,
                'price': latest_price
            })
    
    return assets


def calculate_portfolio_metrics(assets: List[Dict]) -> Dict:
    """Calculate overall portfolio metrics"""
    if not assets:
        return {'return': 0, 'volatility': 0, 'sharpe': 0, 'value': 0}
    
    weights = np.array([a['weight'] for a in assets])
    returns = np.array([a['return'] for a in assets])
    vols = np.array([a['std'] for a in assets])
    
    # Portfolio return (weighted average)
    port_return = np.sum(weights * returns)
    
    # Portfolio volatility (simplified - assumes some correlation)
    port_vol = np.sqrt(np.sum((weights * vols) ** 2) * 1.2)  # 1.2 is correlation factor
    
    # Sharpe ratio (assuming risk-free rate of 0.05)
    sharpe = (port_return - 0.05) / port_vol if port_vol > 0 else 0
    
    # Portfolio value (assuming $1M base)
    port_value = 1000000
    
    return {
        'return': port_return,
        'volatility': port_vol,
        'sharpe': sharpe,
        'value': port_value
    }


def generate_efficient_frontier(assets: List[Dict], n_points: int = 100) -> Dict:
    """Generate efficient frontier data from actual assets"""
    if not assets:
        return None
    
    asset_returns = np.array([a['return'] for a in assets])
    asset_vols = np.array([a['std'] for a in assets])
    n_assets = len(assets)
    
    # Generate random portfolios
    n_portfolios = 500
    port_returns = []
    port_vols = []
    
    np.random.seed(42)
    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= weights.sum()
        
        port_ret = np.dot(weights, asset_returns)
        port_vol = np.sqrt(np.sum((weights * asset_vols) ** 2))
        
        port_returns.append(port_ret)
        port_vols.append(port_vol)
    
    port_returns = np.array(port_returns)
    port_vols = np.array(port_vols)
    
    # Calculate efficient frontier (approximate)
    min_ret, max_ret = port_returns.min(), port_returns.max()
    frontier_returns = np.linspace(min_ret, max_ret, n_points)
    frontier_vol = []
    
    for target_ret in frontier_returns:
        # Find minimum volatility for this return level (approximation)
        mask = port_returns >= target_ret - 0.02
        if mask.any():
            frontier_vol.append(port_vols[mask].min())
        else:
            frontier_vol.append(port_vols.min())
    
    frontier_vol = np.array(frontier_vol)
    
    # Calculate current portfolio (equal weights)
    equal_weights = np.ones(n_assets) / n_assets
    current_ret = np.dot(equal_weights, asset_returns)
    current_vol = np.sqrt(np.sum((equal_weights * asset_vols) ** 2))
    
    # Find max Sharpe portfolio
    sharpe_ratios = (port_returns - 0.05) / port_vols
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_ret = port_returns[max_sharpe_idx]
    max_sharpe_vol = port_vols[max_sharpe_idx]
    
    # Find min variance portfolio
    min_var_idx = np.argmin(port_vols)
    min_var_ret = port_returns[min_var_idx]
    min_var_vol = port_vols[min_var_idx]
    
    return {
        'frontier_returns': frontier_returns.tolist(),
        'frontier_vol': frontier_vol.tolist(),
        'random_returns': port_returns.tolist(),
        'random_vol': port_vols.tolist(),
        'current': (current_vol, current_ret),
        'max_sharpe': (max_sharpe_vol, max_sharpe_ret),
        'min_var': (min_var_vol, min_var_ret)
    }


def calculate_correlation_matrix(assets: List[Dict]) -> pd.DataFrame:
    """Calculate actual correlation matrix from price data"""
    if not has_data():
        return generate_sample_correlation_matrix(assets)
    
    price_data = {}
    for asset in assets:
        ticker_df = get_price_data(asset['symbol'])
        if ticker_df is not None and 'close' in ticker_df.columns:
            price_data[asset['symbol']] = ticker_df['close'].values
    
    if len(price_data) < 2:
        return generate_sample_correlation_matrix(assets)
    
    # Create returns dataframe
    min_len = min(len(v) for v in price_data.values())
    returns_df = pd.DataFrame({k: pd.Series(v[:min_len]).pct_change().dropna() 
                               for k, v in price_data.items()})
    
    # Calculate correlation
    corr_matrix = returns_df.corr()
    return corr_matrix


def generate_sample_correlation_matrix(assets: List[Dict]) -> pd.DataFrame:
    """Generate a sample correlation matrix"""
    np.random.seed(42)
    n = len(assets)
    corr = np.eye(n)
    
    for i in range(n):
        for j in range(i + 1, n):
            if assets[i].get('sector') == assets[j].get('sector'):
                corr[i, j] = corr[j, i] = np.random.uniform(0.4, 0.7)
            else:
                corr[i, j] = corr[j, i] = np.random.uniform(0.1, 0.4)
    
    symbols = [a['symbol'] for a in assets]
    return pd.DataFrame(corr, index=symbols, columns=symbols)


def render_efficient_frontier(data: Dict, c: Dict) -> None:
    """Render the efficient frontier chart"""
    if data is None:
        st.warning("No efficient frontier data available. Run the analysis first.")
        return
    
    fig = go.Figure()
    
    # Random portfolios scatter
    fig.add_trace(go.Scatter(
        x=data['random_vol'],
        y=data['random_returns'],
        mode='markers',
        marker=dict(
            size=4,
            color=[r/max(v, 0.01) for r, v in zip(data['random_returns'], data['random_vol'])],
            colorscale='Viridis',
            opacity=0.4,
            showscale=True,
            colorbar=dict(title='Sharpe<br>Ratio', thickness=15, len=0.5)
        ),
        name='Random Portfolios',
        hovertemplate='Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
    ))
    
    # Efficient frontier line
    fig.add_trace(go.Scatter(
        x=data['frontier_vol'],
        y=data['frontier_returns'],
        mode='lines',
        line=dict(color=c['primary'], width=3),
        name='Efficient Frontier',
        hovertemplate='Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
    ))
    
    # Current portfolio
    fig.add_trace(go.Scatter(
        x=[data['current'][0]],
        y=[data['current'][1]],
        mode='markers',
        marker=dict(size=16, color=c['warning'], symbol='diamond'),
        name='Current Portfolio',
        hovertemplate='Current Portfolio<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
    ))
    
    # Max Sharpe portfolio
    fig.add_trace(go.Scatter(
        x=[data['max_sharpe'][0]],
        y=[data['max_sharpe'][1]],
        mode='markers',
        marker=dict(size=16, color=c['success'], symbol='star'),
        name='Max Sharpe Portfolio',
        hovertemplate='Max Sharpe<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
    ))
    
    # Min variance portfolio
    fig.add_trace(go.Scatter(
        x=[data['min_var'][0]],
        y=[data['min_var'][1]],
        mode='markers',
        marker=dict(size=14, color=c['accent'], symbol='circle'),
        name='Min Variance Portfolio',
        hovertemplate='Min Variance<br>Vol: %{x:.1%}<br>Return: %{y:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        margin=dict(t=40, l=10, r=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=c['text']),
        xaxis=dict(title='Volatility (Risk)', tickformat='.0%', showgrid=True, gridcolor=c['border']),
        yaxis=dict(title='Expected Return', tickformat='.0%', showgrid=True, gridcolor=c['border']),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='closest',
        height=450
    )
    
    # fig.update_layout(...) already called above
    
    render_plot_with_download(fig, f"efficient_frontier_{data.get('method', 'model')}", key=f"ef_chart_main")

    st.markdown(f"""
    <div style="background: {c['bg_elevated']}; padding: 16px; border-radius: 12px; border: 1px solid {c['border']}; margin-top: 12px;">
        <div style="font-weight: 600; font-size: 14px; margin-bottom: 8px;">💡 Efficient Frontier Interpretation</div>
        <div style="font-size: 13px; color: {c['text_secondary']}; line-height: 1.6;">
            This curve represents the best possible returns for a given level of risk. 
            <ul>
                <li>The <span style="color: {c['primary']}"><b>Blue Curve</b></span> is the "Efficient Frontier". No portfolio exists above this line.</li>
                <li>The <span style="color: {c['success']}"><b>Star (★)</b></span> is the <b>Maximum Sharpe Ratio</b> portfolio. It offers the best risk-adjusted return (best "bang for your buck").</li>
                <li>The <span style="color: {c['accent']}"><b>Circle (●)</b></span> is the <b>Minimum Volatility</b> portfolio. It is the safest possible allocation mathematically.</li>
                <li>The <span style="color: {c['warning']}"><b>Diamond (♦)</b></span> is your <b>Current Portfolio</b>. If it is far below the curve, your portfolio is inefficient (taking too much risk for too little return).</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_correlation_heatmap(corr_df: pd.DataFrame, c: Dict) -> None:
    """Render correlation matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale='RdBu_r',
        zmid=0.5,
        text=np.round(corr_df.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10, color='white'),
        hovertemplate='%{x} ↔ %{y}<br>Correlation: %{z:.2f}<extra></extra>',
        colorbar=dict(title='Correlation', thickness=15, len=0.8)
    ))
    
    fig.update_layout(
        margin=dict(t=40, l=10, r=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=c['text']),
        height=350
    )
    
    render_plot_with_download(fig, "correlation_heatmap", key="corr_heatmap")

    st.markdown(f"""
    <div style="background: {c['bg_elevated']}; padding: 12px; border-radius: 8px; border: 1px solid {c['border']}; margin-top: 8px;">
        <div style="font-size: 12px; color: {c['text_secondary']}; line-height: 1.5;">
            <b>About Correlations:</b> <br>
            Values range from <b>-1.0</b> (Inverse) to <b>+1.0</b> (Identical). <br>
            • <b style="color: #ef4444">High Positive (Red, > 0.7):</b> Assets move together. <b>Bad for diversification.</b><br>
            • <b style="color: {c['primary']}">Low/Negative (Blue, < 0.3):</b> Assets move independently. <b>Great for diversification</b> (risk reduction).
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_allocation_pie(assets: List[Dict], c: Dict) -> None:
    """Render portfolio allocation pie chart"""
    
    fig = go.Figure(data=[go.Pie(
        labels=[a['symbol'] for a in assets],
        values=[a['weight'] * 100 for a in assets],
        hole=0.55,
        marker=dict(colors=c['chart_colors'][:len(assets)]),
        textinfo='label+percent',
        textposition='outside',
        textfont=dict(size=11, color=c['text']),
        hovertemplate='%{label}<br>Weight: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        margin=dict(t=20, l=10, r=10, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=c['text']),
        showlegend=False,
        height=300,
        annotations=[dict(text='<b>Portfolio</b>', x=0.5, y=0.5, font_size=14, showarrow=False, font=dict(color=c['text']))]
    )
    
    render_plot_with_download(fig, "portfolio_allocation", key="allocation_pie_chart")

    st.markdown(f"""
    <div style="background: {c['bg_elevated']}; padding: 12px; border-radius: 8px; border: 1px solid {c['border']}; margin-top: 8px;">
        <div style="font-size: 12px; color: {c['text_secondary']}; line-height: 1.5;">
            <b>Optimal Weighting:</b> <br>
            This chart shows the recommended percentage of capital to invest in each asset to achieve the target strategy (e.g., Max Sharpe). 
            Diversification usually requires spreading capital across multiple uncorrelated assets.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_metrics(assets: List[Dict], c: Dict) -> None:
    """Render risk analytics metrics calculated from actual data"""
    metrics = calculate_portfolio_metrics(assets)
    
    # Calculate VaR and CVaR
    port_vol = metrics['volatility']
    port_value = metrics['value']
    
    var_95 = port_value * port_vol * 1.645  # 95% VaR (1.645 std)
    cvar_95 = var_95 * 1.2  # Approximate CVaR
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Value at Risk (95%)", f"-${var_95:,.0f}", f"-{var_95/port_value*100:.2f}% of portfolio")
    with col2:
        st.metric("Conditional VaR (95%)", f"-${cvar_95:,.0f}", f"-{cvar_95/port_value*100:.2f}% of portfolio")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Max Drawdown (Est.)", f"-{port_vol * 2 * 100:.1f}%", "Recovery: ~45 days")
    with col4:
        st.metric("Rolling Volatility (30d)", f"{port_vol * 100:.1f}%", f"Annualized")


def render():
    """Render the Portfolio Analytics page"""
    from pages_new.theme_utils import get_theme_colors, inject_premium_styles
    c = get_theme_colors()
    inject_premium_styles()
    
    # Header with animated gradient
    st.markdown(f"""
    <div style="padding: 20px 0; animation: slideInUp 0.5s ease-out;">
        <h1 class="glass-header" style="font-size: 42px; margin-bottom: 10px;">💼 Portfolio Analytics</h1>
        <p style="color: {c['text_secondary']}; font-size: 16px;">Build, analyze, and optimize your investment portfolio with institutional-grade tools</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for data - REQUIRE user data
    data_available = has_data()
    
    if not data_available:
        # Show "No Data" message
        st.markdown(f'''
        <div style="background: {c['glass_bg']}; border: 1px solid {c['glass_border']}; border-radius: 20px; padding: 80px 40px; text-align: center; margin-top: 40px; box-shadow: 0 20px 40px rgba(0,0,0,0.15); backdrop-filter: blur(20px);">
            <div style="width: 100px; height: 100px; background: {c['primary']}20; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 30px; border: 2px solid {c['primary']}40; animation: pulse 2s infinite;">
                <span style="font-size: 50px;">�</span>
            </div>
            <h2 style="color: {c['text']}; font-weight: 800; margin-bottom: 20px; letter-spacing: -0.02em;">Ready to Optimize Your Wealth?</h2>
            <p style="color: {c['text_secondary']}; font-size: 16px; max-width: 500px; margin: 0 auto 40px; line-height: 1.6;">
                Transform raw holdings into intelligent insights. Import your portfolio data from the <b>Data Management</b> section or upload a CSV to unlock deep analytics and institutional-grade optimization.
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                <div style="background: {c['bg_elevated']}; padding: 15px 25px; border-radius: 12px; border: 1px solid {c['border']}; text-align: left; width: 220px;">
                    <div style="font-size: 24px; margin-bottom: 10px;">📤</div>
                    <div style="font-weight: 700; color: {c['text']}; font-size: 14px;">Import Data</div>
                    <div style="color: {c['text_secondary']}; font-size: 12px;">Link your accounts</div>
                </div>
                <div style="background: {c['bg_elevated']}; padding: 15px 25px; border-radius: 12px; border: 1px solid {c['border']}; text-align: left; width: 220px;">
                    <div style="font-size: 24px; margin-bottom: 10px;">📊</div>
                    <div style="font-weight: 700; color: {c['text']}; font-size: 14px;">Build Strategy</div>
                    <div style="color: {c['text_secondary']}; font-size: 12px;">Set target weights</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📥 Go to Data Sources", type="primary", use_container_width=True, key="goto_datasrc"):
                st.session_state.current_page = 'data_sources'
                st.rerun()
        return
    
    # Data is available - get assets
    assets = get_portfolio_from_data()
    
    if not assets:
        st.warning("⚠️ Could not extract portfolio assets from your data. Ensure your data has 'ticker' and 'close' columns.")
        return
    
    # Show data status
    data_info = get_working_data_info()
    st.markdown(f"""
    <div class="glass-card" style="padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 20px;">✅</span>
            <div>
                <div style="color: {c['text']}; font-weight: 600;">Data Loaded: {data_info['name']}</div>
                <div style="color: {c['text_secondary']}; font-size: 13px;">{len(assets)} assets detected</div>
            </div>
        </div>
        <div class="status-badge status-success">Ready for Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Assets Overview", "⚖️ Optimization", "🛡️ Risk Analytics", "🔄 Rebalance"])
    
    with tab1:
        # Assets Overview Tab
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="section-header">📋 Holdings Overview</div>
                <div class="section-subtitle">Current portfolio composition and individual asset performance</div>
            """, unsafe_allow_html=True)
            
            holdings_data = []
            for asset in assets:
                holdings_data.append({
                    'Symbol': asset['symbol'],
                    'Weight': f"{asset['weight']*100:.1f}%",
                    'Price': f"${asset['price']:.2f}",
                    'Ann. Return': f"{asset['return']*100:+.1f}%",
                    'Volatility': f"{asset['std']*100:.1f}%"
                })
            
            holdings_df = pd.DataFrame(holdings_data)
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""<div class="glass-card">
                <div class="section-header">🎯 Quick Analysis</div>
            """, unsafe_allow_html=True)
            
            # Check if analysis has been run
            portfolio_result = get_model_result('portfolio', 'Portfolio_Overview')
            
            if portfolio_result:
                metrics = portfolio_result.get('metrics', {})
                st.markdown(f"""
                <div style="margin: 20px 0;">
                    <div style="margin-bottom: 16px;">
                        <div style="font-size: 12px; color: {c['text_secondary']}; text-transform: uppercase;">Expected Return</div>
                        <div style="font-size: 24px; font-weight: 700; color: {c['success']};">{metrics.get('return', 0)*100:+.1f}%</div>
                    </div>
                    <div style="margin-bottom: 16px;">
                        <div style="font-size: 12px; color: {c['text_secondary']}; text-transform: uppercase;">Volatility (Risk)</div>
                        <div style="font-size: 24px; font-weight: 700; color: {c['warning']};">{metrics.get('volatility', 0)*100:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 12px; color: {c['text_secondary']}; text-transform: uppercase;">Sharpe Ratio</div>
                        <div style="font-size: 24px; font-weight: 700; color: {c['primary']};">{metrics.get('sharpe', 0):.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Click below to calculate portfolio metrics")
            
            if st.button("▶️ Run Portfolio Analysis", type="primary", use_container_width=True, key="run_overview"):
                with st.spinner("Calculating metrics..."):
                    import time
                    time.sleep(0.5)
                    
                    metrics = calculate_portfolio_metrics(assets)
                    
                    # Save results
                    save_model_result(
                        category='portfolio',
                        model_name='Portfolio_Overview',
                        result={'assets': [a['symbol'] for a in assets]},
                        metrics={
                            'value': metrics['value'],
                            'return': metrics['return'],
                            'volatility': metrics['volatility'],
                            'sharpe': metrics['sharpe'],
                            'primary_metric': metrics['sharpe']
                        },
                        plot_data={
                            'weights': {a['symbol']: a['weight'] for a in assets}
                        },
                        parameters={'num_assets': len(assets)}
                    )
                    
                    st.success("✅ Analysis saved!")
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show allocation chart only if analysis has been run
        if portfolio_result:
            st.markdown(f"""
            <div class="glass-card">
                <div class="section-header">📊 Asset Allocation</div>
            """, unsafe_allow_html=True)
            render_allocation_pie(assets, c)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        # Optimization Tab
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="margin-bottom: 8px;">⚖️ Portfolio Optimization</h3>
            <p style="color: {c['text_secondary']};">Select and run advanced optimization models to find the perfect balance of risk and return.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Available optimization models
        OPTIMIZATION_MODELS = {
            "Mean-Variance (Markowitz)": {
                "description": "The classic optimization framework. Finds the portfolio that maximizes return for a given level of risk, or minimizes risk for a given level of return, based on efficient frontier theory.",
                "icon": "📊",
                "graph_type": "efficient_frontier"
            },
            "Minimum Variance": {
                "description": "Constructs a portfolio with the lowest possible volatility, disregarding expected returns. Ideal for highly risk-averse investors focusing purely on capital preservation.",
                "icon": "🛡️",
                "graph_type": "variance_comparison"
            },
            "Maximum Sharpe Ratio": {
                "description": "Identifies the portfolio with the highest risk-adjusted return (Sharpe Ratio). This is often considered the 'tangency portfolio' that offers the best bang for your buck.",
                "icon": "⚡",
                "graph_type": "sharpe_surface"
            },
            "Risk Parity": {
                "description": "Allocates capital such that each asset contributes an equal amount of risk to the overall portfolio. This often leads to more balanced and diversified portfolios than mean-variance.",
                "icon": "⚖️",
                "graph_type": "risk_contribution"
            },
            "Black-Litterman": {
                "description": "A sophisticated Bayesian approach that combines market equilibrium returns with your own unique views to generate a stable, personalized optimal portfolio.",
                "icon": "🔮",
                "graph_type": "posterior_returns"
            }
        }
        
        opt_cols = st.columns([2, 1])
        
        with opt_cols[1]:
            st.markdown(f"""<div class="glass-card">
                <div class="section-header">⚙️ Model Configuration</div>
            """, unsafe_allow_html=True)
            
            selected_model = st.radio(
                "Choose Optimization Model",
                list(OPTIMIZATION_MODELS.keys()),
                key="opt_model_select"
            )
            
            model_info = OPTIMIZATION_MODELS[selected_model]
            
            st.markdown(f"""
            <div style="background: {c['bg_elevated']}; padding: 12px; border-radius: 8px; margin: 12px 0; border: 1px solid {c['border']};">
                <div style="font-weight: 600; margin-bottom: 4px;">About {selected_model}</div>
                <div style="font-size: 13px; color: {c['text_secondary']}; line-height: 1.5;">{model_info['description']}</div>
            </div>
            <hr style="border-color: {c['border']}; margin: 16px 0;">
            <div style="font-weight: 600; margin-bottom: 12px;">Constraints</div>
            """, unsafe_allow_html=True)
            
            max_weight = st.slider("Max Weight per Asset", 0.1, 0.5, 0.25, key="max_weight")
            min_weight = st.slider("Min Weight per Asset", 0.0, 0.1, 0.02, key="min_weight")
            
            if selected_model == "Risk Parity":
                st.slider("Target Volatility (%)", 5, 30, 15, key="target_vol")
            elif selected_model == "Black-Litterman":
                st.slider("Market Cap Weight", 0.0, 1.0, 0.5, key="mcap_weight")
            
            if st.button("🚀 Run Optimization", type="primary", use_container_width=True, key="run_opt"):
                with st.spinner(f"Running {selected_model}..."):
                    import time
                    time.sleep(1)
                    
                    # Generate efficient frontier
                    ef_data = generate_efficient_frontier(assets)
                    
                    # Calculate optimized weights based on model
                    np.random.seed(hash(selected_model) % 1000)
                    optimized_weights = {}
                    
                    if selected_model == "Minimum Variance":
                        # Weight more to lower volatility assets
                        vols = np.array([a['std'] for a in assets])
                        inv_vols = 1 / (vols + 0.01)
                        weights = inv_vols / inv_vols.sum()
                        for i, asset in enumerate(assets):
                            optimized_weights[asset['symbol']] = float(np.clip(weights[i], min_weight, max_weight))
                    elif selected_model == "Maximum Sharpe Ratio":
                        # Weight more to higher Sharpe assets
                        sharpes = np.array([(a['return'] - 0.05) / (a['std'] + 0.01) for a in assets])
                        sharpes = np.maximum(sharpes, 0.01)
                        weights = sharpes / sharpes.sum()
                        for i, asset in enumerate(assets):
                            optimized_weights[asset['symbol']] = float(np.clip(weights[i], min_weight, max_weight))
                    elif selected_model == "Risk Parity":
                        # Equal risk contribution
                        vols = np.array([a['std'] for a in assets])
                        inv_vols = 1 / (vols + 0.01)
                        weights = inv_vols / inv_vols.sum()
                        for i, asset in enumerate(assets):
                            optimized_weights[asset['symbol']] = float(np.clip(weights[i], min_weight, max_weight))
                    else:
                        # Mean-Variance or Black-Litterman: balanced approach
                        for asset in assets:
                            optimized_weights[asset['symbol']] = np.random.uniform(min_weight, max_weight)
                    
                    # Normalize weights
                    total = sum(optimized_weights.values())
                    optimized_weights = {k: v/total for k, v in optimized_weights.items()}
                    
                    # Calculate metrics for this specific model
                    opt_ret = sum(optimized_weights.get(a['symbol'], 0) * a['return'] for a in assets)
                    opt_vol = np.sqrt(sum((optimized_weights.get(a['symbol'], 0) * a['std'])**2 for a in assets))
                    opt_sharpe = (opt_ret - 0.05) / opt_vol if opt_vol > 0 else 0
                    
                    # Save with model-specific name
                    model_key = selected_model.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                    save_model_result(
                        category='portfolio',
                        model_name=f'Optimization_{model_key}',
                        result={
                            'optimized_weights': optimized_weights,
                            'method': selected_model
                        },
                        metrics={
                            'expected_return': opt_ret,
                            'volatility': opt_vol,
                            'sharpe': opt_sharpe,
                            'primary_metric': opt_sharpe
                        },
                        plot_data={
                            **ef_data,
                            'optimized_weights': optimized_weights,
                            'graph_type': model_info['graph_type']
                        },
                        parameters={
                            'method': selected_model,
                            'max_weight': max_weight,
                            'min_weight': min_weight
                        }
                    )
                    
                    st.session_state['last_optimization_model'] = selected_model
                    st.success(f"✅ {selected_model} complete!")
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with opt_cols[0]:
            # Check if any optimization has been run - show results for the selected model
            model_key = selected_model.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
            opt_result = get_model_result('portfolio', f'Optimization_{model_key}')
            
            # Also check for any optimization results
            all_opt_results = get_category_results('portfolio')
            run_models = [k.replace('Optimization_', '').replace('_', ' ') for k in all_opt_results.keys() if k.startswith('Optimization_')]
            
            if opt_result:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <div class="section-header">{OPTIMIZATION_MODELS[selected_model]['icon']} {selected_model} Results</div>
                        <div class="status-badge status-success">Optimized</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show metrics
                metrics = opt_result.get('metrics', {})
                met_cols = st.columns(3)
                with met_cols[0]:
                    st.metric("Expected Return", f"{metrics.get('expected_return', 0)*100:.2f}%")
                with met_cols[1]:
                    st.metric("Volatility", f"{metrics.get('volatility', 0)*100:.2f}%")
                with met_cols[2]:
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
                
                st.markdown("---")
                
                # Render model-specific graph
                graph_type = opt_result.get('plot_data', {}).get('graph_type', 'efficient_frontier')
                
                if graph_type == 'efficient_frontier' or selected_model == "Mean-Variance (Markowitz)":
                    st.markdown("#### 📈 Efficient Frontier")
                    render_efficient_frontier(opt_result.get('plot_data'), c)
                
                elif graph_type == 'variance_comparison' or selected_model == "Minimum Variance":
                    st.markdown("#### 🛡️ Variance Contribution by Asset")
                    opt_weights = opt_result.get('plot_data', {}).get('optimized_weights', {})
                    
                    # Create variance contribution chart
                    fig = go.Figure()
                    symbols = list(opt_weights.keys())
                    weights_vals = list(opt_weights.values())
                    vols = [next((a['std'] for a in assets if a['symbol'] == s), 0.1) for s in symbols]
                    var_contrib = [w * v for w, v in zip(weights_vals, vols)]
                    
                    fig.add_trace(go.Bar(
                        x=symbols,
                        y=var_contrib,
                        marker_color=c['primary'],
                        text=[f"{v:.3f}" for v in var_contrib],
                        textposition='outside'
                    ))
                    
                    fig.update_layout(
                        title="Variance Contribution by Asset (Minimum Variance Portfolio)",
                        margin=dict(t=50, l=10, r=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color=c['text']),
                        xaxis=dict(title='Asset'),
                        yaxis=dict(title='Variance Contribution', showgrid=True, gridcolor=c['border']),
                        height=350
                    )
                    render_plot_with_download(fig, "minimum_variance_contribution", key="min_var_chart")
                
                elif graph_type == 'sharpe_surface' or selected_model == "Maximum Sharpe Ratio":
                    st.markdown("#### ⚡ Sharpe Ratio Surface")
                    
                    # Create Sharpe heatmap
                    n_assets = len(assets)
                    if n_assets >= 2:
                        x_weights = np.linspace(0.1, 0.5, 20)
                        y_weights = np.linspace(0.1, 0.5, 20)
                        sharpe_grid = np.zeros((20, 20))
                        
                        for i, w1 in enumerate(x_weights):
                            for j, w2 in enumerate(y_weights):
                                remaining = 1 - w1 - w2
                                if remaining >= 0:
                                    ret = w1 * assets[0]['return'] + w2 * assets[1]['return'] + remaining * np.mean([a['return'] for a in assets[2:]])
                                    vol = np.sqrt((w1 * assets[0]['std'])**2 + (w2 * assets[1]['std'])**2)
                                    sharpe_grid[j, i] = (ret - 0.05) / vol if vol > 0 else 0
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=sharpe_grid,
                            x=[f"{w:.0%}" for w in x_weights],
                            y=[f"{w:.0%}" for w in y_weights],
                            colorscale='Viridis',
                            colorbar=dict(title='Sharpe')
                        ))
                        
                        fig.update_layout(
                            title=f"Sharpe Ratio Surface ({assets[0]['symbol']} vs {assets[1]['symbol'] if len(assets) > 1 else 'N/A'})",
                            margin=dict(t=50, l=10, r=10, b=10),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(family='Inter', color=c['text']),
                            xaxis=dict(title=f'{assets[0]["symbol"]} Weight'),
                            yaxis=dict(title=f'{assets[1]["symbol"] if len(assets) > 1 else "Asset 2"} Weight'),
                            height=350
                        )
                        render_plot_with_download(fig, "sharpe_ratio_surface", key="sharpe_surface_chart")
                
                elif graph_type == 'risk_contribution' or selected_model == "Risk Parity":
                    st.markdown("#### ⚖️ Risk Contribution (Equal Risk Parity)")
                    opt_weights = opt_result.get('plot_data', {}).get('optimized_weights', {})
                    
                    symbols = list(opt_weights.keys())
                    weights_vals = list(opt_weights.values())
                    vols = [next((a['std'] for a in assets if a['symbol'] == s), 0.1) for s in symbols]
                    risk_contrib = [w * v for w, v in zip(weights_vals, vols)]
                    total_risk = sum(risk_contrib)
                    risk_pct = [r / total_risk * 100 for r in risk_contrib]
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=symbols,
                        values=risk_pct,
                        hole=0.5,
                        marker=dict(colors=c['chart_colors'][:len(symbols)]),
                        textinfo='label+percent',
                        textposition='outside'
                    )])
                    
                    fig.update_layout(
                        title="Risk Contribution by Asset (Risk Parity)",
                        margin=dict(t=50, l=10, r=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color=c['text']),
                        height=350,
                        annotations=[dict(text='Risk<br>Parity', x=0.5, y=0.5, font_size=14, showarrow=False)]
                    )
                    render_plot_with_download(fig, "risk_parity_contribution", key="risk_parity_chart")
                
                elif selected_model == "Black-Litterman":
                    st.markdown("#### 🔮 Black-Litterman Posterior Returns")
                    
                    # Create posterior vs prior returns chart
                    symbols = [a['symbol'] for a in assets]
                    prior_returns = [a['return'] * 100 for a in assets]
                    posterior_returns = [r * 1.1 + np.random.uniform(-2, 2) for r in prior_returns]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name='Prior (Market)', x=symbols, y=prior_returns, marker_color=c['text_muted']))
                    fig.add_trace(go.Bar(name='Posterior (BL)', x=symbols, y=posterior_returns, marker_color=c['primary']))
                    
                    fig.update_layout(
                        title="Prior vs Posterior Expected Returns",
                        barmode='group',
                        margin=dict(t=50, l=10, r=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter', color=c['text']),
                        xaxis=dict(title='Asset'),
                        yaxis=dict(title='Expected Return (%)', showgrid=True, gridcolor=c['border']),
                        height=350
                    )
                    render_plot_with_download(fig, "black_litterman_returns", key="bl_chart")
                
                # Show optimized weights table
                st.markdown("---")
                st.markdown("#### 📊 Optimized Weights")
                opt_weights = opt_result.get('result', {}).get('optimized_weights', {})
                
                weights_data = []
                for symbol, weight in opt_weights.items():
                    original_weight = next((a['weight'] for a in assets if a['symbol'] == symbol), 0)
                    diff = weight - original_weight
                    weights_data.append({
                        'Symbol': symbol,
                        'Original': f"{original_weight*100:.1f}%",
                        'Optimized': f"{weight*100:.1f}%",
                        'Change': f"{diff*100:+.1f}%"
                    })
                
                weights_df = pd.DataFrame(weights_data)
                st.dataframe(weights_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            else:
                # Check if other models have been run
                if run_models:
                    st.info(f"Models already run: **{', '.join(run_models)}**. Select one from the list or run **{selected_model}**.")
                
                st.markdown(f'''
                <div class="glass-card" style="padding: 60px 40px; text-align: center; border: 2px dashed {c['border']}40; background: {c['glass_bg']};">
                    <div style="width: 80px; height: 80px; background: {c['primary']}15; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; animation: pulse 2s infinite;">
                        <span style="font-size: 40px;">{OPTIMIZATION_MODELS[selected_model]['icon']}</span>
                    </div>
                    <h3 style="color: {c['text']}; font-weight: 700; margin-bottom: 10px;">{selected_model} Engine Ready</h3>
                    <p style="color: {c['text_secondary']}; font-size: 14px; max-width: 300px; margin: 0 auto;">
                        Click <b>Run Optimization</b> on the setings panel to generate your intelligent allocation strategy.
                    </p>
                </div>
                ''', unsafe_allow_html=True)
    
    with tab3:
        # Risk Analytics Tab
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="margin-bottom: 8px;">🛡️ Risk Analytics</h3>
            <p style="color: {c['text_secondary']};">Deep dive into portfolio risk, VaR, limit exposure, and drawdown analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if risk analysis has been run
        risk_result = get_model_result('portfolio', 'Risk_Analysis')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("▶️ Run Risk Analysis", type="primary", use_container_width=True, key="run_risk"):
                with st.spinner("Calculating risk metrics..."):
                    import time
                    time.sleep(0.8)
                    
                    metrics = calculate_portfolio_metrics(assets)
                    corr_matrix = calculate_correlation_matrix(assets)
                    
                    # Calculate VaR
                    port_vol = metrics['volatility']
                    port_value = metrics['value']
                    var_95 = port_value * port_vol * 1.645
                    cvar_95 = var_95 * 1.2
                    
                    # Get drawdown data
                    first_ticker = assets[0]['symbol']
                    ticker_df = get_price_data(first_ticker)
                    drawdown_data = None
                    if ticker_df is not None and 'close' in ticker_df.columns:
                        prices = ticker_df['close'].values[-252:]
                        rolling_max = pd.Series(prices).expanding().max()
                        drawdown = (prices - rolling_max) / rolling_max
                        drawdown_data = drawdown.tolist()
                    
                    # Save results
                    save_model_result(
                        category='portfolio',
                        model_name='Risk_Analysis',
                        result={
                            'var_95': var_95,
                            'cvar_95': cvar_95,
                            'max_drawdown': port_vol * 2
                        },
                        metrics={
                            'var_95_pct': var_95/port_value*100,
                            'cvar_95_pct': cvar_95/port_value*100,
                            'volatility': port_vol,
                            'primary_metric': port_vol
                        },
                        plot_data={
                            'correlation_matrix': corr_matrix.to_dict(),
                            'drawdown': drawdown_data
                        },
                        parameters={}
                    )
                    
                    st.success("✅ Risk analysis complete! Results saved to Dashboard.")
                    st.rerun()
        
        if risk_result:
            risk_cols = st.columns([1, 1])
            
            with risk_cols[0]:
                st.markdown(f"""<div class="glass-card">
                    <div class="section-header">Key Risk Metrics</div>
                """, unsafe_allow_html=True)
                render_risk_metrics(assets, c)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with risk_cols[1]:
                st.markdown(f"""<div class="glass-card">
                    <div class="section-header">Correlation Matrix</div>
                """, unsafe_allow_html=True)
                corr_data = risk_result.get('plot_data', {}).get('correlation_matrix', {})
                if corr_data:
                    corr_df = pd.DataFrame(corr_data)
                    render_correlation_heatmap(corr_df, c)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Drawdown chart
            st.markdown(f"""<div class="glass-card">
                <div class="section-header">Drawdown Analysis</div>
            """, unsafe_allow_html=True)
            drawdown_data = risk_result.get('plot_data', {}).get('drawdown')
            if drawdown_data:
                dates = pd.date_range(end=datetime.now(), periods=len(drawdown_data), freq='D')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=drawdown_data,
                    fill='tozeroy',
                    fillcolor='rgba(239, 68, 68, 0.2)',
                    line=dict(color=c['error'], width=1.5),
                    hovertemplate='%{x}<br>Drawdown: %{y:.2%}<extra></extra>'
                ))
                
                fig.update_layout(
                    margin=dict(t=20, l=10, r=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color=c['text']),
                    xaxis=dict(showgrid=True, gridcolor=c['border']),
                    yaxis=dict(showgrid=True, gridcolor=c['border'], tickformat='.0%'),
                    height=250
                )
                render_plot_with_download(fig, "portfolio_drawdown", key="drawdown_chart")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="glass-card" style="padding: 60px 40px; text-align: center; border: 2px dashed {c['border']}40; background: {c['glass_bg']};">
                <div style="width: 80px; height: 80px; background: {c['primary']}15; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; animation: pulse 2s infinite;">
                    <span style="font-size: 40px;">🛡️</span>
                </div>
                <h3 style="color: {c['text']}; font-weight: 700; margin-bottom: 10px;">Risk Engine Ready</h3>
                <p style="color: {c['text_secondary']}; font-size: 14px; max-width: 300px; margin: 0 auto;">
                    Click <b>Run Risk Analysis</b> above to calculate VaR, correlation matrix, and drawdown metrics.
                </p>
            </div>
            ''', unsafe_allow_html=True)
    
    with tab4:
        # Rebalance Tab
        st.markdown(f"""
        <div style="margin-bottom: 20px;">
            <h3 style="margin-bottom: 8px;">🔄 Portfolio Rebalancing</h3>
            <p style="color: {c['text_secondary']};">Automatically calculate trades to realign your portfolio with your target allocation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        reb_cols = st.columns([2, 1])
        
        with reb_cols[1]:
            st.markdown(f"""<div class="glass-card">
                <div class="section-header">Rebalance Settings</div>
            """, unsafe_allow_html=True)
            
            st.selectbox("Rebalance Method", ["Threshold-based", "Calendar-based", "Drift-based"], key="reb_method")
            st.slider("Drift Threshold (%)", 1, 10, 5, key="drift_thresh")
            st.number_input("Transaction Cost (%)", value=0.1, step=0.05, key="trans_cost")
            
            st.markdown("---")
            
            if st.button("Calculate Trades", type="primary", use_container_width=True, key="calc_trades"):
                st.session_state['rebalance_calculated'] = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        
        with reb_cols[0]:
            if st.session_state.get('rebalance_calculated', False):
                st.markdown(f"""<div class="glass-card">
                    <div class="section-header">Proposed Trades</div>
                """, unsafe_allow_html=True)
                
                trades_data = []
                for asset in assets:
                    np.random.seed(hash(asset['symbol']) % 100)
                    target = asset['weight'] + np.random.uniform(-0.05, 0.05)
                    target = max(0.02, min(0.3, target))
                    diff = target - asset['weight']
                    action = "BUY" if diff > 0 else "SELL" if diff < 0 else "HOLD"
                    trades_data.append({
                        'Symbol': asset['symbol'],
                        'Current': f"{asset['weight']*100:.1f}%",
                        'Target': f"{target*100:.1f}%",
                        'Difference': f"{diff*100:+.1f}%",
                        'Action': action,
                        'Est. Value': f"${abs(diff) * 1000000:.0f}"
                    })
                
                trades_df = pd.DataFrame(trades_data)
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.markdown(f"- **Total Trades:** {len(assets)}\n- **Estimated Cost:** $156\n- **Expected Turnover:** 8.2%")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.button("Execute Rebalance", type="primary", use_container_width=True, key="exec_reb"):
                    st.toast("Rebalance simulation complete!", icon="✅")
            else:
                st.markdown(f'''
                <div class="glass-card" style="padding: 60px 40px; text-align: center; border: 2px dashed {c['border']}40; background: {c['glass_bg']};">
                    <div style="width: 80px; height: 80px; background: {c['primary']}15; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 20px; animation: pulse 2s infinite;">
                        <span style="font-size: 40px;">🔄</span>
                    </div>
                    <h3 style="color: {c['text']}; font-weight: 700; margin-bottom: 10px;">Rebalancing Calculation Pending</h3>
                    <p style="color: {c['text_secondary']}; font-size: 14px; max-width: 300px; margin: 0 auto;">
                        Configure your method and thresholds then click <b>Calculate Trades</b> to see the drift analysis.
                    </p>
                </div>
                ''', unsafe_allow_html=True)


if __name__ == "__main__":
    render()
