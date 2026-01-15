"""
Financial Analytics Suite - Scenario Analysis Page
Stress testing, Monte Carlo simulation, and scenario comparison
SEPARATE SECTIONS: Default Scenarios vs Your Data Analysis
WITH DOWNLOAD OPTIONS for graphs and tables
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from typing import Dict, Optional
import io

# Import data manager
from pages_new.data_manager import (
    get_working_data, get_working_data_info, has_data,
    get_available_tickers, get_price_data
)

# Import theme utilities
from pages_new.theme_utils import get_theme_colors

# Import model results manager
from pages_new.model_results_manager import (
    save_model_result, get_model_result, get_category_results
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


# Pre-defined historical scenarios with fixed data
DEFAULT_SCENARIOS = {
    "2008 Financial Crisis": {
        "description": "Global financial crisis triggered by subprime mortgage collapse",
        "equity_shock": -40,
        "volatility": 80,
        "duration_days": 365,
        "returns_data": np.array([-0.08, -0.12, -0.05, -0.15, -0.10, 0.03, -0.08, -0.06, 0.02, -0.04, -0.07, 0.05])
    },
    "COVID-19 Crash (2020)": {
        "description": "Pandemic-induced market crash in March 2020",
        "equity_shock": -35,
        "volatility": 70,
        "duration_days": 90,
        "returns_data": np.array([-0.12, -0.15, -0.20, 0.08, 0.05, 0.10, -0.03, 0.07, 0.12, 0.08, 0.05, 0.03])
    },
    "Dot-com Bubble (2000)": {
        "description": "Technology stock bubble burst",
        "equity_shock": -25,
        "volatility": 40,
        "duration_days": 730,
        "returns_data": np.array([-0.05, -0.08, -0.10, -0.06, -0.03, 0.02, -0.04, -0.07, -0.02, 0.01, -0.05, -0.03])
    },
    "2022 Rate Hike": {
        "description": "Fed aggressive rate hiking cycle",
        "equity_shock": -20,
        "volatility": 30,
        "duration_days": 365,
        "returns_data": np.array([-0.05, -0.03, 0.02, -0.06, -0.04, 0.01, -0.02, -0.05, 0.03, -0.01, -0.03, 0.02])
    }
}


def calculate_returns(df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
    """Calculate daily returns from price data"""
    if price_col in df.columns:
        return df[price_col].pct_change().dropna()
    return pd.Series([])


def run_monte_carlo(initial_value: float, mean_return: float, volatility: float, 
                    n_simulations: int, n_days: int, seed: int = 42) -> np.ndarray:
    """Run Monte Carlo simulation for portfolio values"""
    np.random.seed(seed)
    paths = np.zeros((n_days, n_simulations))
    paths[0] = initial_value
    
    for i in range(1, n_days):
        returns = np.random.normal(mean_return, volatility, n_simulations)
        paths[i] = paths[i-1] * (1 + returns)
    
    return paths


def render_default_scenario_section(c: Dict):
    """Render the Default/Historical Scenarios section"""
    st.markdown("### 📚 Historical Scenarios (Pre-defined)")
    st.caption("Explore how historical market events would impact a portfolio. These use fixed historical data.")
    
    # Scenario selector
    selected_scenario = st.selectbox(
        "Select Historical Scenario", 
        list(DEFAULT_SCENARIOS.keys()),
        key="default_scenario_select"
    )
    
    if selected_scenario:
        scenario = DEFAULT_SCENARIOS[selected_scenario]
        
        # Show scenario info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"#### {selected_scenario}")
            st.markdown(f"*{scenario['description']}*")
            
            info_cols = st.columns(3)
            with info_cols[0]:
                shock_color = c['error'] if scenario['equity_shock'] < -20 else c['warning']
                st.markdown(f"**Equity Shock:** <span style='color:{shock_color}'>{scenario['equity_shock']}%</span>", unsafe_allow_html=True)
            with info_cols[1]:
                st.markdown(f"**Volatility:** {scenario['volatility']}%")
            with info_cols[2]:
                st.markdown(f"**Duration:** {scenario['duration_days']} days")
        
        with col2:
            initial_value = st.number_input("Portfolio Value ($)", 100000, 10000000, 1000000, step=100000, key="default_init_val")
        
        # Run analysis button
        if st.button("▶️ Run Historical Scenario Analysis", type="primary", use_container_width=True, key="run_default_scenario"):
            with st.spinner(f"Simulating {selected_scenario}..."):
                import time
                time.sleep(0.8)
                
                # Calculate impact
                shocked_value = initial_value * (1 + scenario['equity_shock'] / 100)
                loss = initial_value - shocked_value
                
                # Generate simulation paths based on historical data
                monthly_returns = scenario['returns_data']
                daily_vol = np.std(monthly_returns) / np.sqrt(21)
                daily_mean = np.mean(monthly_returns) / 21
                
                paths = run_monte_carlo(initial_value, daily_mean, daily_vol, 200, scenario['duration_days'], seed=hash(selected_scenario) % 1000)
                
                st.session_state[f'default_scenario_{selected_scenario}'] = {
                    'paths': paths,
                    'shocked_value': shocked_value,
                    'loss': loss,
                    'scenario': scenario
                }
            
            st.success(f"✅ {selected_scenario} simulation complete!")
        
        # Show results if available
        result_key = f'default_scenario_{selected_scenario}'
        if result_key in st.session_state:
            results = st.session_state[result_key]
            
            st.markdown("---")
            st.markdown("#### 📊 Results")
            
            # Metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Initial Value", f"${initial_value:,.0f}")
            with metric_cols[1]:
                st.metric("Shocked Value", f"${results['shocked_value']:,.0f}", delta=f"{scenario['equity_shock']}%")
            with metric_cols[2]:
                st.metric("Expected Loss", f"-${abs(results['loss']):,.0f}")
            with metric_cols[3]:
                prob_loss = (results['paths'][-1, :] < initial_value).mean() * 100
                st.metric("Prob of Loss", f"{prob_loss:.1f}%")
            
            # Chart
            paths = results['paths']
            fig = go.Figure()
            
            # Sample paths
            n_show = min(50, paths.shape[1])
            for j in range(n_show):
                fig.add_trace(go.Scatter(
                    x=list(range(paths.shape[0])),
                    y=paths[:, j],
                    mode='lines',
                    line=dict(width=0.5, color=c['primary']),
                    opacity=0.2,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Percentiles
            p5 = np.percentile(paths, 5, axis=1)
            p50 = np.percentile(paths, 50, axis=1)
            p95 = np.percentile(paths, 95, axis=1)
            
            fig.add_trace(go.Scatter(x=list(range(len(p5))), y=p5, mode='lines', 
                                    line=dict(color=c['error'], width=2), name='Worst Case (5%)'))
            fig.add_trace(go.Scatter(x=list(range(len(p50))), y=p50, mode='lines', 
                                    line=dict(color=c['accent'], width=2), name='Median'))
            fig.add_trace(go.Scatter(x=list(range(len(p95))), y=p95, mode='lines', 
                                    line=dict(color=c['success'], width=2), name='Best Case (95%)'))
            
            fig.update_layout(
                title=f"{selected_scenario} - Portfolio Simulation",
                margin=dict(t=50, l=10, r=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color=c['text']),
                xaxis=dict(title='Trading Days', showgrid=True, gridcolor=c['border']),
                yaxis=dict(title='Portfolio Value ($)', showgrid=True, gridcolor=c['border'], tickprefix='$', tickformat=',.0f'),
                legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"default_scenario_chart_{selected_scenario.replace(' ', '_')}")


def render_user_data_scenario_section(c: Dict):
    """Render the User Data Scenario Analysis section"""
    st.markdown("### 📊 Your Data - Custom Scenario Analysis")
    st.caption("Run stress tests and Monte Carlo simulations on YOUR uploaded data")
    
    if not has_data():
        st.warning("⚠️ Upload your data in Data Sources to enable custom scenario analysis on your portfolio.")
        if st.button("📥 Go to Data Sources", key="goto_data_scenario"):
            st.session_state.current_page = 'data_sources'
            st.rerun()
        return
    
    # Get user data
    df = get_working_data()
    tickers = get_available_tickers()
    data_info = get_working_data_info()
    
    st.markdown(f'<div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 12px 20px; margin-bottom: 20px;"><span style="font-size: 16px;">✅</span> <span style="color: {c["text"]}; font-weight: 600;">Using Your Data:</span> <span style="color: {c["text_secondary"]};">{data_info["name"]} ({data_info["rows"]:,} rows, {len(tickers)} assets)</span></div>', unsafe_allow_html=True)
    
    # Analysis tabs
    sub_tab1, sub_tab2 = st.tabs(["🎲 Monte Carlo", "🔥 Stress Test"])
    
    with sub_tab1:
        st.markdown("#### Monte Carlo Simulation on Your Data")
        
        config_col, result_col = st.columns([1, 2])
        
        with config_col:
            st.markdown("##### ⚙️ Configuration")
            
            if tickers:
                selected_ticker = st.selectbox("Select Asset", tickers, key="user_mc_ticker")
                ticker_df = get_price_data(selected_ticker)
                
                if ticker_df is not None and 'close' in ticker_df.columns:
                    returns = ticker_df['close'].pct_change().dropna()
                    real_mean = returns.mean()
                    real_vol = returns.std()
                    st.success(f"📊 Stats from {selected_ticker}")
                    st.caption(f"Daily Mean: {real_mean:.6f} | Vol: {real_vol:.4f}")
                else:
                    real_mean = 0.0004
                    real_vol = 0.015
                    st.warning("Using default parameters")
            else:
                real_mean = 0.0004
                real_vol = 0.015
                st.warning("No tickers found")
            
            n_sims = st.number_input("Simulations", 100, 10000, 1000, key="user_n_sims")
            n_days = st.number_input("Horizon (days)", 1, 365, 252, key="user_n_days")
            initial_value = st.number_input("Initial Value ($)", 10000, 10000000, 1000000, step=10000, key="user_init_val")
            
            run_mc = st.button("▶️ Run Monte Carlo", type="primary", use_container_width=True, key="run_user_mc")
        
        with result_col:
            if run_mc:
                with st.spinner("Running Monte Carlo on your data..."):
                    paths = run_monte_carlo(initial_value, real_mean, real_vol, min(n_sims, 500), n_days, seed=123)
                    
                    # Save to model results manager
                    final_values = paths[-1, :]
                    save_model_result(
                        category='scenario',
                        model_name=f'Monte_Carlo_{selected_ticker if tickers else "Portfolio"}',
                        result={'final_values': final_values.tolist()[:100]},
                        metrics={
                            'expected_value': float(np.median(final_values)),
                            'var_95': float(initial_value - np.percentile(final_values, 5)),
                            'prob_loss': float((final_values < initial_value).mean()),
                            'primary_metric': float(np.median(final_values))
                        },
                        plot_data={
                            'simulations': paths[:, :20].tolist(),
                            'mean_path': np.percentile(paths, 50, axis=1).tolist()
                        },
                        parameters={'n_sims': n_sims, 'n_days': n_days, 'initial_value': initial_value}
                    )
                    
                    st.session_state['user_mc_paths'] = paths
                
                st.success("✅ Simulation complete! Results saved to Dashboard.")
            
            if 'user_mc_paths' in st.session_state:
                paths = st.session_state['user_mc_paths']
                
                fig = go.Figure()
                
                n_show = min(100, paths.shape[1])
                for j in range(n_show):
                    fig.add_trace(go.Scatter(
                        x=list(range(paths.shape[0])),
                        y=paths[:, j],
                        mode='lines',
                        line=dict(width=0.5, color=c['primary']),
                        opacity=0.2,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                p5 = np.percentile(paths, 5, axis=1)
                p50 = np.percentile(paths, 50, axis=1)
                p95 = np.percentile(paths, 95, axis=1)
                
                fig.add_trace(go.Scatter(x=list(range(len(p5))), y=p5, mode='lines', 
                                        line=dict(color=c['error'], width=2), name='5th Percentile'))
                fig.add_trace(go.Scatter(x=list(range(len(p50))), y=p50, mode='lines', 
                                        line=dict(color=c['accent'], width=2), name='Median'))
                fig.add_trace(go.Scatter(x=list(range(len(p95))), y=p95, mode='lines', 
                                        line=dict(color=c['success'], width=2), name='95th Percentile'))
                
                fig.update_layout(
                    title="Monte Carlo Simulation (Your Data)",
                    margin=dict(t=50, l=10, r=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter', color=c['text']),
                    xaxis=dict(title='Trading Days', showgrid=True, gridcolor=c['border']),
                    yaxis=dict(title='Portfolio Value ($)', showgrid=True, gridcolor=c['border'], tickprefix='$', tickformat=',.0f'),
                    legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="user_mc_chart")
                
                # Stats
                final_values = paths[-1, :]
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Best Case (95%)", f"${np.percentile(final_values, 95):,.0f}")
                with stat_cols[1]:
                    st.metric("Expected (Median)", f"${np.median(final_values):,.0f}")
                with stat_cols[2]:
                    st.metric("Worst Case (5%)", f"${np.percentile(final_values, 5):,.0f}")
                with stat_cols[3]:
                    prob_loss = (final_values < initial_value).mean() * 100
                    st.metric("Prob of Loss", f"{prob_loss:.1f}%")
            else:
                st.info("👆 Configure parameters and click 'Run Monte Carlo' to see results")
    
    with sub_tab2:
        st.markdown("#### Custom Stress Test on Your Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("##### Define Your Stress Scenario")
            
            custom_name = st.text_input("Scenario Name", "My Custom Stress Test", key="user_stress_name")
            
            shock_cols = st.columns(3)
            with shock_cols[0]:
                equity_shock = st.slider("Equity Shock", -50, 20, -20, format="%d%%", key="user_equity_shock")
            with shock_cols[1]:
                vol_increase = st.slider("Volatility Increase", 0, 100, 50, format="%d%%", key="user_vol_shock")
            with shock_cols[2]:
                duration = st.number_input("Duration (days)", 30, 365, 90, key="user_stress_duration")
        
        with col2:
            stress_init_val = st.number_input("Portfolio Value ($)", 100000, 10000000, 1000000, step=100000, key="user_stress_init")
        
        if st.button("▶️ Run Custom Stress Test", type="primary", use_container_width=True, key="run_user_stress"):
            with st.spinner("Running stress test on your data..."):
                import time
                time.sleep(0.5)
                
                # Calculate stressed value
                shocked_value = stress_init_val * (1 + equity_shock / 100)
                loss = stress_init_val - shocked_value
                
                # Generate stress path using real volatility if available
                if tickers and ticker_df is not None and 'close' in ticker_df.columns:
                    base_vol = returns.std() * (1 + vol_increase / 100)
                else:
                    base_vol = 0.02 * (1 + vol_increase / 100)
                
                # Create stress path
                stress_paths = run_monte_carlo(stress_init_val, equity_shock/100/duration, base_vol, 100, duration, seed=456)
                
                # Save to model results manager
                save_model_result(
                    category='scenario',
                    model_name=f'Stress_Test_{custom_name.replace(" ", "_")}',
                    result={'shocked_value': shocked_value, 'loss': loss},
                    metrics={
                        'shocked_value': shocked_value,
                        'loss': loss,
                        'equity_shock': equity_shock,
                        'primary_metric': equity_shock
                    },
                    plot_data={
                        'stress_paths': stress_paths[:, :10].tolist()
                    },
                    parameters={'name': custom_name, 'equity_shock': equity_shock, 'vol_increase': vol_increase}
                )
                
                st.session_state['user_stress_result'] = {
                    'shocked_value': shocked_value,
                    'loss': loss,
                    'paths': stress_paths,
                    'name': custom_name
                }
            
            st.success("✅ Stress test complete! Results saved to Dashboard.")
        
        if 'user_stress_result' in st.session_state:
            result = st.session_state['user_stress_result']
            
            st.markdown("---")
            st.markdown(f"#### Results: {result['name']}")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Initial Value", f"${stress_init_val:,.0f}")
            with metric_cols[1]:
                st.metric("Stressed Value", f"${result['shocked_value']:,.0f}", delta=f"{equity_shock}%")
            with metric_cols[2]:
                st.metric("Expected Loss", f"-${abs(result['loss']):,.0f}")
            with metric_cols[3]:
                st.metric("Volatility Impact", f"+{vol_increase}%")
            
            # Show stress paths
            paths = result['paths']
            fig = go.Figure()
            
            for j in range(min(20, paths.shape[1])):
                fig.add_trace(go.Scatter(
                    x=list(range(paths.shape[0])),
                    y=paths[:, j],
                    mode='lines',
                    line=dict(width=1, color=c['error']),
                    opacity=0.3,
                    showlegend=False
                ))
            
            fig.add_trace(go.Scatter(
                x=list(range(paths.shape[0])),
                y=np.median(paths, axis=1),
                mode='lines',
                line=dict(width=2, color=c['warning']),
                name='Expected Path'
            ))
            
            fig.update_layout(
                title=f"Stress Test: {result['name']}",
                margin=dict(t=50, l=10, r=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color=c['text']),
                xaxis=dict(title='Days', showgrid=True, gridcolor=c['border']),
                yaxis=dict(title='Portfolio Value ($)', showgrid=True, gridcolor=c['border'], tickprefix='$', tickformat=',.0f'),
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key="user_stress_chart")


def render():
    """Render the Scenario Analysis page"""
    c = get_theme_colors()
    
    st.title("🎭 Scenario Analysis")
    st.markdown(f"<p style='color: {c['text_secondary']};'>Stress test and simulate different market scenarios</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main tabs separating Default Scenarios from User Data Analysis
    tab1, tab2 = st.tabs(["📚 Historical Scenarios", "📊 Your Data Analysis"])
    
    with tab1:
        render_default_scenario_section(c)
    
    with tab2:
        render_user_data_scenario_section(c)


if __name__ == "__main__":
    render()
