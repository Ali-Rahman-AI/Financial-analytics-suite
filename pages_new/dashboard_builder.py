"""
Financial Analytics Suite - Dashboard Builder Page  
Shows results from models that have been RUN - No auto-generation without user action
WITH DOWNLOAD OPTIONS for all graphs, tables, and full dashboard export
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import io
import json

# Import data manager
from pages_new.data_manager import (
    get_working_data, get_working_data_info, has_data,
    get_available_tickers, get_price_data
)

# Import model results manager
from pages_new.model_results_manager import (
    get_dashboard_data, get_all_run_models, get_model_result,
    get_run_history, count_run_models, get_models_for_comparison,
    get_category_results
)


# Import theme utilities
from pages_new.theme_utils import get_theme_colors, inject_premium_styles


def fig_to_image_bytes(fig, format='png'):
    """Convert plotly figure to image bytes for download"""
    try:
        img_bytes = fig.to_image(format=format, width=1200, height=600, scale=2)
        return img_bytes
    except Exception as e:
        return None


def dataframe_to_csv(df):
    """Convert dataframe to CSV bytes"""
    return df.to_csv(index=False).encode('utf-8')


def dataframe_to_excel(df):
    """Convert dataframe to Excel bytes"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    return output.getvalue()


def render_download_buttons(fig=None, df=None, name="download", c=None):
    """Render download buttons for figure and/or dataframe"""
    cols = st.columns(4)
    
    if fig is not None:
        with cols[0]:
            # PNG download
            try:
                png_bytes = fig.to_image(format='png', width=1200, height=600, scale=2)
                st.download_button(
                    "📥 PNG",
                    data=png_bytes,
                    file_name=f"{name}.png",
                    mime="image/png",
                    key=f"dl_png_{name}_{hash(str(fig))%10000}",
                    use_container_width=True
                )
            except:
                st.button("📥 PNG", disabled=True, use_container_width=True, key=f"dl_png_disabled_{name}")
        
        with cols[1]:
            # HTML download (interactive)
            html_str = fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                "📥 HTML",
                data=html_str,
                file_name=f"{name}.html",
                mime="text/html",
                key=f"dl_html_{name}_{hash(str(fig))%10000}",
                use_container_width=True
            )
    
    if df is not None:
        with cols[2]:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 CSV",
                data=csv_data,
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"dl_csv_{name}_{hash(str(df.values.tobytes()))%10000}",
                use_container_width=True
            )
        
        with cols[3]:
            try:
                excel_data = dataframe_to_excel(df)
                st.download_button(
                    "📥 Excel",
                    data=excel_data,
                    file_name=f"{name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_xlsx_{name}_{hash(str(df.values.tobytes()))%10000}",
                    use_container_width=True
                )
            except:
                st.button("📥 Excel", disabled=True, use_container_width=True, key=f"dl_xlsx_disabled_{name}")


def render_no_results_message(c: Dict):
    """Render message when no models have been run"""
    st.markdown(f'''
    <div style="background: {c['glass_bg']}; border: 2px dashed {c['border']}; border-radius: 20px; padding: 60px 40px; text-align: center; margin: 40px 0;">
        <div style="font-size: 72px; margin-bottom: 20px;">📊</div>
        <h2 style="color: {c['text']}; font-size: 28px; margin-bottom: 16px;">No Analysis Results Yet</h2>
        <p style="color: {c['text_secondary']}; font-size: 16px; max-width: 500px; margin: 0 auto 24px;">
            The dashboard displays results from models you have run. Go to the analysis pages to run forecasting, 
            ML models, portfolio optimization, or scenario analysis.
        </p>
        <p style="color: {c['text_muted']}; font-size: 14px;">
            Once you run any model, its results will automatically appear here.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Quick action buttons
    st.markdown("### 🚀 Quick Actions - Run Your First Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔮 Time Series Forecast", use_container_width=True, key="goto_forecast"):
            st.session_state.current_page = 'forecasting'
            st.rerun()
    
    with col2:
        if st.button("🧪 Machine Learning", use_container_width=True, key="goto_ml"):
            st.session_state.current_page = 'ml_lab'
            st.rerun()
    
    with col3:
        if st.button("💼 Portfolio Analysis", use_container_width=True, key="goto_portfolio"):
            st.session_state.current_page = 'portfolio'
            st.rerun()
    
    with col4:
        if st.button("📈 Scenario Analysis", use_container_width=True, key="goto_scenario"):
            st.session_state.current_page = 'scenario'
            st.rerun()


def create_forecasting_figure(plot_data: Dict, c: Dict) -> go.Figure:
    """Create a forecasting plot figure"""
    fig = go.Figure()
    
    # Historical data
    if 'historical_dates' in plot_data and 'historical_values' in plot_data:
        fig.add_trace(go.Scatter(
            x=plot_data['historical_dates'],
            y=plot_data['historical_values'],
            mode='lines',
            name='Historical',
            line=dict(color=c['text_secondary'], width=2)
        ))
    
    # Forecast
    if 'forecast_dates' in plot_data and 'forecast_values' in plot_data:
        fig.add_trace(go.Scatter(
            x=plot_data['forecast_dates'],
            y=plot_data['forecast_values'],
            mode='lines',
            name='Forecast',
            line=dict(color=c['primary'], width=2, dash='dash')
        ))
    
    # Confidence interval
    if 'upper_bound' in plot_data and 'lower_bound' in plot_data:
        fig.add_trace(go.Scatter(
            x=plot_data['forecast_dates'] + plot_data['forecast_dates'][::-1],
            y=plot_data['upper_bound'] + plot_data['lower_bound'][::-1],
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
    
    fig.update_layout(
        title="Forecast Results",
        margin=dict(t=50, l=10, r=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=c['text']),
        xaxis=dict(showgrid=True, gridcolor=c['border']),
        yaxis=dict(showgrid=True, gridcolor=c['border']),
        legend=dict(orientation='h', y=1.1),
        height=350
    )
    
    return fig


def create_ml_figure(plot_data: Dict, c: Dict) -> go.Figure:
    """Create an ML model plot figure"""
    fig = go.Figure()
    
    if 'feature_importance' in plot_data:
        features = plot_data['feature_importance']
        fig = go.Figure(go.Bar(
            x=list(features.values()),
            y=list(features.keys()),
            orientation='h',
            marker_color=c['primary']
        ))
        fig.update_layout(
            title="Feature Importance",
            margin=dict(t=50, l=10, r=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color=c['text']),
            height=350
        )
    
    elif 'actual' in plot_data and 'predicted' in plot_data:
        fig.add_trace(go.Scatter(
            x=plot_data.get('x', list(range(len(plot_data['actual'])))),
            y=plot_data['actual'],
            mode='lines',
            name='Actual',
            line=dict(color=c['text_secondary'], width=2)
        ))
        fig.add_trace(go.Scatter(
            x=plot_data.get('x', list(range(len(plot_data['predicted'])))),
            y=plot_data['predicted'],
            mode='lines',
            name='Predicted',
            line=dict(color=c['primary'], width=2)
        ))
        fig.update_layout(
            title="Actual vs Predicted",
            margin=dict(t=50, l=10, r=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color=c['text']),
            height=350
        )
    
    return fig


def create_portfolio_figure(plot_data: Dict, c: Dict) -> go.Figure:
    """Create a portfolio analysis plot figure"""
    fig = go.Figure()
    
    if 'weights' in plot_data or 'optimized_weights' in plot_data:
        weights = plot_data.get('optimized_weights', plot_data.get('weights', {}))
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.5,
            marker=dict(colors=c['chart_colors'][:len(weights)])
        )])
        fig.update_layout(
            title="Portfolio Allocation",
            margin=dict(t=50, l=10, r=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color=c['text']),
            height=350
        )
    
    elif 'random_returns' in plot_data and 'random_vol' in plot_data:
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=plot_data['random_vol'][:100],
            y=plot_data['random_returns'][:100],
            mode='markers',
            marker=dict(size=4, color=c['primary'], opacity=0.4),
            name='Portfolios'
        ))
        if 'frontier_vol' in plot_data:
            fig.add_trace(go.Scatter(
                x=plot_data['frontier_vol'],
                y=plot_data['frontier_returns'],
                mode='lines',
                line=dict(color=c['accent'], width=3),
                name='Efficient Frontier'
            ))
        fig.update_layout(
            title="Efficient Frontier",
            margin=dict(t=50, l=10, r=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color=c['text']),
            xaxis=dict(title='Volatility', tickformat='.0%'),
            yaxis=dict(title='Return', tickformat='.0%'),
            height=350
        )
    
    return fig


def create_scenario_figure(plot_data: Dict, c: Dict) -> go.Figure:
    """Create a scenario analysis plot figure"""
    fig = go.Figure()
    
    if 'simulations' in plot_data:
        # Monte Carlo paths
        for i, sim in enumerate(plot_data['simulations'][:20]):
            fig.add_trace(go.Scatter(
                x=list(range(len(sim))),
                y=sim,
                mode='lines',
                line=dict(color=c['primary'], width=0.5),
                opacity=0.3,
                showlegend=False
            ))
    
    if 'mean_path' in plot_data:
        fig.add_trace(go.Scatter(
            x=list(range(len(plot_data['mean_path']))),
            y=plot_data['mean_path'],
            mode='lines',
            name='Mean',
            line=dict(color=c['warning'], width=2)
        ))
    
    fig.update_layout(
        title="Monte Carlo Simulation",
        margin=dict(t=50, l=10, r=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color=c['text']),
        height=350
    )
    
    return fig


def render_model_result_section(category: str, model_name: str, result: Dict, c: Dict, index: int):
    """Render a complete section for a model result with metrics, plot, and download options"""
    metrics = result.get('metrics', {})
    plot_data = result.get('plot_data', {})
    timestamp = result.get('timestamp', '')
    
    # Format timestamp
    try:
        dt = datetime.fromisoformat(timestamp)
        time_str = dt.strftime('%b %d, %H:%M')
    except:
        time_str = 'N/A'
    
    # Category icons
    icons = {
        'forecasting': '🔮',
        'ml_models': '🧪',
        'portfolio': '💼',
        'scenario': '📈',
        'risk': '🛡️'
    }
    icon = icons.get(category, '📊')
    
    # Section container
    st.markdown(f'''
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 16px; padding: 20px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <span style="font-size: 28px; margin-right: 12px;">{icon}</span>
                <span style="font-size: 20px; font-weight: 700; color: {c['text']};">{model_name}</span>
                <span style="font-size: 12px; color: {c['text_muted']}; margin-left: 12px; background: {c['bg_surface']}; padding: 4px 10px; border-radius: 6px;">{category.replace('_', ' ').title()}</span>
            </div>
            <span style="font-size: 11px; color: {c['text_muted']};">Run at: {time_str}</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Metrics row
    st.markdown("##### 📊 Key Metrics")
    metric_cols = st.columns(min(len(metrics), 5))
    
    metrics_df_data = []
    for i, (key, value) in enumerate(list(metrics.items())[:5]):
        if key != 'primary_metric':
            with metric_cols[i % len(metric_cols)]:
                if isinstance(value, float):
                    if abs(value) < 1:
                        display_val = f"{value:.4f}"
                    else:
                        display_val = f"{value:.2f}"
                else:
                    display_val = str(value)
                st.metric(key.replace('_', ' ').title(), display_val)
                metrics_df_data.append({'Metric': key.replace('_', ' ').title(), 'Value': display_val})
    
    # Create metrics dataframe for download
    metrics_df = pd.DataFrame(metrics_df_data) if metrics_df_data else None
    
    # Plot section
    if plot_data:
        st.markdown("##### 📈 Visualization")
        
        # Create figure based on category
        if category == 'forecasting':
            fig = create_forecasting_figure(plot_data, c)
        elif category == 'ml_models':
            fig = create_ml_figure(plot_data, c)
        elif category == 'portfolio':
            fig = create_portfolio_figure(plot_data, c)
        elif category == 'scenario':
            fig = create_scenario_figure(plot_data, c)
        else:
            fig = go.Figure()
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True}, 
                       key=f"dashboard_plot_{category}_{model_name.replace(' ', '_')}_{index}")
        
        # Download options for this model
        st.markdown("##### 💾 Download Options")
        render_download_buttons(fig=fig, df=metrics_df, name=f"{category}_{model_name.replace(' ', '_')}", c=c)
    
    st.markdown("---")


def render_model_comparison(c: Dict):
    """Render comparison of models that have actually been run"""
    models_for_comparison = get_models_for_comparison()
    
    if len(models_for_comparison) < 2:
        st.info("Run at least 2 models to enable comparison. Currently you have run {} model(s).".format(len(models_for_comparison)))
        return
    
    st.markdown("### 📊 Model Comparison (Only Run Models)")
    
    # Let user select which models to compare
    model_options = [f"{m['category']}: {m['model_name']}" for m in models_for_comparison]
    
    selected = st.multiselect(
        "Select models to compare",
        model_options,
        default=model_options[:min(3, len(model_options))],
        key="comparison_select"
    )
    
    if len(selected) < 2:
        st.warning("Select at least 2 models to compare")
        return
    
    # Build comparison table
    comparison_data = []
    for selection in selected:
        parts = selection.split(': ')
        category = parts[0]
        model_name = parts[1]
        
        result = get_model_result(category, model_name)
        if result:
            metrics = result.get('metrics', {})
            comparison_data.append({
                'Model': model_name,
                'Category': category.replace('_', ' ').title(),
                **{k.replace('_', ' ').title(): f"{v:.4f}" if isinstance(v, float) else str(v) 
                   for k, v in metrics.items() if k != 'primary_metric'}
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Download comparison table
        st.markdown("##### 💾 Download Comparison")
        dl_cols = st.columns(4)
        with dl_cols[0]:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 CSV", data=csv_data, file_name="model_comparison.csv", 
                             mime="text/csv", key="dl_comparison_csv", use_container_width=True)
        with dl_cols[1]:
            try:
                excel_data = dataframe_to_excel(df)
                st.download_button("📥 Excel", data=excel_data, file_name="model_comparison.xlsx",
                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 key="dl_comparison_xlsx", use_container_width=True)
            except:
                pass


def render_run_history(c: Dict):
    """Render the run history with download option"""
    history = get_run_history()
    
    if not history:
        st.info("No run history yet. Run some models to see history here.")
        return
    
    st.markdown("### 📜 Recent Runs")
    
    # Create history dataframe
    history_data = []
    for run in reversed(history[-20:]):
        try:
            dt = datetime.fromisoformat(run['timestamp'])
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            time_str = 'N/A'
        
        history_data.append({
            'Model': run['model_name'],
            'Category': run['category'].replace('_', ' ').title(),
            'Run ID': run.get('run_id', 'N/A'),
            'Timestamp': time_str
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    # Download history
    st.markdown("##### 💾 Download History")
    dl_cols = st.columns(4)
    with dl_cols[0]:
        csv_data = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 CSV", data=csv_data, file_name="run_history.csv", 
                         mime="text/csv", key="dl_history_csv", use_container_width=True)


def generate_full_project_html(c: Dict) -> str:
    """Generate a complete HTML report with all project data, analyses, and graphs"""
    from pages_new.data_manager import get_working_data_info, has_data, get_working_data
    
    # Get all data
    all_results = get_all_run_models()
    data_info = get_working_data_info() if has_data() else {}
    
    # HTML template with beautiful styling
    html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Analytics Project Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
            padding: 60px 40px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.1));
            border-radius: 24px;
            margin-bottom: 40px;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }}
        .header h1 {{
            font-size: 48px;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #22d3ee);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 16px;
        }}
        .header p {{ color: #94a3b8; font-size: 18px; }}
        .header .meta {{ 
            display: flex; 
            justify-content: center; 
            gap: 40px; 
            margin-top: 24px;
            flex-wrap: wrap;
        }}
        .header .meta-item {{
            background: rgba(99, 102, 241, 0.15);
            padding: 12px 24px;
            border-radius: 12px;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }}
        .header .meta-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }}
        .header .meta-value {{ font-size: 20px; font-weight: 700; color: #f8fafc; margin-top: 4px; }}
        
        .section {{
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(51, 65, 85, 0.5);
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 32px;
            backdrop-filter: blur(10px);
        }}
        .section-header {{
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 2px solid rgba(99, 102, 241, 0.3);
        }}
        .section-icon {{ font-size: 32px; }}
        .section-title {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: linear-gradient(145deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.05));
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        }}
        .kpi-label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
        .kpi-value {{
            font-size: 32px;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .kpi-delta {{ font-size: 13px; margin-top: 8px; padding: 4px 12px; border-radius: 20px; display: inline-block; }}
        .kpi-delta.positive {{ background: rgba(16, 185, 129, 0.15); color: #10b981; }}
        .kpi-delta.negative {{ background: rgba(239, 68, 68, 0.15); color: #ef4444; }}
        
        .model-card {{
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(51, 65, 85, 0.5);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .model-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .model-name {{ font-size: 20px; font-weight: 700; color: #f8fafc; }}
        .model-category {{
            background: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .model-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}
        .metric-item {{
            background: rgba(99, 102, 241, 0.08);
            padding: 12px 16px;
            border-radius: 10px;
        }}
        .metric-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; }}
        .metric-value {{ font-size: 18px; font-weight: 700; color: #f8fafc; }}
        
        .chart-container {{
            background: rgba(15, 23, 42, 0.3);
            border-radius: 12px;
            padding: 16px;
            margin-top: 16px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid rgba(51, 65, 85, 0.5);
        }}
        th {{
            background: rgba(99, 102, 241, 0.1);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #a5b4fc;
        }}
        tr:hover {{ background: rgba(99, 102, 241, 0.05); }}
        
        .footer {{
            text-align: center;
            padding: 40px;
            color: #64748b;
            font-size: 14px;
        }}
        .footer a {{ color: #6366f1; text-decoration: none; }}
        
        @media print {{
            body {{ background: white; color: #0f172a; }}
            .section {{ box-shadow: 0 2px 10px rgba(0,0,0,0.1); border: 1px solid #e2e8f0; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Financial Analytics Report</h1>
            <p>Complete project analysis with all models, metrics, and visualizations</p>
            <div class="meta">
                <div class="meta-item">
                    <div class="meta-label">Generated</div>
                    <div class="meta-value">{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Data Source</div>
                    <div class="meta-value">{data_info.get('name', 'N/A')}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Total Models Run</div>
                    <div class="meta-value">{count_run_models()}</div>
                </div>
            </div>
        </div>
'''
    
    # Data Overview Section
    if has_data():
        df = get_working_data()
        if df is not None:
            html_content += f'''
        <div class="section">
            <div class="section-header">
                <span class="section-icon">📈</span>
                <span class="section-title">Data Overview</span>
            </div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Rows</div>
                    <div class="kpi-value">{len(df):,}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Columns</div>
                    <div class="kpi-value">{len(df.columns)}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Assets</div>
                    <div class="kpi-value">{len(data_info.get('tickers', []))}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Date Range</div>
                    <div class="kpi-value" style="font-size: 16px;">{data_info.get('date_range', 'N/A')}</div>
                </div>
            </div>
        </div>
'''
    
    # Helper function to create chart HTML from plot_data
    def create_chart_html(plot_data, category, model_name, chart_id):
        try:
            fig = go.Figure()
            
            if category == 'forecasting':
                if 'historical_values' in plot_data:
                    x_vals = plot_data.get('historical_dates', list(range(len(plot_data['historical_values']))))
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=plot_data['historical_values'],
                        mode='lines', name='Historical',
                        line=dict(color='#94a3b8', width=2)
                    ))
                if 'forecast_values' in plot_data:
                    x_vals = plot_data.get('forecast_dates', list(range(len(plot_data['forecast_values']))))
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=plot_data['forecast_values'],
                        mode='lines', name='Forecast',
                        line=dict(color='#6366f1', width=2, dash='dash')
                    ))
            
            elif category == 'ml_models':
                if 'feature_importance' in plot_data:
                    features = plot_data['feature_importance']
                    fig = go.Figure(go.Bar(
                        x=list(features.values()),
                        y=list(features.keys()),
                        orientation='h',
                        marker_color='#6366f1'
                    ))
                elif 'actual' in plot_data and 'predicted' in plot_data:
                    x_vals = list(range(len(plot_data['actual'])))
                    fig.add_trace(go.Scatter(x=x_vals, y=plot_data['actual'], name='Actual', line=dict(color='#94a3b8')))
                    fig.add_trace(go.Scatter(x=x_vals, y=plot_data['predicted'], name='Predicted', line=dict(color='#6366f1')))
            
            elif category == 'portfolio':
                if 'weights' in plot_data or 'optimized_weights' in plot_data:
                    weights = plot_data.get('optimized_weights', plot_data.get('weights', {}))
                    fig = go.Figure(data=[go.Pie(
                        labels=list(weights.keys()),
                        values=list(weights.values()),
                        hole=0.5,
                        marker=dict(colors=['#6366f1', '#22d3ee', '#f472b6', '#fbbf24', '#34d399'])
                    )])
                elif 'random_returns' in plot_data:
                    fig.add_trace(go.Scatter(
                        x=plot_data.get('random_vol', [])[:100],
                        y=plot_data.get('random_returns', [])[:100],
                        mode='markers', name='Portfolios',
                        marker=dict(size=4, color='#6366f1', opacity=0.4)
                    ))
            
            elif category == 'scenario':
                if 'simulations' in plot_data:
                    for i, sim in enumerate(plot_data['simulations'][:20]):
                        fig.add_trace(go.Scatter(
                            x=list(range(len(sim))), y=sim,
                            mode='lines', line=dict(color='#6366f1', width=0.5),
                            opacity=0.3, showlegend=False
                        ))
                    if 'mean_path' in plot_data:
                        fig.add_trace(go.Scatter(
                            x=list(range(len(plot_data['mean_path']))),
                            y=plot_data['mean_path'],
                            mode='lines', name='Mean',
                            line=dict(color='#f59e0b', width=2)
                        ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,23,42,0.5)',
                font=dict(color='#f8fafc'),
                margin=dict(t=40, l=50, r=30, b=50),
                height=300,
                xaxis=dict(gridcolor='rgba(51,65,85,0.5)'),
                yaxis=dict(gridcolor='rgba(51,65,85,0.5)')
            )
            
            return fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)
        except:
            return '<div style="text-align:center; padding:30px; color:#64748b;">Chart unavailable</div>'
    
    # Model Results Sections with embedded charts
    chart_counter = 0
    for category, models in all_results.items():
        if models:
            cat_results = get_category_results(category)
            icons = {'forecasting': '🔮', 'ml_models': '🧪', 'portfolio': '💼', 'scenario': '📈', 'risk': '🛡️'}
            icon = icons.get(category, '📊')
            
            html_content += f'''
        <div class="section">
            <div class="section-header">
                <span class="section-icon">{icon}</span>
                <span class="section-title">{category.replace('_', ' ').title()} Analysis</span>
            </div>
'''
            
            for model_name, result in cat_results.items():
                metrics = result.get('metrics', {})
                plot_data = result.get('plot_data', {})
                timestamp = result.get('timestamp', '')
                
                html_content += f'''
            <div class="model-card">
                <div class="model-header">
                    <span class="model-name">📊 {model_name}</span>
                    <span class="model-category">{category.replace('_', ' ').title()}</span>
                </div>
                <div class="model-metrics">
'''
                for key, value in metrics.items():
                    if key != 'primary_metric':
                        display_val = f"{value:.4f}" if isinstance(value, float) and abs(value) < 10 else str(value)
                        html_content += f'''
                    <div class="metric-item">
                        <div class="metric-label">{key.replace('_', ' ').title()}</div>
                        <div class="metric-value">{display_val}</div>
                    </div>
'''
                
                html_content += '''
                </div>
'''
                
                # Add chart if plot_data exists
                if plot_data:
                    chart_id = f"dashboard_chart_{chart_counter}"
                    chart_html = create_chart_html(plot_data, category, model_name, chart_id)
                    html_content += f'''
                <div class="chart-container" style="margin-top: 20px;">
                    <div style="font-size: 14px; font-weight: 600; color: #94a3b8; margin-bottom: 12px;">📈 Visualization</div>
                    {chart_html}
                </div>
'''
                    chart_counter += 1
                
                html_content += '''
            </div>
'''
            
            html_content += '''
        </div>
'''
    
    # Summary Table
    html_content += '''
        <div class="section">
            <div class="section-header">
                <span class="section-icon">📋</span>
                <span class="section-title">Summary Table</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Model</th>
                        <th>Primary Metric</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
'''
    
    for category, models in all_results.items():
        cat_results = get_category_results(category)
        for model_name, result in cat_results.items():
            metrics = result.get('metrics', {})
            primary = metrics.get('primary_metric', 'N/A')
            if isinstance(primary, float):
                primary = f"{primary:.4f}"
            html_content += f'''
                    <tr>
                        <td>{category.replace('_', ' ').title()}</td>
                        <td>{model_name}</td>
                        <td>{primary}</td>
                        <td><span style="color: #10b981;">✓ Complete</span></td>
                    </tr>
'''
    
    html_content += '''
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Generated by <strong>Financial Analytics Suite</strong></p>
            <p style="margin-top: 8px;">Enterprise-grade financial analytics platform</p>
        </div>
    </div>
</body>
</html>
'''
    
    return html_content


def render_full_dashboard_export(c: Dict):
    """Render export options for the full dashboard including HTML project export"""
    st.markdown("### 📥 Export Complete Project")
    st.markdown(f"<p style='color: {c['text_secondary']}'>Download your entire analysis project including all models, metrics, and visualizations</p>", unsafe_allow_html=True)
    
    all_results = get_all_run_models()
    
    if not all_results:
        st.info("No results to export. Run some models first.")
        return
    
    # Collect all data
    export_data = {
        'export_date': datetime.now().isoformat(),
        'models': {}
    }
    
    all_metrics = []
    
    for category, models in all_results.items():
        cat_results = get_category_results(category)
        for model_name, result in cat_results.items():
            metrics = result.get('metrics', {})
            all_metrics.append({
                'Category': category.replace('_', ' ').title(),
                'Model': model_name,
                **{k.replace('_', ' ').title(): v for k, v in metrics.items() if k != 'primary_metric'}
            })
            export_data['models'][f"{category}_{model_name}"] = {
                'metrics': metrics,
                'parameters': result.get('parameters', {}),
                'timestamp': result.get('timestamp', '')
            }
    
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        
        # Full Project HTML Export (Main Feature)
        st.markdown("#### 🌐 Full Project HTML Report")
        st.info("📄 Download a complete, beautifully styled HTML report with all your analysis results, metrics, and summary tables.")
        
        html_report = generate_full_project_html(c)
        st.download_button(
            "📥 Download Complete Project (HTML)",
            data=html_report,
            file_name=f"financial_analytics_project_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            key="dl_full_project_html",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("#### 📊 Individual Export Options")
        
        export_cols = st.columns(3)
        
        with export_cols[0]:
            csv_data = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Summary (CSV)",
                data=csv_data,
                file_name="dashboard_summary.csv",
                mime="text/csv",
                key="dl_dashboard_csv",
                use_container_width=True
            )
        
        with export_cols[1]:
            try:
                excel_data = dataframe_to_excel(summary_df)
                st.download_button(
                    "📥 Summary (Excel)",
                    data=excel_data,
                    file_name="dashboard_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_dashboard_xlsx",
                    use_container_width=True
                )
            except:
                st.button("📥 Excel (N/A)", disabled=True, use_container_width=True)
        
        with export_cols[2]:
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                "📥 Full Data (JSON)",
                data=json_str,
                file_name="dashboard_full_export.json",
                mime="application/json",
                key="dl_dashboard_json",
                use_container_width=True
            )


def render():
    """Render the Dashboard Builder page"""
    c = get_theme_colors()
    inject_premium_styles()
    
    # Header
    st.title("📊 Dashboard Builder")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 Dashboard")
        st.markdown(f"<p style='color: {c['text_secondary']};'>View results from your analysis runs - only shows models you have executed</p>", unsafe_allow_html=True)
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, key="refresh_dash"):
            st.rerun()
    
    st.markdown("---")
    
    # Check for data first
    if not has_data():
        st.warning("⚠️ No data uploaded. Please upload your data in Data Sources first.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📥 Go to Data Sources", type="primary", use_container_width=True, key="goto_data"):
                st.session_state.current_page = 'data_sources'
                st.rerun()
        return
    
    # Get dashboard data (only from run models)
    dashboard_data = get_dashboard_data()
    
    # Show status bar
    total_runs = count_run_models()
    run_models = get_all_run_models()
    
    # Status bar
    st.markdown(f'''
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 12px; padding: 16px 20px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 14px; font-weight: 600; color: {c['text']};">Analysis Status</span>
            </div>
            <div style="display: flex; gap: 24px;">
                <div>
                    <span style="font-size: 24px; font-weight: 700; color: {c['primary']};">{total_runs}</span>
                    <span style="font-size: 12px; color: {c['text_muted']}; margin-left: 4px;">Models Run</span>
                </div>
                <div>
                    <span style="font-size: 24px; font-weight: 700; color: {c['success']};">{len(run_models)}</span>
                    <span style="font-size: 12px; color: {c['text_muted']}; margin-left: 4px;">Categories</span>
                </div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # If no models have been run
    if not dashboard_data['has_results']:
        render_no_results_message(c)
        return
    
    # Main dashboard content - only showing RUN models
    st.markdown("## 🎯 Your Analysis Results")
    st.caption("Showing only models you have executed. Each result includes download options.")
    
    # Import PowerBI dashboard
    try:
        from pages_new.powerbi_dashboard import render_powerbi_dashboard
        has_powerbi = True
    except ImportError:
        has_powerbi = False
    
    # Tabs for different views - PowerBI Dashboard as first tab
    if has_powerbi:
        tab0, tab1, tab2, tab3, tab4 = st.tabs(["🎯 Executive Dashboard", "📈 Results Gallery", "📊 Comparison", "📜 History", "📥 Export All"])
        
        with tab0:
            render_powerbi_dashboard()
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Results Gallery", "📊 Comparison", "📜 History", "📥 Export All"])
    
    with tab1:
        # Display all model results with plots and download options
        index = 0
        for category, models in dashboard_data['categories'].items():
            if models:
                cat_results = get_category_results(category)
                
                for model_data in models:
                    model_name = model_data.get('model_name', 'Unknown')
                    full_result = cat_results.get(model_name, {})
                    
                    if full_result:
                        render_model_result_section(category, model_name, full_result, c, index)
                        index += 1
    
    with tab2:
        render_model_comparison(c)
    
    with tab3:
        render_run_history(c)
    
    with tab4:
        render_full_dashboard_export(c)


if __name__ == "__main__":
    render()
