"""
Financial Analytics Suite - Projects Page
Full project management with creation, editing, and organization
"""

import streamlit as st
from typing import Dict, List
from datetime import datetime
import json
import os


def get_theme_colors() -> Dict[str, str]:
    """Get current theme colors"""
    theme_mode = st.session_state.get('theme_mode', 'dark')
    if theme_mode == 'dark':
        return {
            'text': '#f8fafc', 'text_muted': '#64748b', 'text_secondary': '#94a3b8',
            'primary': '#6366f1', 'secondary': '#8b5cf6', 'accent': '#22d3ee',
            'success': '#10b981', 'warning': '#f59e0b', 'error': '#ef4444',
            'glass_bg': 'rgba(30, 41, 59, 0.7)', 'glass_border': 'rgba(148, 163, 184, 0.1)',
            'bg_card': '#1e293b', 'border': '#334155',
            'gradient': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)',
        }
    else:
        return {
            'text': '#0f172a', 'text_muted': '#64748b', 'text_secondary': '#475569',
            'primary': '#4f46e5', 'secondary': '#7c3aed', 'accent': '#0891b2',
            'success': '#059669', 'warning': '#d97706', 'error': '#dc2626',
            'glass_bg': 'rgba(255, 255, 255, 0.8)', 'glass_border': 'rgba(0, 0, 0, 0.06)',
            'bg_card': '#ffffff', 'border': '#e2e8f0',
            'gradient': 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%)',
        }


def init_projects():
    """Initialize projects in session state - start empty"""
    if 'projects' not in st.session_state:
        st.session_state.projects = []  # Start with empty - no demo projects
    
    if 'show_create_project' not in st.session_state:
        st.session_state.show_create_project = False
    
    if 'active_project' not in st.session_state:
        st.session_state.active_project = None
    
    if 'editing_project' not in st.session_state:
        st.session_state.editing_project = None


def auto_save_analysis(analysis_name: str, analysis_type: str, data_source: str = None):
    """Auto-save an analysis as a project entry"""
    from pages_new.data_manager import get_working_data_info, has_data
    
    init_projects()
    
    # Get data info
    if has_data():
        data_info = get_working_data_info()
        data_source = data_info.get('name', 'Unknown')
    
    # Create project entry
    project_id = f"auto-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    project = {
        'id': project_id,
        'name': f"{analysis_type}: {analysis_name}",
        'description': f"Auto-saved {analysis_type} analysis",
        'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'updated': 'Just now',
        'owner': 'You',
        'status': 'completed',
        'tags': [analysis_type],
        'data_sources': [data_source] if data_source else [],
        'analyses': [analysis_name]
    }
    
    st.session_state.projects.append(project)
    return project_id


def save_project(project: Dict):
    """Save a project to session state"""
    projects = st.session_state.projects
    
    # Check if updating existing project
    for i, p in enumerate(projects):
        if p['id'] == project['id']:
            projects[i] = project
            st.session_state.projects = projects
            return
    
    # Add new project
    st.session_state.projects.append(project)


def delete_project(project_id: str):
    """Delete a project"""
    st.session_state.projects = [p for p in st.session_state.projects if p['id'] != project_id]


def render_project_card(project: Dict, c: Dict):
    """Render a project card"""
    status_colors = {
        'active': c['success'],
        'draft': c['warning'],
        'completed': c['primary'],
        'archived': c['text_muted']
    }
    status_color = status_colors.get(project['status'], c['text_muted'])
    
    # Convert hex colors to rgba for backgrounds
    primary_hex = c['primary'].lstrip('#')
    primary_rgb = tuple(int(primary_hex[i:i+2], 16) for i in (0, 2, 4))
    tag_bg = f'rgba({primary_rgb[0]}, {primary_rgb[1]}, {primary_rgb[2]}, 0.12)'
    
    status_hex = status_color.lstrip('#')
    status_rgb = tuple(int(status_hex[i:i+2], 16) for i in (0, 2, 4))
    status_bg = f'rgba({status_rgb[0]}, {status_rgb[1]}, {status_rgb[2]}, 0.12)'
    
    tags_html = ''.join([f'<span style="background: {tag_bg}; color: {c["primary"]}; padding: 3px 10px; border-radius: 6px; font-size: 10px; font-weight: 600; margin-right: 6px;">{tag}</span>' for tag in project.get('tags', [])])
    
    desc = project['description'][:100] + ('...' if len(project['description']) > 100 else '')
    
    card_html = f'<div style="background: {c["glass_bg"]}; backdrop-filter: blur(16px); border: 1px solid {c["glass_border"]}; border-radius: 16px; padding: 24px; height: 100%; position: relative; overflow: hidden;"><div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: {c["gradient"]};"></div><div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;"><span style="background: {status_bg}; color: {status_color}; padding: 4px 12px; border-radius: 20px; font-size: 10px; font-weight: 600; text-transform: uppercase;">{project["status"]}</span><span style="color: {c["text_muted"]}; font-size: 11px;">📅 {project["updated"]}</span></div><h3 style="color: {c["text"]}; font-size: 18px; font-weight: 700; margin: 0 0 8px 0;">{project["name"]}</h3><p style="color: {c["text_secondary"]}; font-size: 13px; line-height: 1.5; margin: 0 0 16px 0; min-height: 40px;">{desc}</p><div style="margin-bottom: 16px;">{tags_html}</div><div style="display: flex; gap: 8px; font-size: 11px; color: {c["text_muted"]};"><span>📊 {len(project.get("data_sources", []))} sources</span><span>•</span><span>🔬 {len(project.get("analyses", []))} analyses</span></div></div>'
    st.markdown(card_html, unsafe_allow_html=True)


def render_create_project_form(c: Dict):
    """Render the project creation wizard"""
    st.markdown(f'''
    <div style="background: {c['glass_bg']}; backdrop-filter: blur(20px); border: 1px solid {c['glass_border']}; border-radius: 20px; padding: 32px; margin-bottom: 24px;">
        <h2 style="color: {c['text']}; font-size: 24px; font-weight: 700; margin: 0 0 8px 0;">✨ Create New Project</h2>
        <p style="color: {c['text_secondary']}; font-size: 14px; margin: 0;">Set up a new analysis workspace for your financial data</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Project Details
    col1, col2 = st.columns(2)
    
    with col1:
        project_name = st.text_input(
            "Project Name *",
            placeholder="e.g., Portfolio Analysis Q1 2026",
            help="Give your project a descriptive name"
        )
        
        project_type = st.selectbox(
            "Project Type",
            ["Portfolio Analysis", "Time Series Forecasting", "Risk Assessment", "ML Prediction", "Custom Analysis"]
        )
        
        tags = st.multiselect(
            "Tags",
            ["Portfolio", "Risk", "Forecasting", "ML", "Stocks", "Bonds", "Crypto", "Custom"],
            default=[]
        )
    
    with col2:
        project_description = st.text_area(
            "Description",
            placeholder="Describe what you want to analyze...",
            height=100
        )
        
        data_source = st.selectbox(
            "Initial Data Source",
            ["None - Add Later", "Upload New File", "Use Demo Data"],
            help="You can add more data sources after creating the project"
        )
    
    st.markdown("---")
    
    # Analysis Configuration
    st.markdown(f"<h4 style='color: {c['text']}; margin-bottom: 16px;'>🔬 Configure Initial Analysis</h4>", unsafe_allow_html=True)
    
    analysis_cols = st.columns(4)
    
    with analysis_cols[0]:
        include_portfolio = st.checkbox("💼 Portfolio Optimization", value=project_type == "Portfolio Analysis")
    with analysis_cols[1]:
        include_forecast = st.checkbox("🔮 Time Series Forecast", value=project_type == "Time Series Forecasting")
    with analysis_cols[2]:
        include_risk = st.checkbox("🛡️ Risk Analysis", value=project_type == "Risk Assessment")
    with analysis_cols[3]:
        include_ml = st.checkbox("🧠 ML Predictions", value=project_type == "ML Prediction")
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # File Upload (if selected)
    uploaded_file = None
    if data_source == "Upload New File":
        st.markdown(f"<h4 style='color: {c['text']}; margin-bottom: 16px;'>📁 Upload Your Data</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your financial data file"
        )
        
        if uploaded_file:
            st.success(f"✅ File ready: {uploaded_file.name}")
            
            # Preview the data
            import pandas as pd
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.markdown("**Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                st.caption(f"📊 {len(df)} rows × {len(df.columns)} columns")
                
                # Save to session state for use in analysis
                st.session_state.uploaded_data = df
                st.session_state.uploaded_filename = uploaded_file.name
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    st.markdown("---")
    
    # Buttons
    btn_cols = st.columns([1, 1, 4])
    
    with btn_cols[0]:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.show_create_project = False
            st.rerun()
    
    with btn_cols[1]:
        create_clicked = st.button("✅ Create Project", type="primary", use_container_width=True)
    
    if create_clicked:
        if not project_name:
            st.error("⚠️ Please enter a project name")
        else:
            # Build analyses list
            analyses = []
            if include_portfolio:
                analyses.append("Portfolio Optimization")
            if include_forecast:
                analyses.append("Time Series Forecast")
            if include_risk:
                analyses.append("Risk Analysis")
            if include_ml:
                analyses.append("ML Predictions")
            
            # Build data sources list
            data_sources = []
            if uploaded_file:
                data_sources.append(uploaded_file.name)
            elif data_source == "Use Demo Data":
                data_sources.append("Demo_Financial_Data.csv")
            
            # Create the project
            new_project = {
                'id': f"proj-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'name': project_name,
                'description': project_description or f"{project_type} project",
                'created': datetime.now().strftime('%Y-%m-%d'),
                'updated': 'Just now',
                'owner': 'You',
                'status': 'active',
                'tags': tags if tags else [project_type.split()[0]],
                'data_sources': data_sources,
                'analyses': analyses,
                'type': project_type
            }
            
            save_project(new_project)
            st.session_state.show_create_project = False
            st.session_state.active_project = new_project['id']
            st.success(f"🎉 Project '{project_name}' created successfully!")
            st.balloons()
            st.rerun()


def render_project_view(project: Dict, c: Dict):
    """Render the project workspace view with run analyses and HTML export"""
    from pages_new.model_results_manager import get_all_run_models, get_category_results, count_run_models
    
    # Header
    st.markdown(f'''
    <div style="background: {c['gradient']}; border-radius: 20px; padding: 32px; margin-bottom: 24px; position: relative; overflow: hidden;">
        <div style="position: relative; z-index: 1;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h1 style="color: white; font-size: 32px; font-weight: 800; margin: 0 0 8px 0;">{project['name']}</h1>
                    <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin: 0;">{project['description']}</p>
                </div>
                <span style="background: rgba(255,255,255,0.2); color: white; padding: 6px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; text-transform: uppercase;">{project['status']}</span>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.active_project = None
            st.rerun()
    
    with col2:
        if st.button("📊 Add Data", use_container_width=True):
            st.session_state.current_page = 'data_sources'
            st.rerun()
    
    with col3:
        if st.button("🔬 Analysis", use_container_width=True):
            st.session_state.current_page = 'forecasting'
            st.rerun()
    
    with col4:
        if st.button("📈 Dashboard", use_container_width=True):
            st.session_state.current_page = 'dashboard_builder'
            st.rerun()
    
    with col5:
        if st.button("📄 Report", use_container_width=True):
            st.session_state.current_page = 'reports'
            st.rerun()
    
    st.markdown("---")
    
    # Project Details
    detail_cols = st.columns(3)
    
    with detail_cols[0]:
        st.markdown(f"**📅 Created:** {project['created']}")
        st.markdown(f"**🔄 Last Updated:** {project['updated']}")
    
    with detail_cols[1]:
        st.markdown(f"**👤 Owner:** {project['owner']}")
        st.markdown(f"**📊 Type:** {project.get('type', 'Custom Analysis')}")
    
    with detail_cols[2]:
        tags_str = ", ".join(project.get('tags', []))
        st.markdown(f"**🏷️ Tags:** {tags_str}")
    
    st.markdown("---")
    
    # Run Analyses from model_results_manager
    st.subheader("🔬 Run Analyses Results")
    
    all_results = get_all_run_models()
    total_runs = count_run_models()
    
    if total_runs > 0:
        # Summary cards
        st.markdown(f'''
        <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 16px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <div style="font-size: 32px; font-weight: 800; color: {c['primary']};">{total_runs}</div>
                    <div style="font-size: 12px; color: {c['text_muted']};">Total Models Run</div>
                </div>
                <div>
                    <div style="font-size: 32px; font-weight: 800; color: {c['success']};">{len(all_results)}</div>
                    <div style="font-size: 12px; color: {c['text_muted']};">Categories</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # List all run models by category
        for category, models in all_results.items():
            if models:
                cat_results = get_category_results(category)
                icons = {'forecasting': '🔮', 'ml_models': '🧪', 'portfolio': '💼', 'scenario': '📈', 'risk': '🛡️'}
                icon = icons.get(category, '📊')
                
                st.markdown(f"#### {icon} {category.replace('_', ' ').title()}")
                
                for model_name, result in cat_results.items():
                    metrics = result.get('metrics', {})
                    timestamp = result.get('timestamp', '')
                    
                    with st.expander(f"📊 {model_name}", expanded=False):
                        metric_cols = st.columns(min(len(metrics), 4))
                        for i, (key, value) in enumerate(list(metrics.items())[:4]):
                            if key != 'primary_metric':
                                with metric_cols[i % len(metric_cols)]:
                                    display_val = f"{value:.4f}" if isinstance(value, float) and abs(value) < 10 else str(value)
                                    st.metric(key.replace('_', ' ').title(), display_val)
        
        # HTML Export for Project
        st.markdown("---")
        st.subheader("📥 Export Project")
        
        def generate_project_html():
            import plotly.graph_objects as go
            
            # Helper function to create a chart from plot_data
            def create_chart_html(plot_data, category, model_name, chart_id):
                try:
                    fig = go.Figure()
                    
                    if category == 'forecasting':
                        # Forecasting chart
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
                        fig.update_layout(title=f"{model_name} Forecast")
                    
                    elif category == 'ml_models':
                        # ML chart - feature importance or actual vs predicted
                        if 'feature_importance' in plot_data:
                            features = plot_data['feature_importance']
                            fig = go.Figure(go.Bar(
                                x=list(features.values()),
                                y=list(features.keys()),
                                orientation='h',
                                marker_color='#6366f1'
                            ))
                            fig.update_layout(title=f"{model_name} - Feature Importance")
                        elif 'actual' in plot_data and 'predicted' in plot_data:
                            x_vals = list(range(len(plot_data['actual'])))
                            fig.add_trace(go.Scatter(x=x_vals, y=plot_data['actual'], name='Actual', line=dict(color='#94a3b8')))
                            fig.add_trace(go.Scatter(x=x_vals, y=plot_data['predicted'], name='Predicted', line=dict(color='#6366f1')))
                            fig.update_layout(title=f"{model_name} - Actual vs Predicted")
                    
                    elif category == 'portfolio':
                        # Portfolio chart - weights or efficient frontier
                        if 'weights' in plot_data or 'optimized_weights' in plot_data:
                            weights = plot_data.get('optimized_weights', plot_data.get('weights', {}))
                            fig = go.Figure(data=[go.Pie(
                                labels=list(weights.keys()),
                                values=list(weights.values()),
                                hole=0.5,
                                marker=dict(colors=['#6366f1', '#22d3ee', '#f472b6', '#fbbf24', '#34d399'])
                            )])
                            fig.update_layout(title=f"{model_name} - Portfolio Allocation")
                        elif 'random_returns' in plot_data:
                            fig.add_trace(go.Scatter(
                                x=plot_data.get('random_vol', [])[:100],
                                y=plot_data.get('random_returns', [])[:100],
                                mode='markers', name='Portfolios',
                                marker=dict(size=4, color='#6366f1', opacity=0.4)
                            ))
                            fig.update_layout(title=f"{model_name} - Efficient Frontier", 
                                            xaxis_title='Volatility', yaxis_title='Return')
                    
                    elif category == 'scenario':
                        # Scenario chart - Monte Carlo or stress test
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
                            fig.update_layout(title=f"{model_name} - Monte Carlo Simulation")
                    
                    # Common layout settings
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(15,23,42,0.5)',
                        font=dict(color='#f8fafc'),
                        margin=dict(t=50, l=50, r=30, b=50),
                        height=350,
                        xaxis=dict(gridcolor='rgba(51,65,85,0.5)'),
                        yaxis=dict(gridcolor='rgba(51,65,85,0.5)')
                    )
                    
                    # Convert to HTML div
                    chart_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=chart_id)
                    return chart_html
                except Exception as e:
                    return f'<div style="text-align:center; padding:40px; color:#64748b;">Chart unavailable</div>'
            
            # ═══════════════════════════════════════════════════════════════
            # FULL PROJECT REPORT GENERATION (Combines Executive Style + Detailed Analysis)
            # ═══════════════════════════════════════════════════════════════
            
            # Get Executive Dashboard data/insights for the summary section
            from pages_new.data_manager import get_working_data, has_data
            from pages_new.powerbi_dashboard import calculate_business_insights
            
            df = get_working_data() if has_data() else None
            insights = calculate_business_insights(df) if df is not None and not df.empty else {}
            
            # Use the premium styling from the Executive Dashboard
            signal_color = '#10b981' if insights.get('recommendation') in ['Strong Buy', 'Buy'] else '#ef4444'
            
            html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Full Project Analysis: {project['name']}</title>
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
        
        /* HEADER STYLES */
        .header {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.1));
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 24px;
            padding: 40px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 42px; font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #22d3ee);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        /* EXECUTIVE SUMMARY BANNER */
        .signal-banner {{
            background: linear-gradient(90deg, {signal_color}25, {signal_color}10, transparent);
            border-left: 4px solid {signal_color};
            border-radius: 16px; 
            padding: 25px 30px; margin-bottom: 30px;
            display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;
        }}
        .signal-main {{ font-size: 32px; font-weight: 800; color: {signal_color}; }}
        .signal-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
        .signal-stat-value {{ font-size: 24px; font-weight: 700; color: #f8fafc; }}
        
        /* KPI GRID */
        .kpi-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 20px; margin-bottom: 40px;
        }}
        .kpi-card {{
            background: #1e293b; border: 1px solid #334155; border-radius: 16px;
            padding: 20px; text-align: center; border-top: 3px solid #6366f1;
        }}
        .kpi-value {{ font-size: 22px; font-weight: 800; color: #f8fafc; }}
        
        /* ANALYSIS SECTIONS */
        .section {{
            background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(51, 65, 85, 0.5);
            border-radius: 20px; padding: 30px; margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 24px; font-weight: 700; color: #f8fafc;
            margin-bottom: 25px; padding-bottom: 15px;
            border-bottom: 2px solid rgba(99, 102, 241, 0.3);
            display: flex; align-items: center; gap: 10px;
        }}
        
        /* MODEL RESULTS */
        .model-card {{
            background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(51, 65, 85, 0.5);
            border-radius: 16px; padding: 25px; margin-bottom: 30px;
        }}
        .model-header {{
            display: flex; justify-content: space-between; margin-bottom: 20px;
        }}
        .model-name {{ font-size: 20px; font-weight: 700; color: #f8fafc; }}
        
        .metrics-grid {{
            display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px; margin-bottom: 25px;
        }}
        .metric-box {{
            background: rgba(99, 102, 241, 0.1); padding: 15px; border-radius: 12px;
        }}
        .metric-label {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; }}
        .metric-val {{ font-size: 18px; font-weight: 700; color: #f8fafc; }}
        
        .footer {{
            text-align: center; padding: 40px; color: #64748b; border-top: 1px solid #334155;
        }}
    </style>
</head>
<body>
    <div class="container">
        
        <!-- HEADER -->
        <div class="header">
            <h1>📊 {project['name']}</h1>
            <p>Full Project Analysis Report</p>
            <p style="margin-top: 10px; font-size: 14px; color: #94a3b8;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        
        <!-- EXECUTIVE SUMMARY SECTION -->
        <div class="section">
            <div class="section-title">⚡ Executive Summary</div>
            
            <div class="signal-banner">
                <div>
                    <div class="signal-label">Overall Signal</div>
                    <div class="signal-main">{insights.get('recommendation', 'Neutral')}</div>
                </div>
                <div style="display: flex; gap: 30px;">
                    <div>
                        <div class="signal-label">Confidence</div>
                        <div class="signal-stat-value">{insights.get('confidence', 0)}%</div>
                    </div>
                    <div>
                        <div class="signal-label">Risk Check</div>
                        <div class="signal-stat-value">{insights.get('risk_level', 'Medium')}</div>
                    </div>
                </div>
            </div>
            
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-value">{insights.get('total_return', 0):.1f}%</div>
                    <div class="signal-label">Total Return</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{insights.get('volatility', 0):.1f}%</div>
                    <div class="signal-label">Volatility</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{insights.get('sharpe_ratio', 0):.2f}</div>
                    <div class="signal-label">Sharpe Ratio</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-value">{total_runs}</div>
                    <div class="signal-label">Models Analyzed</div>
                </div>
            </div>
        </div>
'''
            
            # DETAILED ANALYSIS SECTION (Iterate through all results)
            chart_counter = 0
            for category, models in all_results.items():
                if models:
                    cat_results = get_category_results(category)
                    icons = {'forecasting': '🔮', 'ml_models': '🧪', 'portfolio': '💼', 'scenario': '📈', 'risk': '🛡️'}
                    icon = icons.get(category, '📊')
                    
                    html += f'''
        <div class="section">
            <div class="section-title">{icon} {category.replace('_', ' ').title()} Analysis</div>
'''
                    for model_name, result in cat_results.items():
                        metrics = result.get('metrics', {})
                        plot_data = result.get('plot_data', {})
                        
                        html += f'''
            <div class="model-card">
                <div class="model-header">
                    <div class="model-name">{model_name}</div>
                </div>
                
                <div class="metrics-grid">
'''
                        # Add Metrics
                        for key, value in metrics.items():
                            if key != 'primary_metric':
                                display_val = f"{value:.4f}" if isinstance(value, float) and abs(value) < 10 else str(value)
                                html += f'''
                    <div class="metric-box">
                        <div class="metric-label">{key.replace('_', ' ').title()}</div>
                        <div class="metric-val">{display_val}</div>
                    </div>
'''
                        html += '''
                </div>
'''
                        # Add Chart if available
                        if plot_data:
                            chart_id = f"chart_{chart_counter}"
                            chart_html = create_chart_html(plot_data, category, model_name, chart_id)
                            html += f'''
                <div style="background: rgba(15, 23, 42, 0.4); border-radius: 12px; padding: 20px; border: 1px solid rgba(51, 65, 85, 0.3);">
                    <div style="font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 15px;">VISUALIZATION</div>
                    {chart_html}
                </div>
'''
                            chart_counter += 1
                        
                        html += '''
            </div>
'''
                    html += '''
        </div>
'''
            
            html += '''
        <div class="footer">
            <p style="font-size: 16px; font-weight: 600; color: #f8fafc;">💎 Financial Analytics Suite</p>
            <p>Generated by AI-Powered Investment Engine</p>
        </div>
    </div>
</body>
</html>
'''
            return html
        
        # Primary Export - PowerBI Executive Dashboard
        st.markdown(f'''
        <div style="
            background: linear-gradient(135deg, {c['primary']}20, {c['secondary']}10);
            border: 1px solid {c['primary']}40;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
        ">
            <div style="font-size: 18px; font-weight: 700; color: {c['text']}; margin-bottom: 8px;">
                🎯 Executive Dashboard Export
            </div>
            <div style="font-size: 13px; color: {c['text_secondary']};">
                Download a stunning PowerBI-style dashboard with all your analyses, KPIs, and visualizations in one comprehensive view.
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Try to import and generate PowerBI dashboard
        try:
            from pages_new.powerbi_dashboard import generate_executive_html
            powerbi_html = generate_executive_html(c)
            st.download_button(
                "🎯 Download Executive Dashboard (PowerBI Style)",
                data=powerbi_html,
                file_name=f"executive_dashboard_{project['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                key="dl_powerbi_dashboard",
                use_container_width=True
            )
        except ImportError:
            st.info("Executive Dashboard not available. Using standard project report.")
        
        st.markdown("---")
        st.markdown("##### 📄 Additional Export Options")
        
        export_cols = st.columns(2)
        with export_cols[0]:
            html_report = generate_project_html()
            st.download_button(
                "📥 Download Project Report (HTML)",
                data=html_report,
                file_name=f"project_{project['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
                key="dl_project_html",
                use_container_width=True
            )
        
        with export_cols[1]:
            # JSON export
            import json
            export_data = {
                'project': project,
                'analyses': {},
                'export_date': datetime.now().isoformat()
            }
            for category, models in all_results.items():
                cat_results = get_category_results(category)
                export_data['analyses'][category] = {
                    name: {'metrics': r.get('metrics', {})} for name, r in cat_results.items()
                }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                "📥 Download Project Data (JSON)",
                data=json_str,
                file_name=f"project_{project['name'].replace(' ', '_')}.json",
                mime="application/json",
                key="dl_project_json",
                use_container_width=True
            )
    else:
        st.info("📭 No analyses have been run yet. Go to Forecasting, ML Lab, Portfolio, or Scenario pages to run analyses.")
        
        # Quick action buttons
        st.markdown("**Run your first analysis:**")
        quick_cols = st.columns(4)
        with quick_cols[0]:
            if st.button("🔮 Forecasting", key="quick_forecast", use_container_width=True):
                st.session_state.current_page = 'forecasting'
                st.rerun()
        with quick_cols[1]:
            if st.button("🧪 ML Lab", key="quick_ml", use_container_width=True):
                st.session_state.current_page = 'ml_lab'
                st.rerun()
        with quick_cols[2]:
            if st.button("💼 Portfolio", key="quick_portfolio", use_container_width=True):
                st.session_state.current_page = 'portfolio'
                st.rerun()
        with quick_cols[3]:
            if st.button("📈 Scenario", key="quick_scenario", use_container_width=True):
                st.session_state.current_page = 'scenario'
                st.rerun()
    
    st.markdown("---")
    
    # Data Sources Section
    st.subheader("📁 Data Sources")
    
    if project.get('data_sources'):
        success_hex = c['success'].lstrip('#')
        success_rgb = tuple(int(success_hex[i:i+2], 16) for i in (0, 2, 4))
        success_bg = f'rgba({success_rgb[0]}, {success_rgb[1]}, {success_rgb[2]}, 0.12)'
        
        for source in project['data_sources']:
            source_html = f'<div style="background: {c["bg_card"]}; border: 1px solid {c["border"]}; border-radius: 12px; padding: 16px; margin-bottom: 12px; display: flex; align-items: center; gap: 12px;"><span style="font-size: 24px;">📄</span><div style="flex: 1;"><div style="color: {c["text"]}; font-weight: 600;">{source}</div><div style="color: {c["text_muted"]}; font-size: 12px;">Ready for analysis</div></div><span style="background: {success_bg}; color: {c["success"]}; padding: 4px 12px; border-radius: 20px; font-size: 11px;">Connected</span></div>'
            st.markdown(source_html, unsafe_allow_html=True)
    else:
        st.info("📭 No data sources added yet. Click 'Add Data' to import your data.")


def render():
    """Render the Projects page"""
    c = get_theme_colors()
    init_projects()
    
    # Check if viewing a specific project
    if st.session_state.active_project:
        project = next((p for p in st.session_state.projects if p['id'] == st.session_state.active_project), None)
        if project:
            render_project_view(project, c)
            return
    
    # Check if creating a new project
    if st.session_state.show_create_project:
        render_create_project_form(c)
        return
    
    # Main Projects List View
    st.title("📁 Projects & Workspaces")
    st.markdown(f"<p style='color: {c['text_secondary']};'>Organize your financial analyses into projects</p>", unsafe_allow_html=True)
    
    # Stats
    stat_cols = st.columns(4)
    total_projects = len(st.session_state.projects)
    active_projects = len([p for p in st.session_state.projects if p['status'] == 'active'])
    
    with stat_cols[0]:
        st.metric("Total Projects", total_projects)
    with stat_cols[1]:
        st.metric("Active", active_projects)
    with stat_cols[2]:
        st.metric("Data Sources", sum(len(p.get('data_sources', [])) for p in st.session_state.projects))
    with stat_cols[3]:
        st.metric("Analyses", sum(len(p.get('analyses', [])) for p in st.session_state.projects))
    
    st.markdown("---")
    
    # Create New Project Button
    if st.button("➕ Create New Project", type="primary", use_container_width=False):
        st.session_state.show_create_project = True
        st.rerun()
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    # Project Cards
    if st.session_state.projects:
        cols = st.columns(3)
        
        for i, project in enumerate(st.session_state.projects):
            with cols[i % 3]:
                render_project_card(project, c)
                
                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("📂 Open", key=f"open_{project['id']}", use_container_width=True):
                        st.session_state.active_project = project['id']
                        st.rerun()
                with btn_cols[1]:
                    if st.button("🗑️", key=f"delete_{project['id']}", use_container_width=True):
                        delete_project(project['id'])
                        st.rerun()
                
                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    else:
        st.info("📭 No projects yet. Create your first project to get started!")


if __name__ == "__main__":
    render()
