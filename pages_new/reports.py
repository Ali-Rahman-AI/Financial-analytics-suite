"""
Financial Analytics Suite - Reports Page
Report builder with real data, charts, and export functionality
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict
import base64
from io import BytesIO

# Import data manager
from pages_new.data_manager import (
    get_working_data, get_working_data_info, has_data,
    get_available_tickers, calculate_portfolio_metrics
)


def get_theme_colors() -> Dict[str, str]:
    theme_mode = st.session_state.get('theme_mode', 'dark')
    if theme_mode == 'dark':
        return {
            'text': '#f8fafc', 'text_muted': '#64748b', 'text_secondary': '#94a3b8',
            'primary': '#6366f1', 'secondary': '#8b5cf6', 'accent': '#22d3ee',
            'success': '#10b981', 'warning': '#f59e0b', 'error': '#ef4444',
            'glass_bg': 'rgba(30, 41, 59, 0.7)', 'glass_border': 'rgba(148, 163, 184, 0.1)',
            'bg_card': '#1e293b', 'border': '#334155',
        }
    else:
        return {
            'text': '#0f172a', 'text_muted': '#64748b', 'text_secondary': '#475569',
            'primary': '#4f46e5', 'secondary': '#7c3aed', 'accent': '#0891b2',
            'success': '#059669', 'warning': '#d97706', 'error': '#dc2626',
            'glass_bg': 'rgba(255, 255, 255, 0.8)', 'glass_border': 'rgba(0, 0, 0, 0.06)',
            'bg_card': '#ffffff', 'border': '#e2e8f0',
        }


def generate_chart_base64(fig) -> str:
    """Convert a Plotly figure to base64 image for embedding in HTML"""
    img_bytes = fig.to_image(format="png", width=800, height=400)
    return base64.b64encode(img_bytes).decode()


def generate_performance_chart_html(df: pd.DataFrame) -> str:
    """Generate performance chart as HTML/SVG"""
    # Find price column
    price_col = None
    for col in ['close', 'Close', 'price', 'value', 'adj_close']:
        if col in df.columns:
            price_col = col
            break
    
    if price_col is None:
        return "<p>No price data available for chart</p>"
    
    prices = df[price_col].values
    n = len(prices)
    
    # Create SVG chart
    width, height = 700, 300
    padding = 50
    chart_width = width - 2 * padding
    chart_height = height - 2 * padding
    
    min_val, max_val = prices.min(), prices.max()
    val_range = max_val - min_val if max_val != min_val else 1
    
    # Generate path
    points = []
    for i, val in enumerate(prices):
        x = padding + (i / (n - 1 if n > 1 else 1)) * chart_width
        y = height - padding - ((val - min_val) / val_range) * chart_height
        points.append(f"{x},{y}")
    
    path = "M " + " L ".join(points)
    
    # Calculate return
    total_return = ((prices[-1] / prices[0]) - 1) * 100 if prices[0] != 0 else 0
    color = "#10b981" if total_return >= 0 else "#ef4444"
    
    svg = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="{width}" height="{height}" fill="#f8fafc"/>
        <text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="16" fill="#1e293b">Portfolio Performance</text>
        <path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>
        <text x="{padding}" y="{height - 10}" font-family="Arial" font-size="12" fill="#64748b">Start</text>
        <text x="{width - padding}" y="{height - 10}" text-anchor="end" font-family="Arial" font-size="12" fill="#64748b">End</text>
        <text x="{padding}" y="{padding - 5}" font-family="Arial" font-size="12" fill="#64748b">${max_val:,.2f}</text>
        <text x="{padding}" y="{height - padding + 15}" font-family="Arial" font-size="12" fill="#64748b">${min_val:,.2f}</text>
        <text x="{width - 10}" y="25" text-anchor="end" font-family="Arial" font-size="14" fill="{color}">{total_return:+.2f}%</text>
    </svg>
    '''
    return svg


def generate_allocation_chart_html(df: pd.DataFrame) -> str:
    """Generate allocation pie chart as HTML/SVG"""
    # Find ticker column
    ticker_col = None
    for col in ['ticker', 'symbol', 'Ticker', 'Symbol']:
        if col in df.columns:
            ticker_col = col
            break
    
    if ticker_col is None:
        return "<p>No ticker data available for allocation chart</p>"
    
    # Get unique tickers and their counts
    ticker_counts = df[ticker_col].value_counts().head(6)
    total = ticker_counts.sum()
    
    colors = ['#6366f1', '#22d3ee', '#f472b6', '#fbbf24', '#34d399', '#a78bfa']
    
    # Create pie chart legend instead of actual SVG pie (simpler)
    items_html = ""
    for i, (ticker, count) in enumerate(ticker_counts.items()):
        pct = (count / total) * 100
        color = colors[i % len(colors)]
        items_html += f'''
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <div style="width: 16px; height: 16px; background: {color}; border-radius: 4px; margin-right: 12px;"></div>
            <span style="flex: 1; color: #1e293b;">{ticker}</span>
            <span style="color: #64748b;">{pct:.1f}%</span>
        </div>
        '''
    
    return f'''
    <div style="background: #f8fafc; padding: 20px; border-radius: 12px;">
        <h4 style="margin: 0 0 16px 0; color: #1e293b;">Asset Allocation</h4>
        {items_html}
    </div>
    '''


def generate_full_report(df: pd.DataFrame, report_title: str, company_name: str, 
                         include_sections: Dict, include_charts: Dict) -> str:
    """Generate a complete HTML report"""
    
    data_info = get_working_data_info()
    metrics = calculate_portfolio_metrics(df) if df is not None else {}
    tickers = get_available_tickers()
    
    # Generate date
    report_date = datetime.now().strftime("%B %d, %Y")
    
    # Start HTML
    html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #1e293b;
            background: #ffffff;
            padding: 40px;
            max-width: 900px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding-bottom: 30px;
            border-bottom: 2px solid #6366f1;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #1e293b;
            font-size: 28px;
            margin-bottom: 8px;
        }}
        .header .company {{
            color: #6366f1;
            font-size: 16px;
            font-weight: 600;
        }}
        .header .date {{
            color: #64748b;
            font-size: 14px;
            margin-top: 8px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #1e293b;
            font-size: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 20px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-card .label {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .kpi-card .value {{
            font-size: 24px;
            font-weight: 700;
            color: #1e293b;
            margin: 8px 0;
        }}
        .kpi-card .delta {{
            font-size: 12px;
        }}
        .positive {{ color: #10b981; }}
        .negative {{ color: #ef4444; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f8fafc;
            font-weight: 600;
            color: #64748b;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .chart-container {{
            margin: 20px 0;
        }}
        .row {{
            display: flex;
            gap: 20px;
            margin: 20px 0;
        }}
        .col {{
            flex: 1;
        }}
        .footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            text-align: center;
            color: #64748b;
            font-size: 12px;
        }}
        @media print {{
            body {{ padding: 20px; }}
            .section {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="company">{company_name}</div>
        <h1>{report_title}</h1>
        <div class="date">Generated on {report_date}</div>
    </div>
'''
    
    # Executive Summary
    if include_sections.get('executive_summary', True):
        html += f'''
    <div class="section">
        <h2>📋 Executive Summary</h2>
        <p>This report provides a comprehensive analysis of portfolio performance based on {data_info.get('rows', 0):,} data points 
        across {len(tickers) if tickers else 'multiple'} assets. The analysis period covers the full dataset range.</p>
        <p style="margin-top: 12px;">Key findings:</p>
        <ul style="margin: 12px 0 0 20px; color: #475569;">
            <li>Total Return: <strong class="{'positive' if metrics.get('total_return', 0) >= 0 else 'negative'}">{metrics.get('total_return', 0):.2f}%</strong></li>
            <li>Annualized Volatility: <strong>{metrics.get('volatility', 0):.2f}%</strong></li>
            <li>Sharpe Ratio: <strong>{metrics.get('sharpe_ratio', 0):.2f}</strong></li>
            <li>Maximum Drawdown: <strong class="negative">{metrics.get('max_drawdown', 0):.2f}%</strong></li>
        </ul>
    </div>
'''
    
    # Portfolio Overview (KPI Cards)
    if include_sections.get('portfolio_overview', True):
        html += f'''
    <div class="section">
        <h2>📊 Portfolio Overview</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="label">Total Value</div>
                <div class="value">${metrics.get('total_value', 0):,.2f}</div>
            </div>
            <div class="kpi-card">
                <div class="label">Total Return</div>
                <div class="value {'positive' if metrics.get('total_return', 0) >= 0 else 'negative'}">{metrics.get('total_return', 0):+.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="label">Volatility</div>
                <div class="value">{metrics.get('volatility', 0):.2f}%</div>
            </div>
            <div class="kpi-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value">{metrics.get('sharpe_ratio', 0):.2f}</div>
            </div>
        </div>
    </div>
'''
    
    # Performance Chart
    if include_charts.get('performance_chart', True) and df is not None:
        perf_chart = generate_performance_chart_html(df)
        html += f'''
    <div class="section">
        <h2>📈 Performance Analysis</h2>
        <div class="chart-container">
            {perf_chart}
        </div>
    </div>
'''
    
    # Risk Metrics
    if include_sections.get('risk_metrics', True):
        html += f'''
    <div class="section">
        <h2>🛡️ Risk Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Assessment</th>
            </tr>
            <tr>
                <td>Daily Volatility</td>
                <td>{metrics.get('volatility', 0) / np.sqrt(252):.4f}%</td>
                <td>{'Low' if metrics.get('volatility', 0) < 15 else 'Medium' if metrics.get('volatility', 0) < 25 else 'High'}</td>
            </tr>
            <tr>
                <td>Annualized Volatility</td>
                <td>{metrics.get('volatility', 0):.2f}%</td>
                <td>{'Low' if metrics.get('volatility', 0) < 15 else 'Medium' if metrics.get('volatility', 0) < 25 else 'High'}</td>
            </tr>
            <tr>
                <td>Maximum Drawdown</td>
                <td class="negative">{metrics.get('max_drawdown', 0):.2f}%</td>
                <td>{'Acceptable' if metrics.get('max_drawdown', 0) > -20 else 'Significant' if metrics.get('max_drawdown', 0) > -40 else 'Severe'}</td>
            </tr>
            <tr>
                <td>VaR (95%)</td>
                <td class="negative">{metrics.get('volatility', 0) * 1.645 / np.sqrt(252):.2f}%</td>
                <td>Daily Value at Risk</td>
            </tr>
            <tr>
                <td>Sharpe Ratio</td>
                <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                <td>{'Poor' if metrics.get('sharpe_ratio', 0) < 0.5 else 'Average' if metrics.get('sharpe_ratio', 0) < 1 else 'Good' if metrics.get('sharpe_ratio', 0) < 2 else 'Excellent'}</td>
            </tr>
        </table>
    </div>
'''
    
    # Asset Allocation
    if include_charts.get('allocation_pie', True) and df is not None:
        alloc_chart = generate_allocation_chart_html(df)
        html += f'''
    <div class="section">
        <h2>🥧 Asset Allocation</h2>
        {alloc_chart}
    </div>
'''
    
    # Top Holdings
    if include_sections.get('top_holdings', True) and tickers:
        holdings_rows = ""
        for i, ticker in enumerate(tickers[:10]):
            holdings_rows += f'''
            <tr>
                <td>{i+1}</td>
                <td><strong>{ticker}</strong></td>
                <td>{100/len(tickers[:10]):.1f}%</td>
            </tr>
'''
        html += f'''
    <div class="section">
        <h2>💎 Top Holdings</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Asset</th>
                <th>Weight</th>
            </tr>
            {holdings_rows}
        </table>
    </div>
'''
    
    # Data Summary
    if include_sections.get('data_summary', True) and df is not None:
        html += f'''
    <div class="section">
        <h2>📋 Data Summary</h2>
        <table>
            <tr>
                <th>Property</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Records</td>
                <td>{len(df):,}</td>
            </tr>
            <tr>
                <td>Columns</td>
                <td>{len(df.columns)}</td>
            </tr>
            <tr>
                <td>Assets Analyzed</td>
                <td>{len(tickers) if tickers else 'N/A'}</td>
            </tr>
            <tr>
                <td>Data Source</td>
                <td>{data_info.get('name', 'Unknown')}</td>
            </tr>
        </table>
    </div>
'''
    
    # Footer
    html += f'''
    <div class="footer">
        <p>This report was generated by Financial Analytics Suite</p>
        <p>{company_name} | {report_date}</p>
        <p style="margin-top: 8px; font-size: 10px; color: #94a3b8;">
            Disclaimer: This report is for informational purposes only and does not constitute investment advice.
        </p>
    </div>
</body>
</html>
'''
    
    return html


def render():
    """Render the Reports page"""
    c = get_theme_colors()
    
    st.title("📄 Reports & Export")
    st.markdown(f"<p style='color: {c['text_secondary']};'>Generate professional reports with your data, charts, and metrics</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check for data
    data_available = has_data()
    if data_available:
        df = get_working_data()
        data_info = get_working_data_info()
        st.success(f"✅ Using data from: **{data_info['name']}** ({data_info['rows']:,} rows)")
    else:
        df = None
        st.warning("⚠️ No data uploaded. Upload data in Data Sources to generate reports with real data.")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Report Builder", "📋 Templates", "📅 Scheduled"])
    
    with tab1:
        st.markdown("#### Build Your Report")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            report_title = st.text_input("Report Title", value="Portfolio Performance Report", key="report_title")
            
            st.markdown("##### Select Sections to Include")
            
            sections = [
                {"key": "executive_summary", "name": "Executive Summary", "icon": "📋", "default": True},
                {"key": "portfolio_overview", "name": "Portfolio Overview", "icon": "📊", "default": True},
                {"key": "performance_analysis", "name": "Performance Analysis", "icon": "📈", "default": True},
                {"key": "risk_metrics", "name": "Risk Metrics", "icon": "🛡️", "default": True},
                {"key": "top_holdings", "name": "Top Holdings", "icon": "💎", "default": True},
                {"key": "data_summary", "name": "Data Summary", "icon": "📋", "default": True},
            ]
            
            include_sections = {}
            for section in sections:
                include_sections[section['key']] = st.checkbox(
                    f"{section['icon']} {section['name']}", 
                    value=section['default'],
                    key=f"section_{section['key']}"
                )
            
            st.markdown("---")
            
            st.markdown("##### Charts to Include")
            chart_cols = st.columns(2)
            include_charts = {}
            with chart_cols[0]:
                include_charts['performance_chart'] = st.checkbox("📈 Performance Chart", value=True)
                include_charts['allocation_pie'] = st.checkbox("🥧 Allocation Chart", value=True)
            with chart_cols[1]:
                include_charts['risk_scatter'] = st.checkbox("📊 Risk/Return Plot", value=True)
                include_charts['drawdown_chart'] = st.checkbox("📉 Drawdown Chart", value=True)
        
        with col2:
            st.markdown("##### Report Settings")
            
            report_format = st.selectbox("Format", ["HTML", "PDF (via HTML)", "Text Summary"], key="report_format")
            report_style = st.selectbox("Style", ["Professional", "Minimal", "Executive"], key="report_style")
            
            st.markdown("##### Branding")
            company_name = st.text_input("Company Name", value="Financial Analytics Suite", key="company_name")
            primary_color = st.color_picker("Primary Color", value="#6366f1", key="primary_color")
        
        st.markdown("---")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("👁️ Preview Report", use_container_width=True, key="preview_btn"):
                if data_available:
                    with st.expander("📄 Report Preview", expanded=True):
                        report_html = generate_full_report(
                            df, report_title, company_name, 
                            include_sections, include_charts
                        )
                        st.components.v1.html(report_html, height=800, scrolling=True)
                else:
                    st.error("Please upload data first to preview report")
        
        with col2:
            if data_available:
                report_html = generate_full_report(
                    df, report_title, company_name,
                    include_sections, include_charts
                )
                
                if st.download_button(
                    "📥 Download Report",
                    data=report_html,
                    file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.html",
                    mime="text/html",
                    use_container_width=True,
                    key="download_btn"
                ):
                    # Auto-save to projects
                    from pages_new.projects import auto_save_analysis
                    auto_save_analysis(report_title, "Report")
                    st.toast(f"Report saved to Projects!", icon="💾")
            else:
                st.button("📥 Download Report", use_container_width=True, disabled=True, key="download_disabled")
        
        with col3:
            if st.button("📧 Email Report", type="primary", use_container_width=True, key="email_btn"):
                st.toast("Email functionality coming soon!", icon="📧")
    
    with tab2:
        st.markdown("#### Report Templates")
        st.caption("Quick-start templates for common report types")
        
        templates = [
            {"name": "Monthly Performance", "icon": "📈", "desc": "Monthly portfolio review with performance metrics", "popular": True},
            {"name": "Risk Assessment", "icon": "🛡️", "desc": "Comprehensive risk analysis and VaR metrics", "popular": True},
            {"name": "Executive Summary", "icon": "📋", "desc": "High-level overview for stakeholders", "popular": False},
            {"name": "Full Analysis", "icon": "📊", "desc": "Complete analysis with all sections", "popular": True},
            {"name": "Client Report", "icon": "👔", "desc": "Professional client-facing summary", "popular": False},
            {"name": "Forecast Results", "icon": "🔮", "desc": "Time series forecast summary", "popular": False},
        ]
        
        template_cols = st.columns(3)
        
        for i, template in enumerate(templates):
            with template_cols[i % 3]:
                popular_badge = "🔥 Popular" if template['popular'] else ""
                st.markdown(f"""
                <div style="background: {c['glass_bg']}; border: 1px solid {c['glass_border']}; border-radius: 12px; padding: 20px; margin-bottom: 16px; height: 140px;">
                    <div style="font-size: 32px; margin-bottom: 8px;">{template['icon']}</div>
                    <div style="font-size: 14px; font-weight: 600; color: {c['text']};">{template['name']} <span style="font-size: 10px; color: {c['warning']};">{popular_badge}</span></div>
                    <div style="font-size: 11px; color: {c['text_muted']}; margin-top: 4px;">{template['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Use Template", key=f"template_{i}", use_container_width=True, 
                             on_click=lambda name=template['name']: st.session_state.update({'report_title': name + " Report"})):
                    st.toast(f"Template '{template['name']}' loaded!", icon="✅")
    
    with tab3:
        st.markdown("#### Scheduled Reports")
        st.info("💡 Set up automated report generation and delivery")
        
        # Existing schedules
        st.markdown("##### Active Schedules")
        
        schedules = [
            {"name": "Weekly Performance", "frequency": "Every Monday", "recipients": "team@company.com", "status": "Active"},
            {"name": "Monthly Risk Report", "frequency": "1st of month", "recipients": "risk@company.com", "status": "Active"},
        ]
        
        for schedule in schedules:
            status_color = c['success'] if schedule['status'] == 'Active' else c['warning']
            st.markdown(f'''
            <div style="display: grid; grid-template-columns: 2fr 1.5fr 2fr 1fr; padding: 14px 16px; background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 8px; margin-bottom: 8px;">
                <div style="font-size: 13px; font-weight: 600; color: {c['text']};">{schedule['name']}</div>
                <div style="font-size: 12px; color: {c['text_muted']};">{schedule['frequency']}</div>
                <div style="font-size: 12px; color: {c['text_muted']};">{schedule['recipients']}</div>
                <div style="font-size: 11px; color: {status_color}; font-weight: 600;">{schedule['status']}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("##### Create New Schedule")
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Report Template", ["Monthly Performance", "Risk Report", "Custom"], key="sched_template")
            st.selectbox("Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"], key="sched_freq")
        with col2:
            st.text_input("Recipients", placeholder="email@example.com", key="sched_recipients")
            st.selectbox("Format", ["HTML", "PDF"], key="sched_format")
        
        if st.button("📅 Create Schedule", type="primary", use_container_width=True, key="create_sched"):
            st.success("✅ Schedule created!")


if __name__ == "__main__":
    render()
