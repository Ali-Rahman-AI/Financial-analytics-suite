"""
Financial Analytics Suite - About Page
Premium showcase of platform capabilities with 2026 fintech design
"""

import streamlit as st
from typing import Dict
import time

# Import theme colors from central theme_utils
from pages_new.theme_utils import get_theme_colors


def render():
    """Render the About page with premium 2026 fintech design"""
    c = get_theme_colors()
    
    # Inject premium CSS
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
    }}
    
    @keyframes shimmer {{
        0% {{ background-position: -1000px 0; }}
        100% {{ background-position: 1000px 0; }}
    }}
    
    @keyframes pulse-glow {{
        0%, 100% {{ box-shadow: 0 0 20px {c['primary']}40; }}
        50% {{ box-shadow: 0 0 40px {c['primary']}60, 0 0 60px {c['secondary']}40; }}
    }}
    
    @keyframes gradientFlow {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    @keyframes counter {{
        from {{ --num: 0; }}
        to {{ --num: var(--target); }}
    }}
    
    .about-hero {{
        background: {c['gradient']};
        background-size: 200% 200%;
        animation: gradientFlow 6s ease infinite;
        border-radius: 24px;
        padding: 60px 50px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 60px {c['primary']}30;
        margin-bottom: 40px;
    }}
    
    .about-hero::before {{
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 200%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        animation: shimmer 4s infinite;
    }}
    
    .about-hero::after {{
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }}
    
    .hero-icon {{
        font-size: 72px;
        margin-bottom: 20px;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 10px 20px rgba(0,0,0,0.3));
    }}
    
    .hero-title {{
        font-size: 48px;
        font-weight: 800;
        color: white;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 8px;
        letter-spacing: -0.02em;
    }}
    
    .hero-subtitle {{
        font-size: 20px;
        color: rgba(255,255,255,0.9);
        margin-bottom: 30px;
    }}
    
    .hero-version {{
        display: inline-block;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        padding: 10px 24px;
        border-radius: 999px;
        font-weight: 600;
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
    }}
    
    .stat-card {{
        background: {c['glass_bg']};
        backdrop-filter: blur(20px);
        border: 1px solid {c['glass_border']};
        border-radius: 20px;
        padding: 28px 24px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out backwards;
        position: relative;
        overflow: hidden;
    }}
    
    .stat-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 3px;
        background: {c['gradient']};
    }}
    
    .stat-card:hover {{
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px {c['primary']}25;
        border-color: {c['primary']};
    }}
    
    .stat-number {{
        font-size: 42px;
        font-weight: 800;
        background: {c['gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }}
    
    .stat-label {{
        color: {c['text_secondary']};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 12px;
    }}
    
    .feature-card {{
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 20px;
        padding: 32px 28px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out backwards;
        position: relative;
        overflow: hidden;
        height: 100%;
    }}
    
    .feature-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: {c['gradient']};
        opacity: 0;
        transition: opacity 0.3s;
    }}
    
    .feature-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15), 0 0 30px {c['primary']}15;
        border-color: {c['primary']};
    }}
    
    .feature-card:hover::before {{
        opacity: 1;
    }}
    
    .feature-icon {{
        width: 60px;
        height: 60px;
        background: {c['gradient']};
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px {c['primary']}30;
        animation: pulse-glow 3s infinite;
    }}
    
    .feature-title {{
        font-size: 20px;
        font-weight: 700;
        color: {c['text']};
        margin-bottom: 12px;
    }}
    
    .feature-desc {{
        color: {c['text_secondary']};
        line-height: 1.7;
        font-size: 14px;
    }}
    
    .tech-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: {c['glass_bg']};
        backdrop-filter: blur(10px);
        border: 1px solid {c['glass_border']};
        border-radius: 12px;
        padding: 12px 20px;
        margin: 6px;
        font-weight: 600;
        color: {c['text']};
        transition: all 0.3s ease;
        font-size: 13px;
    }}
    
    .tech-badge:hover {{
        transform: translateY(-3px) scale(1.05);
        background: {c['primary']}15;
        border-color: {c['primary']};
        box-shadow: 0 8px 24px {c['primary']}20;
    }}
    
    .section-header {{
        font-size: 32px;
        font-weight: 800;
        background: {c['gradient']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 12px;
        letter-spacing: -0.02em;
    }}
    
    .section-subtitle {{
        color: {c['text_secondary']};
        font-size: 16px;
        margin-bottom: 32px;
    }}
    
    .use-case-card {{
        background: {c['glass_bg']};
        backdrop-filter: blur(16px);
        border: 1px solid {c['glass_border']};
        border-radius: 16px;
        padding: 28px;
        transition: all 0.3s ease;
        height: 100%;
    }}
    
    .use-case-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        border-color: {c['primary']}50;
    }}
    
    .use-case-icon {{
        font-size: 36px;
        margin-bottom: 16px;
    }}
    
    .use-case-title {{
        font-size: 18px;
        font-weight: 700;
        color: {c['text']};
        margin-bottom: 12px;
    }}
    
    .use-case-desc {{
        color: {c['text_secondary']};
        font-size: 14px;
        line-height: 1.6;
    }}
    
    .step-card {{
        display: flex;
        align-items: flex-start;
        gap: 20px;
        padding: 24px;
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 16px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }}
    
    .step-card:hover {{
        transform: translateX(8px);
        border-color: {c['primary']};
        box-shadow: 0 10px 30px {c['primary']}10;
    }}
    
    .step-number {{
        width: 48px;
        height: 48px;
        background: {c['gradient']};
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: 800;
        color: white;
        flex-shrink: 0;
    }}
    
    .step-content h4 {{
        color: {c['text']};
        font-weight: 700;
        margin-bottom: 6px;
    }}
    
    .step-content p {{
        color: {c['text_secondary']};
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }}
    
    .disclaimer-card {{
        background: linear-gradient(135deg, {c['warning']}15, {c['warning']}08);
        border: 1px solid {c['warning']}30;
        border-radius: 16px;
        padding: 24px 28px;
        margin-top: 40px;
    }}
    
    .disclaimer-title {{
        color: {c['warning']};
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    .disclaimer-text {{
        color: {c['text_secondary']};
        font-size: 14px;
        line-height: 1.7;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown(f"""
    <div class="about-hero">
        <div style="text-align: center; position: relative; z-index: 2;">
            <div class="hero-icon">🚀</div>
            <div class="hero-title">Financial Analytics Suite</div>
            <div class="hero-subtitle">Empowering Financial Decisions with Advanced Analytics & AI-Driven Insights</div>
            <span class="hero-version">✨ v3.0 Professional Edition</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Section
    cols = st.columns(4)
    stats = [
        ("12+", "📊", "Analytics Modules"),
        ("10+", "🗄️", "Data Formats"),
        ("15+", "⚙️", "ML Models"),
        ("∞", "🎯", "Possibilities"),
    ]
    
    for i, (value, icon, label) in enumerate(stats):
        with cols[i]:
            st.markdown(f"""
            <div class="stat-card" style="animation-delay: {i * 0.1}s;">
                <div style="font-size: 28px; margin-bottom: 8px;">{icon}</div>
                <div class="stat-number">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    
    # Core Features Section
    st.markdown(f"""
    <div class="section-header">⭐ Core Features</div>
    <div class="section-subtitle">Powerful tools designed for modern financial analysis</div>
    """, unsafe_allow_html=True)
    
    features = [
        ("🚀", "Portfolio Optimization", "Maximize returns while minimizing risk using Modern Portfolio Theory, Black-Litterman models, and Monte Carlo simulations. Optimize asset allocation with efficient frontier analysis."),
        ("🛡️", "Risk Management", "Calculate Value at Risk (VaR), Conditional VaR (CVaR), and perform stress testing. Monitor portfolio volatility with GARCH models and manage downside risk effectively."),
        ("🧠", "AI-Powered Predictions", "Leverage machine learning algorithms including Random Forest, Gradient Boosting, and Neural Networks to forecast market trends and identify profitable trading opportunities."),
        ("📈", "Time Series Forecasting", "Predict future prices using ARIMA, ETS, TBATS, and Prophet models. Compare forecasting accuracy with backtesting and validation metrics like RMSE and MAE."),
    ]
    
    feature_cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        with feature_cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card" style="animation-delay: {i * 0.15}s; margin-bottom: 24px;">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    
    # Technology Stack - Horizontal with logos
    st.markdown(f"""
    <div class="section-header">💻 Technology Stack</div>
    <div class="section-subtitle">Built with cutting-edge technologies for maximum performance</div>
    """, unsafe_allow_html=True)
    
    # Technology data with logo URLs
    techs = [
        ("Python", "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg"),
        ("Streamlit", "https://streamlit.io/images/brand/streamlit-mark-color.svg"),
        ("NumPy", "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg"),
        ("Pandas", "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg"),
        ("Plotly", "https://images.plot.ly/logo/new-branding/plotly-logomark.png"),
        ("Scikit-learn", "https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg"),
        ("XGBoost", "https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png"),
        ("TensorFlow", "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg"),
        ("SQLAlchemy", "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/sqlalchemy/sqlalchemy-original.svg"),
        ("PostgreSQL", "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/postgresql/postgresql-original.svg"),
        ("MySQL", "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/mysql/mysql-original.svg"),
        ("R Lang", "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/r/r-original.svg"),
    ]
    
    # Build all cards HTML in one string
    cards_html = ""
    for name, logo_url in techs:
        cards_html += f'''<div style="display: inline-block; vertical-align: top; width: 90px; padding: 14px 8px; background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 12px; text-align: center; margin: 8px; transition: all 0.3s ease; cursor: pointer;" onmouseover="this.style.transform='translateY(-4px)'; this.style.borderColor='{c['primary']}'; this.style.boxShadow='0 8px 20px {c['primary']}30';" onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='{c['border']}'; this.style.boxShadow='none';"><img src="{logo_url}" alt="{name}" style="width: 36px; height: 36px; object-fit: contain; margin-bottom: 8px;" onerror="this.src='https://cdn-icons-png.flaticon.com/512/2721/2721286.png';"><div style="font-size: 10px; font-weight: 600; color: {c['text']}; line-height: 1.2;">{name}</div></div>'''
    
    # Wrap in container and render
    st.markdown(f'''
    <div style="
        text-align: center;
        padding: 24px 16px;
        background: {c['glass_bg']};
        backdrop-filter: blur(16px);
        border: 1px solid {c['glass_border']};
        border-radius: 20px;
        margin-bottom: 30px;
    ">
        {cards_html}
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    # Perfect For Section
    st.markdown(f"""
    <div class="section-header">👥 Perfect For</div>
    <div class="section-subtitle">Designed for professionals across the financial industry</div>
    """, unsafe_allow_html=True)
    
    use_cases = [
        ("🎓", "Students & Researchers", "Learn financial modeling, portfolio theory, and risk management with hands-on practice and real-world datasets."),
        ("📈", "Financial Analysts", "Perform rapid exploratory analysis, backtest strategies, and generate insightful reports for stakeholders."),
        ("💼", "Portfolio Managers", "Optimize asset allocation, monitor risk metrics, and make data-driven investment decisions with confidence."),
    ]
    
    use_cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(use_cases):
        with use_cols[i]:
            st.markdown(f"""
            <div class="use-case-card">
                <div class="use-case-icon">{icon}</div>
                <div class="use-case-title">{title}</div>
                <div class="use-case-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown(f"""
    <div class="section-header">▶️ Quick Start Guide</div>
    <div class="section-subtitle">Get started in four simple steps</div>
    """, unsafe_allow_html=True)
    
    steps = [
        ("1", "Import Your Data", "Click DATA SOURCES and upload CSV, Excel, or use our demo financial dataset to get started instantly."),
        ("2", "Clean & Prepare", "Handle missing values, remove outliers, and calculate returns using our comprehensive data cleaning tools."),
        ("3", "Analyze & Model", "Run forecasting models, calculate risk metrics, and optimize your portfolio with advanced algorithms."),
        ("4", "Export Results", "Download cleaned data, generate comprehensive reports, and export visualizations for presentations."),
    ]
    
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-number">{num}</div>
            <div class="step-content">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer-card">
        <div class="disclaimer-title">⚠️ Disclaimer</div>
        <div class="disclaimer-text">
            This suite is designed for educational and analytical purposes. Always validate results 
            and consult with financial professionals before making real-world investment decisions. 
            Past performance does not guarantee future results.
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    render()
