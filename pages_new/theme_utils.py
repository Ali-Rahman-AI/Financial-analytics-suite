"""
Financial Analytics Suite - Theme Utilities
Simple, clean theme system
"""

import streamlit as st
from typing import Dict


def get_theme_colors() -> Dict[str, str]:
    """Get current theme colors"""
    theme_mode = st.session_state.get('theme_mode', 'dark')
    
    themes = {
        'dark': {
            'bg': '#0f172a', 'bg_card': '#1e293b', 'bg_surface': '#1e293b', 'bg_elevated': '#334155',
            'text': '#f8fafc', 'text_muted': '#64748b', 'text_secondary': '#94a3b8',
            'primary': '#6366f1', 'secondary': '#8b5cf6', 'accent': '#22d3ee', 'accent2': '#f472b6',
            'success': '#10b981', 'warning': '#f59e0b', 'error': '#ef4444',
            'border': '#334155', 'border_light': '#475569',
            'gradient': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
            'glass_bg': 'rgba(30, 41, 59, 0.8)', 'glass_border': 'rgba(148, 163, 184, 0.1)',
            'glow': '0 0 20px rgba(99, 102, 241, 0.3)',
            'chart_colors': ['#6366f1', '#8b5cf6', '#22d3ee', '#f472b6', '#10b981', '#f59e0b'],
        },
        'light': {
            'bg': '#f8fafc', 'bg_card': '#ffffff', 'bg_surface': '#ffffff', 'bg_elevated': '#f1f5f9',
            'text': '#0f172a', 'text_muted': '#64748b', 'text_secondary': '#475569',
            'primary': '#4f46e5', 'secondary': '#7c3aed', 'accent': '#0891b2', 'accent2': '#db2777',
            'success': '#059669', 'warning': '#d97706', 'error': '#dc2626',
            'border': '#e2e8f0', 'border_light': '#f1f5f9',
            'gradient': 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)',
            'glass_bg': 'rgba(255, 255, 255, 0.9)', 'glass_border': 'rgba(0, 0, 0, 0.05)',
            'glow': '0 0 20px rgba(79, 70, 229, 0.15)',
            'chart_colors': ['#4f46e5', '#7c3aed', '#0891b2', '#db2777', '#059669', '#d97706'],
        },
        'ocean': {
            'bg': '#0c1222', 'bg_card': '#1e3a5f', 'bg_surface': '#111827', 'bg_elevated': '#2d4a6f',
            'text': '#f0f9ff', 'text_muted': '#38bdf8', 'text_secondary': '#7dd3fc',
            'primary': '#0ea5e9', 'secondary': '#06b6d4', 'accent': '#14b8a6', 'accent2': '#f97316',
            'success': '#10b981', 'warning': '#fbbf24', 'error': '#f43f5e',
            'border': '#0369a1', 'border_light': '#0284c7',
            'gradient': 'linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%)',
            'glass_bg': 'rgba(30, 58, 95, 0.8)', 'glass_border': 'rgba(56, 189, 248, 0.15)',
            'glow': '0 0 30px rgba(14, 165, 233, 0.4)',
            'chart_colors': ['#0ea5e9', '#06b6d4', '#14b8a6', '#f97316', '#10b981', '#7dd3fc'],
        },
        'midnight': {
            'bg': '#0a0015', 'bg_card': '#2e1065', 'bg_surface': '#1a0a2e', 'bg_elevated': '#4c1d95',
            'text': '#faf5ff', 'text_muted': '#c084fc', 'text_secondary': '#d8b4fe',
            'primary': '#a855f7', 'secondary': '#ec4899', 'accent': '#f472b6', 'accent2': '#fbbf24',
            'success': '#22c55e', 'warning': '#f59e0b', 'error': '#ef4444',
            'border': '#6b21a8', 'border_light': '#7c3aed',
            'gradient': 'linear-gradient(135deg, #a855f7 0%, #ec4899 100%)',
            'glass_bg': 'rgba(46, 16, 101, 0.8)', 'glass_border': 'rgba(168, 85, 247, 0.2)',
            'glow': '0 0 40px rgba(168, 85, 247, 0.5)',
            'chart_colors': ['#a855f7', '#ec4899', '#f472b6', '#fbbf24', '#22c55e', '#d8b4fe'],
        }
    }
    
    return themes.get(theme_mode, themes['dark'])


def inject_global_styles():
    """Inject simple global CSS styles"""
    c = get_theme_colors()
    
    st.markdown(f'''
    <style>
        .stApp {{
            background: {c['bg']} !important;
            color: {c['text']} !important;
        }}
        
        /* Text visibility */
        .stMarkdown, .stMarkdown p, .stMarkdown span, h1, h2, h3, h4, h5, h6, p, span, label {{
            color: {c['text']} !important;
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background: {c['bg_surface']};
            border-radius: 10px;
            padding: 4px;
            gap: 4px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: {c['text_muted']} !important;
            border-radius: 8px;
            padding: 10px 20px;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: {c['gradient']} !important;
            color: white !important;
        }}
        
        /* Buttons */
        .stButton > button {{
            border-radius: 8px !important;
            transition: all 0.2s ease !important;
            border: 1px solid {c['border']} !important;
            font-weight: 500 !important;
        }}
        
        /* Secondary Buttons */
        .stButton > button[kind="secondary"] {{
            background: {c['bg_card']} !important;
            color: {c['text']} !important;
        }}
        
        /* Primary Buttons */
        .stButton > button[kind="primary"] {{
            background: {c['gradient']} !important;
            color: white !important;
            border: none !important;
            box-shadow: {c['glow']} !important;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-1px);
            border-color: {c['primary']} !important;
            box-shadow: {c['glow']} !important;
        }}
        
        /* Metrics */
        [data-testid="stMetric"] {{
            background: {c['bg_card']};
            border: 1px solid {c['border']};
            border-radius: 12px;
            padding: 16px;
        }}
        
        [data-testid="stMetricLabel"] {{ color: {c['text_secondary']} !important; }}
        [data-testid="stMetricValue"] {{ color: {c['text']} !important; }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: {c['bg_surface']} !important;
            border-right: 1px solid {c['border']} !important;
        }}
        
        [data-testid="stSidebar"] * {{ color: {c['text']} !important; }}
        
        /* Inputs */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {{
            background: {c['bg_card']} !important;
            border: 1px solid {c['border']} !important;
            color: {c['text']} !important;
            border-radius: 8px !important;
        }}
        
        .stSelectbox > div > div {{
            background: {c['bg_card']} !important;
            border: 1px solid {c['border']} !important;
            color: {c['text']} !important;
        }}
        
        /* Tables */
        .stDataFrame {{ border-radius: 10px !important; overflow: hidden !important; }}
        
        /* Expanders */
        .streamlit-expanderHeader {{
            background: {c['bg_card']} !important;
            border: 1px solid {c['border']} !important;
            border-radius: 8px !important;
            color: {c['text']} !important;
        }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: {c['bg_surface']}; }}
        ::-webkit-scrollbar-thumb {{ background: {c['primary']}60; border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: {c['primary']}; }}
    </style>
    ''', unsafe_allow_html=True)


def inject_premium_styles():
    """Inject premium 2026 fintech design system CSS"""
    c = get_theme_colors()
    
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* ═══════════════ KEYFRAME ANIMATIONS ═══════════════ */
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-6px); }}
    }}
    
    @keyframes gradientFlow {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    @keyframes slideInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes scaleIn {{
        from {{ opacity: 0; transform: scale(0.9); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    
    @keyframes pulse-glow {{
        0%, 100% {{ box-shadow: 0 0 20px {c['primary']}40; }}
        50% {{ box-shadow: 0 0 35px {c['primary']}70; }}
    }}
    
    /* ═══════════════ GLASS COMPONENTS ═══════════════ */
    .glass-card {{
        background: {c['glass_bg']};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid {c['glass_border']};
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInUp 0.6s ease-out backwards;
    }}
    
    .glass-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 20px 40px {c['primary']}20;
        border-color: {c['primary']}50;
    }}
    
    .glass-header {{
        background: {c['gradient']};
        background-size: 200% 200%;
        animation: gradientFlow 5s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.02em;
    }}
    
    /* ═══════════════ SECTION HEADERS ═══════════════ */
    .section-header {{
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 8px;
        color: {c['text']};
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .section-subtitle {{
        font-size: 14px;
        color: {c['text_secondary']};
        margin-bottom: 24px;
    }}
    
    /* ═══════════════ STATUS BADGES ═══════════════ */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background: {c['bg_elevated']};
        border: 1px solid {c['border']};
    }}
    
    .status-success {{ color: {c['success']}; border-color: {c['success']}40; background: {c['success']}10; }}
    .status-warning {{ color: {c['warning']}; border-color: {c['warning']}40; background: {c['warning']}10; }}
    .status-error {{ color: {c['error']}; border-color: {c['error']}40; background: {c['error']}10; }}
    
    </style>
    """, unsafe_allow_html=True)


def render_plot_with_download(fig, name: str = "chart", key: str = None):
    """Render a plotly figure with download options"""
    chart_key = key if key else f"chart_{name}_{hash(str(fig)) % 10000}"
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    
    try:
        html_str = fig.to_html(include_plotlyjs='cdn')
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 HTML", data=html_str, file_name=f"{name}.html", 
                             mime="text/html", key=f"dl_html_{chart_key}")
        with col2:
            try:
                img_bytes = fig.to_image(format='png', width=1200, height=600, scale=2)
                st.download_button("📥 PNG", data=img_bytes, file_name=f"{name}.png", 
                                 mime="image/png", key=f"dl_png_{chart_key}")
            except:
                pass
    except:
        pass
