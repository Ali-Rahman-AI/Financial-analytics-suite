"""
Financial Analytics Suite - Main Application
Ultra-Modern 2026 Fintech UI with Premium Design
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Financial Analytics Suite | Enterprise Edition",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Ali-Rahman-AI/financial-analytics-suite',
        'Report a bug': 'https://github.com/Ali-Rahman-AI/financial-analytics-suite/issues',
        'About': '# Financial Analytics Suite\nEnterprise-grade financial analytics platform'
    }
)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        # Core data
        'data': None,
        'cleaned_data': None,
        'edited_data': None,
        
        # Theme & UI
        'theme_mode': 'dark',
        'sidebar_collapsed': False,
        'density_mode': 'comfortable',  # comfortable, compact
        'show_command_palette': False,
        
        # Navigation
        'current_page': 'overview',
        'current_workspace': 'Default Workspace',
        
        # Modals & Panels
        'show_data_modal': False,
        'show_db_modal': False,
        'show_settings_modal': False,
        'show_notifications': False,
        'inspector_panel_open': True,
        
        # Database
        'db_manager': None,
        'db_connected': False,
        
        # Analysis context
        'selected_dataset': None,
        'date_range': (datetime.now() - timedelta(days=365), datetime.now()),
        'selected_currency': 'USD',
        'selected_frequency': 'daily',
        
        # Background jobs
        'background_jobs': [],
        'notifications': [],
        
        # User & Auth
        'user': {
            'name': 'Guest User',
            'email': 'guest@example.com',
            'role': 'viewer',
            'avatar': None
        },
        'authenticated': False,
        
        # Filters
        'global_filters': {},
        'pinned_filters': [],
        
        # Financial warnings
        'financial_warnings': {},
        
        # Recent activity
        'recent_activity': [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ═══════════════════════════════════════════════════════════════════════════════
# THEME SYSTEM - 4 Premium Themes
# ═══════════════════════════════════════════════════════════════════════════════

THEMES = {
    'dark': {
        'name': 'Dark Mode',
        'icon': '🌙',
        'primary': '#6366f1', 'secondary': '#8b5cf6',
        'accent': '#22d3ee', 'accent2': '#f472b6',
        'bg': '#030712', 'bg_surface': '#0f172a', 'bg_card': '#1e293b', 'bg_elevated': '#334155',
        'text': '#f8fafc', 'text_secondary': '#94a3b8', 'text_muted': '#64748b',
        'border': '#334155', 'border_light': '#475569',
        'success': '#10b981', 'warning': '#f59e0b', 'error': '#ef4444',
        'gradient': 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)',
        'glass_bg': 'rgba(30, 41, 59, 0.7)',
        'glass_border': 'rgba(148, 163, 184, 0.1)',
        'glow': '0 0 40px rgba(99, 102, 241, 0.4)',
        'mesh_gradient': '''
            radial-gradient(at 40% 20%, hsla(228,100%,74%,0.12) 0px, transparent 50%),
            radial-gradient(at 80% 0%, hsla(275,100%,70%,0.08) 0px, transparent 50%),
            radial-gradient(at 0% 50%, hsla(339,100%,76%,0.06) 0px, transparent 50%),
            radial-gradient(at 80% 50%, hsla(189,100%,56%,0.06) 0px, transparent 50%)
        '''
    },
    'light': {
        'name': 'Light Mode',
        'icon': '☀️',
        'primary': '#4f46e5', 'secondary': '#7c3aed',
        'accent': '#0891b2', 'accent2': '#db2777',
        'bg': '#fafbfc', 'bg_surface': '#ffffff', 'bg_card': '#ffffff', 'bg_elevated': '#f1f5f9',
        'text': '#0f172a', 'text_secondary': '#475569', 'text_muted': '#64748b',
        'border': '#e2e8f0', 'border_light': '#f1f5f9',
        'success': '#059669', 'warning': '#d97706', 'error': '#dc2626',
        'gradient': 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%)',
        'glass_bg': 'rgba(255, 255, 255, 0.9)',
        'glass_border': 'rgba(0, 0, 0, 0.06)',
        'glow': '0 0 40px rgba(79, 70, 229, 0.15)',
        'mesh_gradient': '''
            radial-gradient(at 40% 20%, hsla(228,100%,74%,0.06) 0px, transparent 50%),
            radial-gradient(at 80% 0%, hsla(275,100%,70%,0.04) 0px, transparent 50%)
        '''
    },
    'ocean': {
        'name': 'Ocean Blue',
        'icon': '🌊',
        'primary': '#0ea5e9', 'secondary': '#06b6d4',
        'accent': '#14b8a6', 'accent2': '#f97316',
        'bg': '#0c1222', 'bg_surface': '#111827', 'bg_card': '#1e3a5f', 'bg_elevated': '#2d4a6f',
        'text': '#f0f9ff', 'text_secondary': '#7dd3fc', 'text_muted': '#38bdf8',
        'border': '#0369a1', 'border_light': '#0284c7',
        'success': '#10b981', 'warning': '#fbbf24', 'error': '#f43f5e',
        'gradient': 'linear-gradient(135deg, #0ea5e9 0%, #06b6d4 50%, #14b8a6 100%)',
        'glass_bg': 'rgba(30, 58, 95, 0.7)',
        'glass_border': 'rgba(56, 189, 248, 0.15)',
        'glow': '0 0 50px rgba(14, 165, 233, 0.5)',
        'mesh_gradient': '''
            radial-gradient(at 20% 30%, hsla(199,100%,60%,0.15) 0px, transparent 50%),
            radial-gradient(at 70% 20%, hsla(188,100%,50%,0.12) 0px, transparent 50%),
            radial-gradient(at 50% 80%, hsla(168,100%,45%,0.08) 0px, transparent 50%)
        '''
    },
    'midnight': {
        'name': 'Midnight Purple',
        'icon': '🔮',
        'primary': '#a855f7', 'secondary': '#ec4899',
        'accent': '#f472b6', 'accent2': '#fbbf24',
        'bg': '#0a0015', 'bg_surface': '#1a0a2e', 'bg_card': '#2e1065', 'bg_elevated': '#4c1d95',
        'text': '#faf5ff', 'text_secondary': '#d8b4fe', 'text_muted': '#c084fc',
        'border': '#6b21a8', 'border_light': '#7c3aed',
        'success': '#22c55e', 'warning': '#f59e0b', 'error': '#ef4444',
        'gradient': 'linear-gradient(135deg, #a855f7 0%, #ec4899 50%, #f472b6 100%)',
        'glass_bg': 'rgba(46, 16, 101, 0.7)',
        'glass_border': 'rgba(168, 85, 247, 0.2)',
        'glow': '0 0 60px rgba(168, 85, 247, 0.6)',
        'mesh_gradient': '''
            radial-gradient(at 30% 20%, hsla(280,100%,70%,0.18) 0px, transparent 50%),
            radial-gradient(at 80% 40%, hsla(330,100%,65%,0.12) 0px, transparent 50%),
            radial-gradient(at 10% 70%, hsla(290,100%,60%,0.10) 0px, transparent 50%)
        '''
    }
}

def get_theme_colors():
    """Get current theme colors"""
    theme_mode = st.session_state.get('theme_mode', 'dark')
    return THEMES.get(theme_mode, THEMES['dark'])

colors = get_theme_colors()


# ═══════════════════════════════════════════════════════════════════════════════
# ULTRA-PREMIUM CSS
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css():
    """Inject the ultra-premium CSS"""
    c = colors
    
    st.markdown(f"""
    <style>
    /* ═══════════════ IMPORTS ═══════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* ═══════════════ RESETS & BASE ═══════════════ */
    #MainMenu, footer {{visibility: hidden !important;}}
    header {{background: transparent !important;}}
    [data-testid="stSidebarNav"] {{display: none !important;}}
    [data-testid="stMainMenu"] {{visibility: hidden !important;}}
    
    html, body, .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    
    pre, code, .stCodeBlock {{
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    }}
    
    /* ═══════════════ ANIMATIONS ═══════════════ */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes fadeInLeft {{
        from {{ opacity: 0; transform: translateX(-20px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    @keyframes fadeInRight {{
        from {{ opacity: 0; transform: translateX(20px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    @keyframes scaleIn {{
        from {{ opacity: 0; transform: scale(0.95); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}
    
    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-6px); }}
    }}
    
    @keyframes gradientFlow {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    @keyframes glow {{
        0%, 100% {{ box-shadow: {c['glow']}; }}
        50% {{ box-shadow: 0 0 60px rgba(99, 102, 241, 0.6); }}
    }}
    
    @keyframes spin {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    @keyframes rotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    @keyframes rotate-rev {{
        from {{ transform: rotate(360deg); }}
        to {{ transform: rotate(0deg); }}
    }}
    
    /* ═══════════════ MAIN APP ═══════════════ */
    .stApp {{
        background: {c['bg']} !important;
        color: {c['text']} !important;
    }}
    
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: {c['mesh_gradient']};
        pointer-events: none;
        z-index: -1;
    }}
    
    .main .block-container {{
        padding: 1.5rem 2rem !important;
        max-width: 100% !important;
        animation: fadeInUp 0.4s ease-out;
    }}
    
    /* ═══════════════ SIDEBAR ═══════════════ */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {c['bg_surface']} 0%, {c['bg']} 100%) !important;
        border-right: 1px solid {c['border']} !important;
        animation: fadeInLeft 0.4s ease-out;
        box-shadow: 5px 0 25px rgba(0, 0, 0, 0.3) !important;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        background: transparent !important;
        padding-top: 1rem !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: {c['text']} !important;
    }}
    
    [data-testid="stSidebar"] button[kind="header"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }}
    
    [data-testid="stSidebar"] button[kind="header"]:hover {{
        background: {c['primary']}20 !important;
        border-color: {c['primary']} !important;
    }}
    
    /* ═══════════════ STUNNING SIDEBAR SCROLLBAR ═══════════════ */
    /* Sidebar scrollbar width and height */
    [data-testid="stSidebar"] ::-webkit-scrollbar,
    [data-testid="stSidebar"]::-webkit-scrollbar,
    section[data-testid="stSidebar"] ::-webkit-scrollbar {{
        width: 10px !important;
        height: 10px !important;
    }}
    
    /* Sidebar scrollbar track */
    [data-testid="stSidebar"] ::-webkit-scrollbar-track,
    [data-testid="stSidebar"]::-webkit-scrollbar-track,
    section[data-testid="stSidebar"] ::-webkit-scrollbar-track {{
        background: linear-gradient(180deg, 
            {c['bg']}60 0%, 
            {c['bg_surface']} 20%, 
            {c['bg_surface']} 80%, 
            {c['bg']}60 100%) !important;
        border-radius: 10px !important;
        margin: 8px 0 !important;
    }}
    
    /* Sidebar scrollbar thumb - gradient with glow */
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb,
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb,
    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, 
            {c['primary']} 0%, 
            {c['secondary']} 50%, 
            {c['primary']} 100%) !important;
        border-radius: 20px !important;
        border: 2px solid {c['bg_surface']} !important;
        box-shadow: 
            0 0 12px {c['primary']}60,
            inset 0 0 6px rgba(255, 255, 255, 0.2) !important;
        transition: all 0.3s ease !important;
    }}
    
    /* Sidebar scrollbar thumb hover - enhanced glow */
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover,
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover,
    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(180deg, 
            {c['accent']} 0%, 
            {c['primary']} 35%,
            {c['secondary']} 65%, 
            {c['accent']} 100%) !important;
        box-shadow: 
            0 0 20px {c['primary']},
            0 0 35px {c['primary']}80,
            inset 0 0 10px rgba(255, 255, 255, 0.3) !important;
        border-color: {c['primary']}80 !important;
    }}
    
    /* Sidebar scrollbar thumb when dragging */
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb:active,
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb:active,
    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:active {{
        background: {c['gradient']} !important;
        box-shadow: 
            0 0 25px {c['primary']},
            0 0 45px {c['secondary']}90 !important;
    }}
    
    /* Hide scrollbar buttons */
    [data-testid="stSidebar"] ::-webkit-scrollbar-button,
    [data-testid="stSidebar"]::-webkit-scrollbar-button {{
        display: none !important;
        height: 0 !important;
    }}
    
    /* Scrollbar corner */
    [data-testid="stSidebar"] ::-webkit-scrollbar-corner {{
        background: transparent !important;
    }}
    
    /* Firefox support */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {{
        scrollbar-width: thin !important;
        scrollbar-color: {c['primary']} {c['bg_surface']} !important;
    }}
    
    /* Global scrollbar styling */
    ::-webkit-scrollbar {{
        width: 12px;
        height: 12px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {c['bg_surface']};
        border-radius: 8px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: linear-gradient(180deg, {c['primary']}70, {c['secondary']}70);
        border-radius: 8px;
        border: 2px solid {c['bg_surface']};
        transition: all 0.3s ease;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {c['gradient']};
        box-shadow: 0 0 15px {c['primary']}60;
    }}
    
    
    /* ═══════════════ TYPOGRAPHY ═══════════════ */
    .stApp h1, .stApp h2, .stApp h3 {{
        color: {c['primary']} !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        animation: fadeInUp 0.4s ease-out;
        -webkit-text-fill-color: {c['primary']} !important;
    }}
    
    .stApp h4, .stApp h5, .stApp h6 {{
        color: {c['text']} !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        -webkit-text-fill-color: {c['text']} !important;
    }}
    
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp li,
    .stMarkdown, .stMarkdown p, .stMarkdown span {{
        color: {c['text']} !important;
    }}
    
    /* ═══════════════ GLASSMORPHISM CARDS ═══════════════ */
    .glass-card {{
        background: {c['glass_bg']} !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid {c['glass_border']} !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        animation: fadeInUp 0.5s ease-out;
    }}
    
    .glass-card:hover {{
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3), {c['glow']} !important;
        border-color: {c['primary']}40 !important;
    }}
    
    .premium-card {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        animation: fadeInUp 0.5s ease-out;
    }}
    
    .premium-card:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25), {c['glow']} !important;
        border-color: {c['primary']} !important;
    }}
    
    /* ═══════════════ BUTTONS ═══════════════ */
    .stButton > button {{
        background: {c['bg_card']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        letter-spacing: 0.02em !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, {c['primary']}15, transparent);
        transition: left 0.5s ease;
    }}
    
    .stButton > button:hover {{
        background: {c['bg_elevated']} !important;
        border-color: {c['primary']} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2), 0 0 20px {c['primary']}25 !important;
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button[kind="primary"] {{
        background: {c['gradient']} !important;
        background-size: 200% 200% !important;
        animation: gradientFlow 3s ease infinite !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 16px {c['primary']}40 !important;
    }}
    
    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 8px 32px {c['primary']}60, {c['glow']} !important;
        transform: translateY(-3px) scale(1.02) !important;
    }}
    
    /* ═══════════════ INPUTS ═══════════════ */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea textarea {{
        background: {c['bg_card']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea textarea:focus {{
        border-color: {c['primary']} !important;
        box-shadow: 0 0 0 3px {c['primary']}20, 0 0 20px {c['primary']}10 !important;
        outline: none !important;
    }}
    
    .stTextInput label, .stNumberInput label, .stTextArea label,
    .stCheckbox label, .stRadio label, .stSelectbox label,
    .stSlider label, .stMultiSelect label {{
        color: {c['text']} !important;
        font-weight: 500 !important;
        font-size: 12px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }}
    
    /* ═══════════════ SELECTS ═══════════════ */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    [data-baseweb="select"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 10px !important;
        color: {c['text']} !important;
        transition: all 0.3s ease !important;
    }}
    
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
        animation: scaleIn 0.15s ease-out;
    }}
    
    [data-baseweb="menu"] li {{
        color: {c['text']} !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
        border-radius: 8px !important;
        margin: 2px 4px !important;
    }}
    
    [data-baseweb="menu"] li:hover {{
        background: {c['primary']}15 !important;
        color: {c['primary']} !important;
    }}
    
    /* ═══════════════ METRICS ═══════════════ */
    [data-testid="stMetric"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 16px !important;
        padding: 20px 24px !important;
        animation: fadeInUp 0.5s ease-out;
        transition: all 0.3s ease !important;
    }}
    
    [data-testid="stMetric"]:hover {{
        border-color: {c['primary']} !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2), 0 0 20px {c['primary']}15 !important;
        transform: translateY(-2px);
    }}
    
    [data-testid="stMetricValue"] {{
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: {c['primary']} !important;
        -webkit-text-fill-color: {c['primary']} !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {c['text_secondary']} !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        font-size: 11px !important;
    }}
    
    /* ═══════════════ DATA TABLES ═══════════════ */
    [data-testid="stDataFrameResizable"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }}
    
    .stDataFrame th {{
        background: {c['primary']} !important;
        color: white !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        font-size: 11px !important;
        padding: 14px 16px !important;
    }}
    
    .stDataFrame td {{
        background: {c['bg_card']} !important;
        color: {c['text']} !important;
        padding: 12px 16px !important;
        border-bottom: 1px solid {c['border']} !important;
    }}
    
    .stDataFrame tr:hover td {{
        background: {c['primary']}10 !important;
    }}
    
    /* ═══════════════ TABS ═══════════════ */
    .stTabs [data-baseweb="tab-list"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 14px !important;
        padding: 6px !important;
        gap: 6px !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {c['text_secondary']} !important;
        border-radius: 10px !important;
        padding: 12px 20px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        transition: all 0.3s ease !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: {c['primary']}15 !important;
        color: {c['text']} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {c['gradient']} !important;
        color: white !important;
        box-shadow: 0 4px 16px {c['primary']}40 !important;
    }}
    
    /* ═══════════════ EXPANDERS ═══════════════ */
    .streamlit-expanderHeader {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        font-weight: 600 !important;
        color: {c['text']} !important;
        transition: all 0.3s ease !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        border-color: {c['primary']} !important;
        background: {c['bg_elevated']} !important;
    }}
    
    [data-testid="stExpander"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 12px !important;
    }}
    
    /* ═══════════════ ALERTS ═══════════════ */
    .stAlert {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        animation: fadeInUp 0.3s ease-out;
    }}
    
    .stSuccess {{ border-left: 4px solid {c['success']} !important; }}
    .stWarning {{ border-left: 4px solid {c['warning']} !important; }}
    .stError {{ border-left: 4px solid {c['error']} !important; }}
    .stInfo {{ border-left: 4px solid {c['accent']} !important; }}
    
    /* ═══════════════ PROGRESS ═══════════════ */
    .stProgress > div > div > div {{
        background: {c['gradient']} !important;
        border-radius: 9999px !important;
    }}
    
    hr {{
        border: 0 !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, {c['border']}, transparent) !important;
        margin: 2rem 0 !important;
        opacity: 0.6 !important;
    }}
    
    /* ═══════════════ FILE UPLOADER ═══════════════ */
    [data-testid="stFileUploader"] {{
        background: {c['bg_card']} !important;
        border: 2px dashed {c['border']} !important;
        border-radius: 16px !important;
        padding: 24px !important;
        transition: all 0.3s ease !important;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: {c['primary']} !important;
        background: {c['primary']}08 !important;
    }}
    
    /* ═══════════════ CHARTS ═══════════════ */
    .js-plotly-plot {{
        animation: fadeInUp 0.5s ease-out;
    }}
    
    .js-plotly-plot .plotly .modebar {{
        background: {c['bg_card']} !important;
        border-radius: 8px !important;
    }}
    
    /* ═══════════════ SCROLLBAR ═══════════════ */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: {c['bg']}; }}
    ::-webkit-scrollbar-thumb {{ background: {c['border']}; border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {c['primary']}; }}
    
    /* ═══════════════ NAVIGATION HEADERS ═══════════════ */
    .nav-section-header {{
        background: linear-gradient(90deg, {c['primary']}15, transparent);
        padding: 10px 16px;
        margin: 20px 0 10px 0;
        border-radius: 8px;
        border-left: 3px solid {c['primary']};
        font-weight: 700;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {c['primary']} !important;
        animation: fadeInLeft 0.3s ease-out;
    }}
    
    /* ═══════════════ STATUS BAR ═══════════════ */
    .status-bar {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: {c['bg_surface']};
        border-top: 1px solid {c['border']};
        padding: 8px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 11px;
        color: {c['text_muted']};
        z-index: 100;
    }}
    
    /* ═══════════════ SKELETON LOADER ═══════════════ */
    .skeleton {{
        background: linear-gradient(90deg, {c['bg_card']} 25%, {c['bg_elevated']} 50%, {c['bg_card']} 75%);
        background-size: 200px 100%;
        animation: shimmer 1.5s ease-in-out infinite;
        border-radius: 8px;
    }}
    
    /* ═══════════════ ACCESSIBILITY ═══════════════ */
    :focus-visible {{
        outline: 2px solid {c['primary']} !important;
        outline-offset: 2px !important;
    }}
    
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
    }}
    
    /* ═══════════════ GLOBAL TEXT VISIBILITY FIX ═══════════════ */
    /* Ensure all text elements are visible with proper colors */
    .stApp, .stApp * {{
        color: {c['text']};
    }}
    
    /* Info, Warning, Error, Success boxes */
    .stAlert > div {{
        color: {c['text']} !important;
    }}
    
    .stSuccess, .stInfo, .stWarning, .stError {{
        color: {c['text']} !important;
    }}
    
    /* Caption text */
    .stCaption, small, .element-container small {{
        color: {c['text_muted']} !important;
    }}
    
    /* Placeholder text in inputs */
    input::placeholder, textarea::placeholder {{
        color: {c['text_muted']} !important;
        opacity: 0.7;
    }}
    
    /* Table text */
    .stDataFrame td, .stDataFrame th, table td, table th {{
        color: {c['text']} !important;
    }}
    
    /* Radio buttons and checkboxes labels */
    .stRadio label, .stCheckbox label {{
        color: {c['text']} !important;
    }}
    
    /* File uploader text */
    .stFileUploader label, .stFileUploader span {{
        color: {c['text']} !important;
    }}
    
    /* Number input text */
    .stNumberInput label, .stNumberInput input {{
        color: {c['text']} !important;
    }}
    
    /* Text input text */
    .stTextInput label, .stTextInput input {{
        color: {c['text']} !important;
    }}
    
    /* Selectbox text */
    .stSelectbox label, .stSelectbox span {{
        color: {c['text']} !important;
    }}
    
    /* Toast messages */
    [data-testid="stToast"] {{
        background: {c['bg_card']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
    }}
    
    /* Expander content */
    .streamlit-expanderContent {{
        background: {c['bg_surface']} !important;
        color: {c['text']} !important;
    }}
    
    .streamlit-expanderContent * {{
        color: {c['text']} !important;
    }}
    
    /* Link colors */
    a {{
        color: {c['primary']} !important;
    }}
    
    a:hover {{
        color: {c['accent']} !important;
    }}
    
        /* ═══════════════ SIDEBAR USER CARD ═══════════════ */
    .sidebar-user-card {{
        margin-top: 16px;
        padding: 12px 16px;
        background: {c['bg_card']};
        border: 1px solid {c['border']};
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        transition: all 0.3s ease;
    }}
    
    .sidebar-user-card:hover {{
        background: {c['bg_elevated']};
        border-color: {c['primary']};
        transform: translateY(-2px);
    }}
    
    .sb-img-container {{
        position: relative;
        width: 42px; height: 42px;
        flex-shrink: 0;
    }}
    
    .sb-ring {{
        position: absolute;
        border-radius: 50%;
        border: 1.5px solid transparent;
    }}
    
    .sb-ring-1 {{
        width: 42px; height: 42px;
        border-top-color: rgba(255,255,255,0.8);
        animation: rotate 4s linear infinite;
    }}
    
    .sb-ring-2 {{
        width: 36px; height: 36px;
        top: 3px; left: 3px;
        border-bottom-color: {c['accent']};
        animation: rotate-rev 3s linear infinite;
    }}
    
    .sb-profile-pic {{
        position: absolute;
        width: 30px; height: 30px;
        top: 6px; left: 6px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid rgba(255,255,255,0.4);
    }}

    /* ═══════════════ PREMIUM SCROLLBAR ═══════════════ */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: transparent;
    }}

    ::-webkit-scrollbar-thumb {{
        background: {c['border']};
        border-radius: 10px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {c['primary']};
    }}

    [data-testid="stSidebar"] ::-webkit-scrollbar {{
        width: 10px !important;
    }}

    [data-testid="stSidebar"] ::-webkit-scrollbar-track {{
        background: linear-gradient(180deg, {c['bg']} 0%, {c['bg_surface']} 20%, {c['bg_surface']} 80%, {c['bg']} 100%) !important;
        border-radius: 10px !important;
    }}

    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {{
        background: {c['gradient']} !important;
        border-radius: 20px !important;
        border: 2px solid {c['bg_surface']} !important;
        box-shadow: 0 0 12px {c['primary']}60 !important;
    }}

    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {{
        background: {c['primary']} !important;
        box-shadow: 0 0 20px {c['primary']}80 !important;
    }}
</style>

<script>
// Force sidebar scrollbar persistence
(function() {{
    function fixScroll() {{
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {{
            sidebar.style.overflowY = 'auto';
            sidebar.style.overflowX = 'hidden';
            
            // Re-apply if sidebar re-renders
            const observer = new MutationObserver((mutations) => {{
                sidebar.style.overflowY = 'auto';
            }});
            observer.observe(sidebar, {{ attributes: true, attributeFilter: ['style'] }});
        }} else {{
            setTimeout(fixScroll, 200);
        }}
    }}
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', fixScroll);
    }} else {{
        fixScroll();
    }}
}}());
</script>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

# Page imports - lazy loading
def load_page(page_name: str):
    """Dynamically load and render page"""
    try:
        if page_name == 'overview':
            from pages_new import overview
            overview.render()
        elif page_name == 'projects':
            from pages_new import projects
            projects.render()
        elif page_name == 'data_sources':
            from pages_new import data_sources
            data_sources.render()
        elif page_name == 'data_cleaning':
            from pages_new import data_cleaning
            data_cleaning.render()
        elif page_name == 'dashboard_builder':
            from pages_new import dashboard_builder
            dashboard_builder.render()
        elif page_name == 'executive_dashboard':
            from pages_new.powerbi_dashboard import render_powerbi_dashboard
            render_powerbi_dashboard()
        elif page_name == 'forecasting':
            from pages_new import forecasting
            forecasting.render()
        elif page_name == 'ml_lab':
            from pages_new import ml_lab
            ml_lab.render()
        elif page_name == 'portfolio':
            from pages_new import portfolio
            portfolio.render()
        elif page_name == 'scenario':
            from pages_new import scenario
            scenario.render()
        elif page_name == 'reports':
            from pages_new import reports
            reports.render()
        elif page_name == 'tutorials':
            from pages_new import tutorials
            tutorials.render()
        elif page_name == 'settings':
            from pages_new import settings
            settings.render()
        elif page_name == 'about':
            from pages_new import about
            about.render()
        elif page_name == 'profile':
            from pages_new import profile
            profile.render()
        elif page_name == 'team':
            from pages_new import team
            team.render()
        else:
            render_overview_fallback()
    except ImportError as e:
        st.error(f"Page not found: {page_name}")
        st.info(f"Error details: {str(e)}")
        render_overview_fallback()


def render_overview_fallback():
    """Fallback overview page"""
    st.title("📊 Financial Analytics Suite")
    st.subheader("Welcome to your analytics dashboard")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assets", "$2.4M", "+12.5%")
    with col2:
        st.metric("Portfolio Return", "18.7%", "+3.2%")
    with col3:
        st.metric("Sharpe Ratio", "1.85", "+0.12")
    with col4:
        st.metric("Max Drawdown", "-8.2%", "-1.1%")
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Quick Actions")
        
        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("📥 Import Data", use_container_width=True):
                st.session_state.current_page = 'data_sources'
                st.rerun()
        with action_cols[1]:
            if st.button("📈 Create Dashboard", use_container_width=True):
                st.session_state.current_page = 'dashboard_builder'
                st.rerun()
        with action_cols[2]:
            if st.button("🔮 Run Forecast", use_container_width=True):
                st.session_state.current_page = 'forecasting'
                st.rerun()
    
    with col2:
        st.subheader("📅 Recent Activity")
        
        activities = [
            ("Model training completed", "2 min ago", "success"),
            ("Data imported: Q4 Results", "15 min ago", "info"),
            ("Dashboard shared", "1 hour ago", "info"),
        ]
        
        for activity, time, status in activities:
            st.markdown(f"""
            <div style="
                padding: 12px 16px;
                background: {colors['bg_card']};
                border-radius: 8px;
                margin-bottom: 8px;
                border-left: 3px solid {colors['success'] if status == 'success' else colors['accent']};
            ">
                <div style="font-size: 13px; color: {colors['text']};">{activity}</div>
                <div style="font-size: 11px; color: {colors['text_muted']}; margin-top: 4px;">{time}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        # Logo & Brand - Using reliable text styling
        st.markdown(f"""
        <div style="
            padding: 16px 0 24px 0;
            text-align: center;
            border-bottom: 1px solid {colors['border']};
            margin-bottom: 16px;
        ">
            <div style="font-size: 32px; margin-bottom: 8px;">💎</div>
            <div style="
                font-size: 16px;
                font-weight: 700;
                color: {colors['primary']};
                letter-spacing: -0.02em;
            ">Financial Analytics</div>
            <div style="font-size: 10px; color: {colors['text_muted']}; margin-top: 2px;">Enterprise Suite v2.0</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Workspace selector
        workspace = st.selectbox(
            "WORKSPACE",
            ["Default Workspace", "Portfolio Analysis", "Risk Management"],
            key="workspace_selector"
        )
        
        st.markdown("---")
        
        # Navigation sections
        st.markdown(f'<div class="nav-section-header">📊 ANALYTICS</div>', unsafe_allow_html=True)
        
        nav_items = [
            ("overview", "🏠", "Overview"),
            ("executive_dashboard", "📊", "Executive Dashboard"),
            ("projects", "📁", "Projects"),
            ("dashboard_builder", "🔧", "Dashboard Builder"),
        ]
        
        for page_id, icon, label in nav_items:
            is_active = st.session_state.current_page == page_id
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_page = page_id
                st.rerun()
        
        st.markdown(f'<div class="nav-section-header">📥 DATA</div>', unsafe_allow_html=True)
        
        data_items = [
            ("data_sources", "🔌", "Data Sources"),
            ("data_cleaning", "🧹", "Data Cleaning"),
        ]
        
        for page_id, icon, label in data_items:
            is_active = st.session_state.current_page == page_id
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_page = page_id
                st.rerun()
        
        st.markdown(f'<div class="nav-section-header">🤖 MODELS</div>', unsafe_allow_html=True)
        
        model_items = [
            ("forecasting", "🔮", "Time Series"),
            ("ml_lab", "🧪", "ML Lab"),
            ("portfolio", "💼", "Portfolio"),
            ("scenario", "🎭", "Scenarios"),
        ]
        
        for page_id, icon, label in model_items:
            is_active = st.session_state.current_page == page_id
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_page = page_id
                st.rerun()
        
        st.markdown(f'<div class="nav-section-header">📋 REPORTS</div>', unsafe_allow_html=True)
        
        if st.button("📄  Reports & Export", key="nav_reports", use_container_width=True):
            st.session_state.current_page = 'reports'
            st.rerun()
        
        # Spacer
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        # Info Pages
        st.markdown(f'<div class="nav-section-header">ℹ️ INFO</div>', unsafe_allow_html=True)
        
        info_items = [
            ("about", "🚀", "About"),
            ("tutorials", "🎓", "Video Tutorials"),
            ("profile", "👤", "Profile"),
            ("team", "👥", "Team"),
        ]
        
        for page_id, icon, label in info_items:
            is_active = st.session_state.current_page == page_id
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{page_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_page = page_id
                st.rerun()
        
        # Settings & User
        st.markdown("---")
        
        if st.button("⚙️  Settings", key="nav_settings", use_container_width=True):
            st.session_state.current_page = 'settings'
            st.rerun()
        
        # Theme selector with all 4 themes
        st.markdown(f'<div class="nav-section-header">🎨 THEME</div>', unsafe_allow_html=True)
        
        theme_cols = st.columns(4)
        for i, (theme_key, theme_data) in enumerate(THEMES.items()):
            with theme_cols[i]:
                is_active = st.session_state.theme_mode == theme_key
                if st.button(
                    theme_data['icon'],
                    key=f"theme_{theme_key}",
                    type="primary" if is_active else "secondary",
                    help=theme_data['name'],
                    use_container_width=True
                ):
                    st.session_state.theme_mode = theme_key
                    st.rerun()
        
        # User card
        # Try to get profile image if exists
        try:
            from pages_new.profile import get_image_base64
            img_b64 = get_image_base64()
            user_img = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "https://cdn-icons-png.flaticon.com/512/3135/3135768.png"
        except:
            user_img = "https://cdn-icons-png.flaticon.com/512/3135/3135768.png"

        st.markdown(f"""
        <div class="sidebar-user-card">
            <div class="sb-img-container">
                <div class="sb-ring sb-ring-1"></div>
                <div class="sb-ring sb-ring-2"></div>
                <img src="{user_img}" class="sb-profile-pic" alt="User">
            </div>
            <div>
                <div style="font-size: 13px; font-weight: 600; color: {colors['text']};">Ali Rahman</div>
                <div style="font-size: 11px; color: {colors['text_muted']};">Full Access</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    # Inject CSS
    inject_css()
    
    # Inject global animated styles from theme_utils
    try:
        from pages_new.theme_utils import inject_global_styles
        inject_global_styles()
    except ImportError:
        pass
    
    # Render sidebar
    render_sidebar()
    
    # Load current page
    load_page(st.session_state.current_page)
    
    # Status bar (at bottom)
    st.markdown(f"""
    <div class="status-bar">
        <div style="display: flex; align-items: center; gap: 16px;">
            <span style="display: flex; align-items: center; gap: 6px;">
                <span style="width: 6px; height: 6px; background: {colors['success']}; border-radius: 50%;"></span>
                Connected
            </span>
            <span>|</span>
            <span>Last sync: Just now</span>
        </div>
        <div style="display: flex; align-items: center; gap: 16px;">
            <span>No background tasks</span>
            <span>|</span>
            <span>{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
