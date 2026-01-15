"""
Financial Analytics Suite - Theme Engine
Dynamic theme generation with dark/light mode and system preference detection
"""

import streamlit as st
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
from .tokens import (
    ColorPalette, DarkThemeColors, LightThemeColors,
    SPACING, FONT_FAMILY, FONT_SIZE, FONT_WEIGHT, RADIUS,
    GRADIENTS, Z_INDEX, TRANSITION, CHART_COLORS
)


class ThemeMode(Enum):
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"


@dataclass
class Theme:
    """Complete theme configuration"""
    mode: ThemeMode
    colors: Dict[str, str]
    palette: ColorPalette
    
    @classmethod
    def create(cls, mode: ThemeMode = ThemeMode.DARK) -> 'Theme':
        """Factory method to create theme with proper colors"""
        palette = ColorPalette()
        
        if mode == ThemeMode.DARK:
            theme_colors = DarkThemeColors()
        else:
            theme_colors = LightThemeColors()
        
        colors = {
            # Core theme colors
            'bg_base': theme_colors.bg_base,
            'bg_surface': theme_colors.bg_surface,
            'bg_card': theme_colors.bg_card,
            'bg_elevated': theme_colors.bg_elevated,
            'bg_hover': theme_colors.bg_hover,
            'bg_active': theme_colors.bg_active,
            'bg_overlay': theme_colors.bg_overlay,
            
            # Glass effects
            'glass_bg': theme_colors.glass_bg,
            'glass_border': theme_colors.glass_border,
            
            # Text
            'text': theme_colors.text_primary,
            'text_secondary': theme_colors.text_secondary,
            'text_muted': theme_colors.text_muted,
            'text_disabled': theme_colors.text_disabled,
            'text_inverse': theme_colors.text_inverse,
            'text_link': theme_colors.text_link,
            
            # Borders
            'border': theme_colors.border_default,
            'border_light': theme_colors.border_light,
            'border_dark': theme_colors.border_dark,
            'border_focus': theme_colors.border_focus,
            
            # Shadows
            'shadow_sm': theme_colors.shadow_sm,
            'shadow_md': theme_colors.shadow_md,
            'shadow_lg': theme_colors.shadow_lg,
            'shadow_xl': theme_colors.shadow_xl,
            'shadow_glow': theme_colors.shadow_glow,
            
            # Brand colors
            'primary': palette.primary_500,
            'primary_hover': palette.primary_600,
            'primary_light': palette.primary_400,
            'primary_bg': palette.primary_500 + "15",
            
            'secondary': palette.secondary_500,
            'secondary_hover': palette.secondary_600,
            
            'accent': palette.accent_400,
            'accent_hover': palette.accent_500,
            
            # Semantic colors
            'success': palette.success_500,
            'success_light': palette.success_400,
            'success_bg': palette.success_500 + "15",
            
            'warning': palette.warning_500,
            'warning_light': palette.warning_400,
            'warning_bg': palette.warning_500 + "15",
            
            'error': palette.error_500,
            'error_light': palette.error_400,
            'error_bg': palette.error_500 + "15",
            
            # Gradients
            'gradient_primary': GRADIENTS['primary'],
            'gradient_secondary': GRADIENTS['secondary'],
            'gradient_accent': GRADIENTS['accent'],
            'gradient_mesh': GRADIENTS['mesh_dark'] if mode == ThemeMode.DARK else GRADIENTS['mesh_light'],
        }
        
        return cls(mode=mode, colors=colors, palette=palette)


def get_current_theme() -> Theme:
    """Get the current theme from session state"""
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = ThemeMode.DARK
    
    return Theme.create(st.session_state.theme_mode)


def toggle_theme():
    """Toggle between dark and light mode"""
    current = st.session_state.get('theme_mode', ThemeMode.DARK)
    st.session_state.theme_mode = ThemeMode.LIGHT if current == ThemeMode.DARK else ThemeMode.DARK


def generate_css(theme: Theme) -> str:
    """Generate the complete CSS for the theme"""
    c = theme.colors
    
    return f"""
    <style>
    /* ═══════════════════════════════════════════════════════════════════════════════
       FINANCIAL ANALYTICS SUITE - 2026 ULTRA-MODERN DESIGN SYSTEM
       Clean, premium fintech aesthetic with glassmorphism and micro-interactions
       ═══════════════════════════════════════════════════════════════════════════════ */
    
    /* ══════════════ IMPORTS & RESETS ══════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    *, *::before, *::after {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }}
    
    /* Hide Streamlit defaults */
    #MainMenu, footer {{visibility: hidden !important;}}
    header {{background: transparent !important;}}
    [data-testid="stSidebarNav"] {{display: none !important;}}
    [data-testid="stMainMenu"] {{visibility: hidden !important;}}
    
    /* ══════════════ BASE TYPOGRAPHY ══════════════ */
    html, body, .stApp {{
        font-family: {FONT_FAMILY['sans']} !important;
        font-size: {FONT_SIZE['base']};
        font-weight: {FONT_WEIGHT['normal']};
        color: {c['text']} !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    
    pre, code, .stCodeBlock, [data-testid="stText"] pre {{
        font-family: {FONT_FAMILY['mono']} !important;
    }}
    
    /* ══════════════ KEYFRAME ANIMATIONS ══════════════ */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes fadeInDown {{
        from {{ opacity: 0; transform: translateY(-20px); }}
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
        50% {{ opacity: 0.7; }}
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
        0%, 100% {{ box-shadow: {c['shadow_glow']}; }}
        50% {{ box-shadow: 0 0 60px rgba(99, 102, 241, 0.6); }}
    }}
    
    @keyframes borderGlow {{
        0%, 100% {{ border-color: {c['primary']}40; }}
        50% {{ border-color: {c['primary']}; }}
    }}
    
    @keyframes spin {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}
    
    @keyframes skeleton {{
        0% {{ background-position: -200px 0; }}
        100% {{ background-position: calc(200px + 100%) 0; }}
    }}
    
    /* ══════════════ MAIN APP CONTAINER ══════════════ */
    .stApp {{
        background: {c['bg_base']} !important;
    }}
    
    /* Animated mesh gradient background */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: {c['gradient_mesh']};
        pointer-events: none;
        z-index: -1;
    }}
    
    /* Main content area */
    .main .block-container {{
        padding: {SPACING['6']} {SPACING['8']} !important;
        max-width: 100% !important;
        animation: fadeInUp 0.5s ease-out;
    }}
    
    /* ══════════════ SIDEBAR - PREMIUM GLASS ══════════════ */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {c['bg_surface']} 0%, {c['bg_base']} 100%) !important;
        border-right: 1px solid {c['border']} !important;
        backdrop-filter: blur(20px);
        animation: fadeInLeft 0.4s ease-out;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        background: transparent !important;
        padding-top: {SPACING['4']} !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: {c['text']} !important;
    }}
    
    /* Sidebar collapse button */
    [data-testid="stSidebar"] button[kind="header"],
    [data-testid="collapsedControl"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['md']} !important;
        transition: {TRANSITION['smooth']} !important;
    }}
    
    [data-testid="stSidebar"] button[kind="header"]:hover {{
        background: {c['primary_bg']} !important;
        border-color: {c['primary']} !important;
    }}
    
    /* ══════════════ TYPOGRAPHY ══════════════ */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        background: {c['gradient_primary']} !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: {FONT_WEIGHT['extrabold']} !important;
        letter-spacing: {LETTER_SPACING['tight']};
        animation: fadeInUp 0.4s ease-out;
    }}
    
    .stApp h1 {{ font-size: {FONT_SIZE['4xl']} !important; }}
    .stApp h2 {{ font-size: {FONT_SIZE['3xl']} !important; }}
    .stApp h3 {{ font-size: {FONT_SIZE['2xl']} !important; }}
    .stApp h4 {{ font-size: {FONT_SIZE['xl']} !important; }}
    
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp li,
    .stMarkdown, .stMarkdown p {{
        color: {c['text']} !important;
    }}
    
    /* ══════════════ GLASSMORPHISM CARDS ══════════════ */
    .glass-card {{
        background: {c['glass_bg']} !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid {c['glass_border']} !important;
        border-radius: {RADIUS['xl']} !important;
        box-shadow: {c['shadow_lg']} !important;
        transition: {TRANSITION['smooth']} !important;
        animation: fadeInUp 0.5s ease-out;
    }}
    
    .glass-card:hover {{
        transform: translateY(-4px) !important;
        box-shadow: {c['shadow_xl']}, {c['shadow_glow']} !important;
        border-color: {c['primary']}40 !important;
    }}
    
    .premium-card {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['xl']} !important;
        padding: {SPACING['6']} !important;
        box-shadow: {c['shadow_md']} !important;
        transition: {TRANSITION['smooth']} !important;
        animation: fadeInUp 0.5s ease-out;
    }}
    
    .premium-card:hover {{
        transform: translateY(-3px) !important;
        box-shadow: {c['shadow_lg']}, {c['shadow_glow']} !important;
        border-color: {c['primary']} !important;
    }}
    
    /* ══════════════ BUTTONS ══════════════ */
    .stButton > button {{
        background: {c['bg_card']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['lg']} !important;
        padding: {SPACING['3']} {SPACING['5']} !important;
        font-weight: {FONT_WEIGHT['semibold']} !important;
        font-size: {FONT_SIZE['sm']} !important;
        letter-spacing: {LETTER_SPACING['wide']};
        transition: {TRANSITION['smooth']} !important;
        box-shadow: {c['shadow_sm']} !important;
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
        background: linear-gradient(90deg, transparent, {c['primary']}20, transparent);
        transition: left 0.5s ease;
    }}
    
    .stButton > button:hover {{
        background: {c['bg_elevated']} !important;
        border-color: {c['primary']} !important;
        transform: translateY(-2px) !important;
        box-shadow: {c['shadow_md']}, 0 0 20px {c['primary']}30 !important;
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}
    
    /* Primary button */
    .stButton > button[kind="primary"] {{
        background: {c['gradient_primary']} !important;
        background-size: 200% 200% !important;
        animation: gradientFlow 3s ease infinite !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 16px {c['primary']}40 !important;
    }}
    
    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 8px 32px {c['primary']}60, {c['shadow_glow']} !important;
        transform: translateY(-3px) scale(1.02) !important;
    }}
    
    /* ══════════════ INPUTS & FORMS ══════════════ */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea textarea {{
        background: {c['bg_card']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['lg']} !important;
        padding: {SPACING['3']} {SPACING['4']} !important;
        font-size: {FONT_SIZE['base']} !important;
        transition: {TRANSITION['smooth']} !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea textarea:focus {{
        border-color: {c['primary']} !important;
        box-shadow: 0 0 0 3px {c['primary']}25, 0 0 20px {c['primary']}15 !important;
        outline: none !important;
    }}
    
    .stTextInput label, .stNumberInput label, .stTextArea label,
    .stCheckbox label, .stRadio label, .stSelectbox label,
    .stSlider label, .stMultiSelect label {{
        color: {c['text']} !important;
        font-weight: {FONT_WEIGHT['medium']} !important;
        font-size: {FONT_SIZE['xs']} !important;
        text-transform: uppercase !important;
        letter-spacing: {LETTER_SPACING['wider']} !important;
        margin-bottom: {SPACING['2']} !important;
    }}
    
    /* ══════════════ SELECTS & DROPDOWNS ══════════════ */
    .stSelectbox > div > div,
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect > div > div,
    [data-baseweb="select"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['lg']} !important;
        color: {c['text']} !important;
        transition: {TRANSITION['smooth']} !important;
    }}
    
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['xl']} !important;
        box-shadow: {c['shadow_xl']} !important;
        animation: scaleIn 0.15s ease-out;
    }}
    
    [data-baseweb="menu"] li {{
        color: {c['text']} !important;
        background: transparent !important;
        transition: {TRANSITION['fast']} !important;
        border-radius: {RADIUS['md']} !important;
        margin: 2px 4px !important;
    }}
    
    [data-baseweb="menu"] li:hover {{
        background: {c['primary_bg']} !important;
        color: {c['primary']} !important;
    }}
    
    /* ══════════════ METRICS - ANIMATED KPI CARDS ══════════════ */
    [data-testid="stMetric"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['xl']} !important;
        padding: {SPACING['5']} {SPACING['6']} !important;
        animation: fadeInUp 0.5s ease-out;
        transition: {TRANSITION['smooth']} !important;
    }}
    
    [data-testid="stMetric"]:hover {{
        border-color: {c['primary']} !important;
        box-shadow: {c['shadow_lg']}, 0 0 20px {c['primary']}20 !important;
        transform: translateY(-2px);
    }}
    
    [data-testid="stMetricValue"] {{
        font-size: {FONT_SIZE['4xl']} !important;
        font-weight: {FONT_WEIGHT['extrabold']} !important;
        background: {c['gradient_primary']} !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {c['text_secondary']} !important;
        font-weight: {FONT_WEIGHT['semibold']} !important;
        text-transform: uppercase !important;
        letter-spacing: {LETTER_SPACING['widest']} !important;
        font-size: {FONT_SIZE['xs']} !important;
    }}
    
    [data-testid="stMetricDelta"] {{
        font-weight: {FONT_WEIGHT['semibold']} !important;
    }}
    
    [data-testid="stMetricDelta"] svg {{
        animation: pulse 2s infinite !important;
    }}
    
    /* ══════════════ DATA TABLES ══════════════ */
    .stDataFrame, .stTable, [data-testid="stDataFrame"] {{
        animation: fadeInUp 0.5s ease-out;
    }}
    
    [data-testid="stDataFrameResizable"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['xl']} !important;
        overflow: hidden !important;
    }}
    
    .stDataFrame th {{
        background: {c['primary']} !important;
        color: white !important;
        font-weight: {FONT_WEIGHT['semibold']} !important;
        text-transform: uppercase !important;
        letter-spacing: {LETTER_SPACING['wide']} !important;
        font-size: {FONT_SIZE['xs']} !important;
        padding: {SPACING['3']} {SPACING['4']} !important;
        position: sticky !important;
        top: 0 !important;
    }}
    
    .stDataFrame td {{
        background: {c['bg_card']} !important;
        color: {c['text']} !important;
        padding: {SPACING['3']} {SPACING['4']} !important;
        border-bottom: 1px solid {c['border']} !important;
        font-size: {FONT_SIZE['sm']} !important;
        transition: background 0.15s ease !important;
    }}
    
    .stDataFrame tr:hover td {{
        background: {c['primary_bg']} !important;
    }}
    
    /* ══════════════ TABS ══════════════ */
    .stTabs [data-baseweb="tab-list"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['xl']} !important;
        padding: {SPACING['1.5']} !important;
        gap: {SPACING['1.5']} !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {c['text_secondary']} !important;
        border-radius: {RADIUS['lg']} !important;
        padding: {SPACING['3']} {SPACING['5']} !important;
        font-weight: {FONT_WEIGHT['semibold']} !important;
        font-size: {FONT_SIZE['sm']} !important;
        transition: {TRANSITION['smooth']} !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: {c['primary_bg']} !important;
        color: {c['text']} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {c['gradient_primary']} !important;
        color: white !important;
        box-shadow: 0 4px 16px {c['primary']}40 !important;
    }}
    
    /* ══════════════ EXPANDERS ══════════════ */
    .streamlit-expanderHeader {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['xl']} !important;
        padding: {SPACING['4']} {SPACING['5']} !important;
        font-weight: {FONT_WEIGHT['semibold']} !important;
        color: {c['text']} !important;
        transition: {TRANSITION['smooth']} !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        border-color: {c['primary']} !important;
        background: {c['bg_elevated']} !important;
    }}
    
    [data-testid="stExpander"] {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['xl']} !important;
        overflow: hidden !important;
    }}
    
    /* ══════════════ ALERTS ══════════════ */
    .stAlert {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['xl']} !important;
        padding: {SPACING['4']} {SPACING['5']} !important;
        animation: fadeInUp 0.3s ease-out;
    }}
    
    .stSuccess {{ border-left: 4px solid {c['success']} !important; }}
    .stWarning {{ border-left: 4px solid {c['warning']} !important; }}
    .stError {{ border-left: 4px solid {c['error']} !important; }}
    .stInfo {{ border-left: 4px solid {c['accent']} !important; }}
    
    /* ══════════════ PROGRESS BARS ══════════════ */
    .stProgress > div > div > div {{
        background: {c['gradient_primary']} !important;
        border-radius: {RADIUS['full']} !important;
    }}
    
    .stProgress > div > div {{
        background: {c['bg_elevated']} !important;
        border-radius: {RADIUS['full']} !important;
    }}
    
    /* ══════════════ SLIDERS ══════════════ */
    .stSlider > div > div {{
        background: {c['bg_card']} !important;
        border-radius: {RADIUS['lg']} !important;
        padding: {SPACING['3']} {SPACING['4']} !important;
    }}
    
    .stSlider [data-baseweb="slider"] div {{
        background: {c['primary']} !important;
    }}
    
    /* ══════════════ RADIO & CHECKBOX ══════════════ */
    .stRadio > div {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['xl']} !important;
        padding: {SPACING['4']} !important;
    }}
    
    .stCheckbox span {{
        color: {c['text']} !important;
    }}
    
    /* ══════════════ FILE UPLOADER ══════════════ */
    [data-testid="stFileUploader"] {{
        background: {c['bg_card']} !important;
        border: 2px dashed {c['border']} !important;
        border-radius: {RADIUS['xl']} !important;
        padding: {SPACING['6']} !important;
        transition: {TRANSITION['smooth']} !important;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: {c['primary']} !important;
        background: {c['primary_bg']} !important;
    }}
    
    /* ══════════════ PLOTLY CHARTS ══════════════ */
    .js-plotly-plot {{
        animation: fadeInUp 0.5s ease-out;
    }}
    
    .js-plotly-plot .plotly .modebar {{
        background: {c['bg_card']} !important;
        border-radius: {RADIUS['md']} !important;
    }}
    
    /* ══════════════ CODE BLOCKS ══════════════ */
    .stCodeBlock, pre, code {{
        background: {c['bg_surface']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['lg']} !important;
    }}
    
    .stJson {{
        background: {c['bg_card']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['lg']} !important;
    }}
    
    /* ══════════════ DOWNLOAD BUTTON ══════════════ */
    .stDownloadButton button {{
        background: {c['bg_card']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['lg']} !important;
        transition: {TRANSITION['smooth']} !important;
    }}
    
    .stDownloadButton button:hover {{
        border-color: {c['success']} !important;
        color: {c['success']} !important;
        box-shadow: 0 4px 16px {c['success']}30 !important;
    }}
    
    /* ══════════════ SPINNER ══════════════ */
    .stSpinner > div {{
        border-color: {c['primary']} transparent transparent !important;
    }}
    
    /* ══════════════ SCROLLBARS ══════════════ */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {c['bg_base']};
        border-radius: {RADIUS['full']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {c['border']};
        border-radius: {RADIUS['full']};
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {c['primary']};
    }}
    
    /* ══════════════ SKELETON LOADER ══════════════ */
    .skeleton {{
        background: linear-gradient(90deg, {c['bg_card']} 25%, {c['bg_elevated']} 50%, {c['bg_card']} 75%);
        background-size: 200px 100%;
        animation: skeleton 1.5s ease-in-out infinite;
        border-radius: {RADIUS['md']};
    }}
    
    /* ══════════════ UTILITY CLASSES ══════════════ */
    .fade-in {{ animation: fadeInUp 0.5s ease-out; }}
    .fade-in-left {{ animation: fadeInLeft 0.5s ease-out; }}
    .fade-in-right {{ animation: fadeInRight 0.5s ease-out; }}
    .scale-in {{ animation: scaleIn 0.3s ease-out; }}
    .floating {{ animation: float 6s ease-in-out infinite; }}
    .glowing {{ animation: glow 3s ease-in-out infinite; }}
    .pulsing {{ animation: pulse 2s ease-in-out infinite; }}
    
    /* ══════════════ NAVIGATION SECTION HEADERS ══════════════ */
    .nav-section-header {{
        background: linear-gradient(90deg, {c['primary_bg']}, transparent);
        padding: {SPACING['2.5']} {SPACING['4']};
        margin: {SPACING['5']} 0 {SPACING['2.5']} 0;
        border-radius: {RADIUS['md']};
        border-left: 3px solid {c['primary']};
        font-weight: {FONT_WEIGHT['bold']};
        font-size: {FONT_SIZE['2xs']};
        text-transform: uppercase;
        letter-spacing: {LETTER_SPACING['widest']};
        color: {c['primary']} !important;
        animation: fadeInLeft 0.3s ease-out;
    }}
    
    /* ══════════════ STATUS INDICATORS ══════════════ */
    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: {RADIUS['full']};
        display: inline-block;
    }}
    
    .status-dot.online {{ background: {c['success']}; box-shadow: 0 0 8px {c['success']}; }}
    .status-dot.offline {{ background: {c['error']}; }}
    .status-dot.pending {{ background: {c['warning']}; animation: pulse 2s infinite; }}
    
    /* ══════════════ TOOLTIP STYLES ══════════════ */
    .tooltip {{
        background: {c['bg_card']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: {RADIUS['md']} !important;
        padding: {SPACING['2']} {SPACING['3']} !important;
        font-size: {FONT_SIZE['xs']} !important;
        box-shadow: {c['shadow_lg']} !important;
    }}
    
    /* ══════════════ BADGE STYLES ══════════════ */
    .badge {{
        display: inline-flex;
        align-items: center;
        padding: {SPACING['1']} {SPACING['2.5']};
        border-radius: {RADIUS['full']};
        font-size: {FONT_SIZE['xs']};
        font-weight: {FONT_WEIGHT['medium']};
    }}
    
    .badge-primary {{ background: {c['primary_bg']}; color: {c['primary']}; }}
    .badge-success {{ background: {c['success_bg']}; color: {c['success']}; }}
    .badge-warning {{ background: {c['warning_bg']}; color: {c['warning']}; }}
    .badge-error {{ background: {c['error_bg']}; color: {c['error']}; }}
    
    /* ══════════════ ACCESSIBILITY ══════════════ */
    :focus-visible {{
        outline: 2px solid {c['primary']} !important;
        outline-offset: 2px !important;
    }}
    
    /* High contrast mode support */
    @media (prefers-contrast: high) {{
        .stApp {{
            --focus-ring: 3px solid {c['primary']};
        }}
    }}
    
    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
    }}
    
    </style>
    """


def apply_theme(theme: Optional[Theme] = None):
    """Apply the theme CSS to the Streamlit app"""
    if theme is None:
        theme = get_current_theme()
    
    css = generate_css(theme)
    st.markdown(css, unsafe_allow_html=True)
    
    return theme
