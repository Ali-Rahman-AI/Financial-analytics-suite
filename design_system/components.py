"""
Financial Analytics Suite - Premium UI Components
Reusable glassmorphism components with animations and micro-interactions
"""

import streamlit as st
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum


class ButtonVariant(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    GHOST = "ghost"
    DANGER = "danger"
    SUCCESS = "success"


class BadgeVariant(Enum):
    DEFAULT = "default"
    PRIMARY = "primary"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class CardVariant(Enum):
    DEFAULT = "default"
    GLASS = "glass"
    ELEVATED = "elevated"
    OUTLINED = "outlined"


# ═══════════════════════════════════════════════════════════════════════════════
# KPI CARD COMPONENT
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_type: str = "normal",  # normal, good, bad
    icon: str = "📊",
    subtitle: Optional[str] = None,
    sparkline_data: Optional[List[float]] = None,
    trend: Optional[str] = None,  # up, down, stable
    help_text: Optional[str] = None,
    variant: str = "default",
    animate: bool = True
) -> None:
    """
    Render a premium KPI card with optional sparkline and animations
    
    Args:
        title: The KPI metric name
        value: The main value to display
        delta: Change value (e.g., "+5.2%")
        delta_type: normal, good, or bad (affects color)
        icon: Emoji or icon to display
        subtitle: Additional context
        sparkline_data: List of values for mini chart
        trend: up, down, or stable
        help_text: Tooltip explanation
    """
    
    delta_colors = {
        "good": "#10b981",
        "bad": "#ef4444",
        "normal": "#64748b"
    }
    
    delta_color = delta_colors.get(delta_type, delta_colors["normal"])
    
    animation_class = "fade-in" if animate else ""
    
    sparkline_svg = ""
    if sparkline_data and len(sparkline_data) > 1:
        # Generate simple SVG sparkline
        min_val = min(sparkline_data)
        max_val = max(sparkline_data)
        range_val = max_val - min_val if max_val != min_val else 1
        
        width = 80
        height = 24
        points = []
        for i, val in enumerate(sparkline_data):
            x = (i / (len(sparkline_data) - 1)) * width
            y = height - ((val - min_val) / range_val * height)
            points.append(f"{x},{y}")
        
        points_str = " ".join(points)
        sparkline_svg = f'''
        <svg width="{width}" height="{height}" style="margin-left: auto;">
            <polyline 
                fill="none" 
                stroke="{delta_color}" 
                stroke-width="2"
                stroke-linecap="round"
                stroke-linejoin="round"
                points="{points_str}"
            />
        </svg>
        '''
    
    trend_icon = ""
    if trend:
        trend_icons = {"up": "↑", "down": "↓", "stable": "→"}
        trend_icon = f'<span style="margin-left: 4px;">{trend_icons.get(trend, "")}</span>'
    
    help_html = ""
    if help_text:
        help_html = f'''
        <div class="kpi-help" title="{help_text}" style="
            position: absolute;
            top: 12px;
            right: 12px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: rgba(99, 102, 241, 0.2);
            color: #6366f1;
            font-size: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: help;
        ">?</div>
        '''
    
    html = f'''
    <div class="kpi-card {animation_class}" style="
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 20px 24px;
        position: relative;
        transition: all 0.3s ease;
        cursor: pointer;
    " onmouseover="this.style.transform='translateY(-4px)'; this.style.boxShadow='0 12px 40px rgba(0,0,0,0.3), 0 0 40px rgba(99,102,241,0.2)'; this.style.borderColor='rgba(99,102,241,0.4)';"
       onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='none'; this.style.borderColor='rgba(148,163,184,0.1)';">
        {help_html}
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 20px;">{icon}</span>
                    <span style="
                        font-size: 11px;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 0.1em;
                        color: #94a3b8;
                    ">{title}</span>
                </div>
                <div style="
                    font-size: 28px;
                    font-weight: 800;
                    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin-bottom: 4px;
                ">{value}</div>
                {"<div style='display: flex; align-items: center; gap: 4px;'><span style=\"font-size: 13px; font-weight: 600; color: " + delta_color + ";\">" + delta + trend_icon + "</span></div>" if delta else ""}
                {"<div style='font-size: 12px; color: #64748b; margin-top: 4px;'>" + subtitle + "</div>" if subtitle else ""}
            </div>
            {sparkline_svg}
        </div>
    </div>
    '''
    
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def nav_section_header(title: str, icon: str = "") -> None:
    """Render a navigation section header"""
    st.markdown(f'''
    <div class="nav-section-header">
        {icon + " " if icon else ""}{title}
    </div>
    ''', unsafe_allow_html=True)


def nav_button(
    label: str,
    icon: str,
    key: str,
    is_active: bool = False,
    badge_count: Optional[int] = None,
    on_click: Optional[Callable] = None
) -> bool:
    """
    Render a navigation button with icon and optional badge
    
    Returns: True if clicked
    """
    badge_html = ""
    if badge_count is not None and badge_count > 0:
        badge_html = f'''
        <span style="
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            padding: 2px 8px;
            border-radius: 9999px;
            font-size: 10px;
            font-weight: 600;
            margin-left: auto;
        ">{badge_count if badge_count < 100 else "99+"}</span>
        '''
    
    active_style = '''
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.1));
        border-color: rgba(99, 102, 241, 0.4);
    ''' if is_active else ''
    
    return st.button(
        f"{icon}  {label}",
        key=key,
        use_container_width=True,
        type="primary" if is_active else "secondary"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS & BADGES
# ═══════════════════════════════════════════════════════════════════════════════

def status_badge(
    text: str,
    variant: str = "default",
    dot: bool = True,
    pulse: bool = False
) -> None:
    """Render a status badge with optional dot indicator"""
    
    colors = {
        "default": ("#64748b", "rgba(100, 116, 139, 0.2)"),
        "primary": ("#6366f1", "rgba(99, 102, 241, 0.2)"),
        "success": ("#10b981", "rgba(16, 185, 129, 0.2)"),
        "warning": ("#f59e0b", "rgba(245, 158, 11, 0.2)"),
        "error": ("#ef4444", "rgba(239, 68, 68, 0.2)"),
        "info": ("#06b6d4", "rgba(6, 182, 212, 0.2)"),
    }
    
    text_color, bg_color = colors.get(variant, colors["default"])
    pulse_style = "animation: pulse 2s infinite;" if pulse else ""
    
    dot_html = ""
    if dot:
        dot_html = f'''
        <span style="
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: {text_color};
            {pulse_style}
        "></span>
        '''
    
    st.markdown(f'''
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        background: {bg_color};
        color: {text_color};
        font-size: 11px;
        font-weight: 600;
    ">
        {dot_html}
        {text}
    </span>
    ''', unsafe_allow_html=True)


def progress_badge(
    value: float,
    max_value: float = 100,
    label: Optional[str] = None,
    variant: str = "primary"
) -> None:
    """Render a mini progress badge"""
    
    percentage = min(100, (value / max_value) * 100)
    
    colors = {
        "primary": "#6366f1",
        "success": "#10b981",
        "warning": "#f59e0b",
        "error": "#ef4444",
    }
    
    color = colors.get(variant, colors["primary"])
    
    st.markdown(f'''
    <div style="
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: rgba(30, 41, 59, 0.5);
        border-radius: 8px;
    ">
        {f'<span style="font-size: 12px; color: #94a3b8;">{label}</span>' if label else ''}
        <div style="
            flex: 1;
            height: 6px;
            background: rgba(100, 116, 139, 0.2);
            border-radius: 3px;
            overflow: hidden;
        ">
            <div style="
                width: {percentage}%;
                height: 100%;
                background: {color};
                border-radius: 3px;
                transition: width 0.5s ease;
            "></div>
        </div>
        <span style="font-size: 11px; font-weight: 600; color: {color};">{percentage:.0f}%</span>
    </div>
    ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GLASS CARDS & CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════════

def glass_container(content: str, padding: str = "24px") -> None:
    """Render a glassmorphism container with content"""
    st.markdown(f'''
    <div style="
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 20px;
        padding: {padding};
        animation: fadeInUp 0.5s ease-out;
    ">
        {content}
    </div>
    ''', unsafe_allow_html=True)


def feature_card(
    title: str,
    description: str,
    icon: str,
    action_label: Optional[str] = None,
    action_key: Optional[str] = None,
    variant: str = "default"
) -> Optional[bool]:
    """
    Render a feature card with icon and optional action
    
    Returns: True if action button clicked, None otherwise
    """
    st.markdown(f'''
    <div class="feature-card" style="
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s ease;
        cursor: pointer;
    " onmouseover="this.style.transform='translateY(-4px)'; this.style.borderColor='rgba(99,102,241,0.4)';"
       onmouseout="this.style.transform='translateY(0)'; this.style.borderColor='rgba(148,163,184,0.1)';">
        <div style="
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.1));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            margin-bottom: 16px;
        ">{icon}</div>
        <h4 style="
            font-size: 16px;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 8px;
            background: none;
            -webkit-text-fill-color: #f8fafc;
        ">{title}</h4>
        <p style="
            font-size: 13px;
            color: #94a3b8;
            line-height: 1.5;
            margin: 0;
        ">{description}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    if action_label and action_key:
        return st.button(action_label, key=action_key, use_container_width=True)
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA DISPLAY COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def data_table_header(
    title: str,
    subtitle: Optional[str] = None,
    row_count: Optional[int] = None,
    actions: Optional[List[Dict[str, str]]] = None
) -> None:
    """Render a styled table header with metadata and actions"""
    
    count_badge = ""
    if row_count is not None:
        count_badge = f'''
        <span style="
            background: rgba(99, 102, 241, 0.2);
            color: #6366f1;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 12px;
        ">{row_count:,} rows</span>
        '''
    
    st.markdown(f'''
    <div style="
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 0;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 16px;
    ">
        <div style="display: flex; align-items: center;">
            <div>
                <h3 style="
                    font-size: 18px;
                    font-weight: 700;
                    color: #f8fafc;
                    margin: 0;
                    background: none;
                    -webkit-text-fill-color: #f8fafc;
                ">{title}{count_badge}</h3>
                {f'<p style="font-size: 12px; color: #64748b; margin: 4px 0 0 0;">{subtitle}</p>' if subtitle else ''}
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


def empty_state(
    title: str,
    description: str,
    icon: str = "📭",
    action_label: Optional[str] = None,
    action_key: Optional[str] = None
) -> Optional[bool]:
    """Render an empty state illustration with optional action"""
    
    st.markdown(f'''
    <div style="
        text-align: center;
        padding: 60px 40px;
        background: rgba(30, 41, 59, 0.5);
        border: 2px dashed rgba(148, 163, 184, 0.2);
        border-radius: 20px;
        animation: fadeInUp 0.5s ease-out;
    ">
        <div style="
            font-size: 64px;
            margin-bottom: 20px;
            animation: float 6s ease-in-out infinite;
        ">{icon}</div>
        <h3 style="
            font-size: 20px;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 8px;
            background: none;
            -webkit-text-fill-color: #f8fafc;
        ">{title}</h3>
        <p style="
            font-size: 14px;
            color: #94a3b8;
            max-width: 400px;
            margin: 0 auto 24px auto;
            line-height: 1.6;
        ">{description}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    if action_label and action_key:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            return st.button(action_label, key=action_key, type="primary", use_container_width=True)
    
    return None


def loading_skeleton(height: str = "200px", variant: str = "card") -> None:
    """Render a loading skeleton placeholder"""
    
    if variant == "card":
        st.markdown(f'''
        <div style="
            background: linear-gradient(90deg, rgba(30, 41, 59, 0.7) 25%, rgba(51, 65, 85, 0.5) 50%, rgba(30, 41, 59, 0.7) 75%);
            background-size: 200px 100%;
            animation: skeleton 1.5s ease-in-out infinite;
            border-radius: 16px;
            height: {height};
        "></div>
        ''', unsafe_allow_html=True)
    elif variant == "table":
        for _ in range(5):
            st.markdown(f'''
            <div style="
                background: linear-gradient(90deg, rgba(30, 41, 59, 0.7) 25%, rgba(51, 65, 85, 0.5) 50%, rgba(30, 41, 59, 0.7) 75%);
                background-size: 200px 100%;
                animation: skeleton 1.5s ease-in-out infinite;
                border-radius: 8px;
                height: 48px;
                margin-bottom: 8px;
            "></div>
            ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT & NOTIFICATION COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def toast_notification(
    message: str,
    variant: str = "info",
    dismissible: bool = True,
    icon: Optional[str] = None
) -> None:
    """Render a toast-style notification"""
    
    configs = {
        "success": {"bg": "rgba(16, 185, 129, 0.15)", "border": "#10b981", "icon": "✓"},
        "error": {"bg": "rgba(239, 68, 68, 0.15)", "border": "#ef4444", "icon": "✕"},
        "warning": {"bg": "rgba(245, 158, 11, 0.15)", "border": "#f59e0b", "icon": "⚠"},
        "info": {"bg": "rgba(6, 182, 212, 0.15)", "border": "#06b6d4", "icon": "ℹ"},
    }
    
    config = configs.get(variant, configs["info"])
    display_icon = icon or config["icon"]
    
    st.markdown(f'''
    <div style="
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px 20px;
        background: {config['bg']};
        border-left: 4px solid {config['border']};
        border-radius: 0 12px 12px 0;
        animation: fadeInRight 0.3s ease-out;
    ">
        <span style="font-size: 18px;">{display_icon}</span>
        <span style="flex: 1; font-size: 14px; color: #f8fafc;">{message}</span>
        {f'<button style="background: none; border: none; color: #94a3b8; cursor: pointer; padding: 4px;">✕</button>' if dismissible else ''}
    </div>
    ''', unsafe_allow_html=True)


def inline_alert(
    message: str,
    variant: str = "info",
    title: Optional[str] = None,
    expandable_details: Optional[str] = None
) -> None:
    """Render an inline alert message"""
    
    colors = {
        "info": ("#06b6d4", "rgba(6, 182, 212, 0.1)"),
        "success": ("#10b981", "rgba(16, 185, 129, 0.1)"),
        "warning": ("#f59e0b", "rgba(245, 158, 11, 0.1)"),
        "error": ("#ef4444", "rgba(239, 68, 68, 0.1)"),
    }
    
    accent, bg = colors.get(variant, colors["info"])
    
    st.markdown(f'''
    <div style="
        background: {bg};
        border: 1px solid {accent}30;
        border-left: 4px solid {accent};
        border-radius: 8px;
        padding: 16px 20px;
        animation: fadeInUp 0.3s ease-out;
    ">
        {f'<h4 style="font-size: 14px; font-weight: 600; color: {accent}; margin: 0 0 6px 0; background: none; -webkit-text-fill-color: {accent};">{title}</h4>' if title else ''}
        <p style="font-size: 13px; color: #e2e8f0; margin: 0; line-height: 1.5;">{message}</p>
    </div>
    ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP INDICATOR & PROGRESS
# ═══════════════════════════════════════════════════════════════════════════════

def step_indicator(
    steps: List[str],
    current_step: int,
    variant: str = "horizontal"
) -> None:
    """Render a multi-step progress indicator"""
    
    if variant == "horizontal":
        steps_html = ""
        for i, step in enumerate(steps):
            if i < current_step:
                status = "completed"
                bg = "#10b981"
                text_color = "#10b981"
            elif i == current_step:
                status = "current"
                bg = "linear-gradient(135deg, #6366f1, #8b5cf6)"
                text_color = "#6366f1"
            else:
                status = "pending"
                bg = "#475569"
                text_color = "#64748b"
            
            connector = ""
            if i < len(steps) - 1:
                connector_color = "#10b981" if i < current_step else "#475569"
                connector = f'''
                <div style="
                    flex: 1;
                    height: 2px;
                    background: {connector_color};
                    margin: 0 8px;
                "></div>
                '''
            
            steps_html += f'''
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div style="
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    background: {bg};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-size: 12px;
                    font-weight: 600;
                    {"box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);" if status == "current" else ""}
                ">{"✓" if status == "completed" else i + 1}</div>
                <span style="
                    font-size: 11px;
                    color: {text_color};
                    margin-top: 8px;
                    font-weight: {"600" if status != "pending" else "400"};
                ">{step}</span>
            </div>
            {connector}
            '''
        
        st.markdown(f'''
        <div style="
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            padding: 20px;
            background: rgba(30, 41, 59, 0.5);
            border-radius: 16px;
        ">
            {steps_html}
        </div>
        ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# AVATAR & USER COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def user_avatar(
    name: str,
    image_url: Optional[str] = None,
    size: str = "md",
    status: Optional[str] = None  # online, offline, away
) -> None:
    """Render a user avatar with optional status indicator"""
    
    sizes = {"sm": 32, "md": 40, "lg": 56, "xl": 80}
    px = sizes.get(size, 40)
    
    initials = "".join([n[0].upper() for n in name.split()[:2]])
    
    status_html = ""
    if status:
        status_colors = {"online": "#10b981", "offline": "#64748b", "away": "#f59e0b"}
        status_html = f'''
        <div style="
            position: absolute;
            bottom: 0;
            right: 0;
            width: {px // 4}px;
            height: {px // 4}px;
            border-radius: 50%;
            background: {status_colors.get(status, "#64748b")};
            border: 2px solid #1e293b;
        "></div>
        '''
    
    if image_url:
        avatar_content = f'<img src="{image_url}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 50%;" />'
    else:
        avatar_content = f'''
        <div style="
            width: 100%;
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            font-size: {px // 3}px;
            font-weight: 600;
            border-radius: 50%;
        ">{initials}</div>
        '''
    
    st.markdown(f'''
    <div style="
        position: relative;
        width: {px}px;
        height: {px}px;
        border-radius: 50%;
        overflow: visible;
    ">
        {avatar_content}
        {status_html}
    </div>
    ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ACTION BUTTON GROUP
# ═══════════════════════════════════════════════════════════════════════════════

def action_button_group(
    buttons: List[Dict[str, Any]],
    key_prefix: str = "action"
) -> Optional[str]:
    """
    Render a group of action buttons
    
    Args:
        buttons: List of dicts with 'label', 'icon', 'variant' keys
        key_prefix: Prefix for button keys
        
    Returns: The label of clicked button, or None
    """
    cols = st.columns(len(buttons))
    
    for i, (col, btn) in enumerate(zip(cols, buttons)):
        with col:
            if st.button(
                f"{btn.get('icon', '')}  {btn['label']}",
                key=f"{key_prefix}_{i}",
                type="primary" if btn.get('variant') == 'primary' else "secondary",
                use_container_width=True
            ):
                return btn['label']
    
    return None
