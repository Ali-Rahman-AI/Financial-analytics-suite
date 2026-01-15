"""
Financial Analytics Suite - Design Tokens
Complete design system with spacing, colors, typography, and elevation
Following 2026 fintech standards: Stripe-like cleanliness with Bloomberg-like density option
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class ThemeMode(Enum):
    """Theme mode options"""
    DARK = "dark"
    LIGHT = "light"


@dataclass
class ColorPalette:
    """Color palette for theming"""
    # Primary colors
    primary_50: str = "#eef2ff"
    primary_100: str = "#e0e7ff"
    primary_200: str = "#c7d2fe"
    primary_300: str = "#a5b4fc"
    primary_400: str = "#818cf8"
    primary_500: str = "#6366f1"
    primary_600: str = "#4f46e5"
    primary_700: str = "#4338ca"
    primary_800: str = "#3730a3"
    primary_900: str = "#312e81"
    
    # Secondary (Violet)
    secondary_50: str = "#f5f3ff"
    secondary_100: str = "#ede9fe"
    secondary_200: str = "#ddd6fe"
    secondary_300: str = "#c4b5fd"
    secondary_400: str = "#a78bfa"
    secondary_500: str = "#8b5cf6"
    secondary_600: str = "#7c3aed"
    secondary_700: str = "#6d28d9"
    secondary_800: str = "#5b21b6"
    secondary_900: str = "#4c1d95"
    
    # Accent (Cyan)
    accent_50: str = "#ecfeff"
    accent_100: str = "#cffafe"
    accent_200: str = "#a5f3fc"
    accent_300: str = "#67e8f9"
    accent_400: str = "#22d3ee"
    accent_500: str = "#06b6d4"
    accent_600: str = "#0891b2"
    accent_700: str = "#0e7490"
    accent_800: str = "#155e75"
    accent_900: str = "#164e63"
    
    # Success (Emerald)
    success_50: str = "#ecfdf5"
    success_100: str = "#d1fae5"
    success_200: str = "#a7f3d0"
    success_300: str = "#6ee7b7"
    success_400: str = "#34d399"
    success_500: str = "#10b981"
    success_600: str = "#059669"
    success_700: str = "#047857"
    success_800: str = "#065f46"
    success_900: str = "#064e3b"
    
    # Warning (Amber)
    warning_50: str = "#fffbeb"
    warning_100: str = "#fef3c7"
    warning_200: str = "#fde68a"
    warning_300: str = "#fcd34d"
    warning_400: str = "#fbbf24"
    warning_500: str = "#f59e0b"
    warning_600: str = "#d97706"
    warning_700: str = "#b45309"
    warning_800: str = "#92400e"
    warning_900: str = "#78350f"
    
    # Error (Rose)
    error_50: str = "#fff1f2"
    error_100: str = "#ffe4e6"
    error_200: str = "#fecdd3"
    error_300: str = "#fda4af"
    error_400: str = "#fb7185"
    error_500: str = "#f43f5e"
    error_600: str = "#e11d48"
    error_700: str = "#be123c"
    error_800: str = "#9f1239"
    error_900: str = "#881337"
    
    # Neutral (Slate)
    neutral_0: str = "#ffffff"
    neutral_50: str = "#f8fafc"
    neutral_100: str = "#f1f5f9"
    neutral_200: str = "#e2e8f0"
    neutral_300: str = "#cbd5e1"
    neutral_400: str = "#94a3b8"
    neutral_500: str = "#64748b"
    neutral_600: str = "#475569"
    neutral_700: str = "#334155"
    neutral_800: str = "#1e293b"
    neutral_900: str = "#0f172a"
    neutral_950: str = "#020617"


@dataclass
class DarkThemeColors:
    """Dark theme specific colors"""
    # Backgrounds
    bg_base: str = "#030712"           # Base background (gray-950)
    bg_surface: str = "#0f172a"        # Surface (slate-900)
    bg_card: str = "#1e293b"           # Card (slate-800)
    bg_elevated: str = "#334155"       # Elevated (slate-700)
    bg_hover: str = "#475569"          # Hover state (slate-600)
    bg_active: str = "#1e40af"         # Active state
    bg_overlay: str = "rgba(0, 0, 0, 0.75)"
    
    # Glass effects
    glass_bg: str = "rgba(30, 41, 59, 0.7)"
    glass_border: str = "rgba(148, 163, 184, 0.1)"
    glass_backdrop: str = "blur(20px)"
    
    # Text
    text_primary: str = "#f8fafc"      # Primary text (slate-50)
    text_secondary: str = "#94a3b8"    # Secondary (slate-400)
    text_muted: str = "#64748b"        # Muted (slate-500)
    text_disabled: str = "#475569"     # Disabled (slate-600)
    text_inverse: str = "#020617"      # Inverse (slate-950)
    text_link: str = "#60a5fa"         # Link (blue-400)
    
    # Borders
    border_default: str = "#334155"    # Default (slate-700)
    border_light: str = "#475569"      # Light (slate-600)
    border_dark: str = "#1e293b"       # Dark (slate-800)
    border_focus: str = "#6366f1"      # Focus (indigo-500)
    
    # Shadows
    shadow_sm: str = "0 1px 2px 0 rgba(0, 0, 0, 0.3)"
    shadow_md: str = "0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -2px rgba(0, 0, 0, 0.3)"
    shadow_lg: str = "0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -4px rgba(0, 0, 0, 0.4)"
    shadow_xl: str = "0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.4)"
    shadow_glow: str = "0 0 40px rgba(99, 102, 241, 0.4)"


@dataclass
class LightThemeColors:
    """Light theme specific colors"""
    # Backgrounds
    bg_base: str = "#fafbfc"           # Base background
    bg_surface: str = "#ffffff"        # Surface
    bg_card: str = "#ffffff"           # Card
    bg_elevated: str = "#f1f5f9"       # Elevated
    bg_hover: str = "#e2e8f0"          # Hover state
    bg_active: str = "#dbeafe"         # Active state
    bg_overlay: str = "rgba(0, 0, 0, 0.5)"
    
    # Glass effects
    glass_bg: str = "rgba(255, 255, 255, 0.8)"
    glass_border: str = "rgba(0, 0, 0, 0.06)"
    glass_backdrop: str = "blur(20px)"
    
    # Text
    text_primary: str = "#0f172a"      # Primary text
    text_secondary: str = "#475569"    # Secondary
    text_muted: str = "#64748b"        # Muted
    text_disabled: str = "#94a3b8"     # Disabled
    text_inverse: str = "#f8fafc"      # Inverse
    text_link: str = "#2563eb"         # Link
    
    # Borders
    border_default: str = "#e2e8f0"    # Default
    border_light: str = "#f1f5f9"      # Light
    border_dark: str = "#cbd5e1"       # Dark
    border_focus: str = "#6366f1"      # Focus
    
    # Shadows
    shadow_sm: str = "0 1px 2px 0 rgba(0, 0, 0, 0.05)"
    shadow_md: str = "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1)"
    shadow_lg: str = "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1)"
    shadow_xl: str = "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)"
    shadow_glow: str = "0 0 40px rgba(99, 102, 241, 0.15)"


# ═══════════════════════════════════════════════════════════════════════════════
# SPACING SCALE (8px base unit)
# ═══════════════════════════════════════════════════════════════════════════════

SPACING = {
    "0": "0px",
    "0.5": "2px",
    "1": "4px",
    "1.5": "6px",
    "2": "8px",
    "2.5": "10px",
    "3": "12px",
    "3.5": "14px",
    "4": "16px",
    "5": "20px",
    "6": "24px",
    "7": "28px",
    "8": "32px",
    "9": "36px",
    "10": "40px",
    "11": "44px",
    "12": "48px",
    "14": "56px",
    "16": "64px",
    "20": "80px",
    "24": "96px",
    "28": "112px",
    "32": "128px",
    "36": "144px",
    "40": "160px",
    "44": "176px",
    "48": "192px",
    "52": "208px",
    "56": "224px",
    "60": "240px",
    "64": "256px",
    "72": "288px",
    "80": "320px",
    "96": "384px",
}


# ═══════════════════════════════════════════════════════════════════════════════
# TYPOGRAPHY SCALE
# ═══════════════════════════════════════════════════════════════════════════════

FONT_FAMILY = {
    "sans": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
    "mono": "'JetBrains Mono', 'Fira Code', 'SF Mono', Monaco, 'Cascadia Mono', monospace",
    "display": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
}

FONT_SIZE = {
    "2xs": "10px",
    "xs": "11px",
    "sm": "12px",
    "base": "14px",
    "md": "15px",
    "lg": "16px",
    "xl": "18px",
    "2xl": "20px",
    "3xl": "24px",
    "4xl": "30px",
    "5xl": "36px",
    "6xl": "48px",
    "7xl": "60px",
    "8xl": "72px",
    "9xl": "96px",
}

FONT_WEIGHT = {
    "thin": 100,
    "extralight": 200,
    "light": 300,
    "normal": 400,
    "medium": 500,
    "semibold": 600,
    "bold": 700,
    "extrabold": 800,
    "black": 900,
}

LINE_HEIGHT = {
    "none": 1,
    "tight": 1.25,
    "snug": 1.375,
    "normal": 1.5,
    "relaxed": 1.625,
    "loose": 2,
}

LETTER_SPACING = {
    "tighter": "-0.05em",
    "tight": "-0.025em",
    "normal": "0em",
    "wide": "0.025em",
    "wider": "0.05em",
    "widest": "0.1em",
}


# ═══════════════════════════════════════════════════════════════════════════════
# BORDER RADIUS
# ═══════════════════════════════════════════════════════════════════════════════

RADIUS = {
    "none": "0px",
    "sm": "4px",
    "default": "6px",
    "md": "8px",
    "lg": "12px",
    "xl": "16px",
    "2xl": "20px",
    "3xl": "24px",
    "4xl": "32px",
    "full": "9999px",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Z-INDEX (ELEVATION LEVELS)
# ═══════════════════════════════════════════════════════════════════════════════

Z_INDEX = {
    "hide": -1,
    "base": 0,
    "surface": 10,
    "dropdown": 50,
    "sticky": 100,
    "overlay": 200,
    "modal": 300,
    "popover": 400,
    "toast": 500,
    "tooltip": 600,
    "max": 9999,
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSITION & ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

TRANSITION = {
    "none": "none",
    "all": "all 0.15s ease",
    "fast": "all 0.1s ease",
    "normal": "all 0.2s ease",
    "slow": "all 0.3s ease",
    "slower": "all 0.5s ease",
    "bounce": "all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55)",
    "smooth": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
}

ANIMATION_DURATION = {
    "instant": "0ms",
    "fast": "100ms",
    "normal": "200ms",
    "slow": "300ms",
    "slower": "500ms",
    "slowest": "1000ms",
}


# ═══════════════════════════════════════════════════════════════════════════════
# BREAKPOINTS (RESPONSIVE)
# ═══════════════════════════════════════════════════════════════════════════════

BREAKPOINTS = {
    "xs": "320px",
    "sm": "640px",
    "md": "768px",
    "lg": "1024px",
    "xl": "1280px",
    "2xl": "1536px",
    "3xl": "1920px",
}


# ═══════════════════════════════════════════════════════════════════════════════
# CHART COLORS (Sequential palettes for data visualization)
# ═══════════════════════════════════════════════════════════════════════════════

CHART_COLORS = {
    "primary": ["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe", "#e0e7ff"],
    "secondary": ["#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe", "#ede9fe"],
    "success": ["#10b981", "#34d399", "#6ee7b7", "#a7f3d0", "#d1fae5"],
    "warning": ["#f59e0b", "#fbbf24", "#fcd34d", "#fde68a", "#fef3c7"],
    "error": ["#f43f5e", "#fb7185", "#fda4af", "#fecdd3", "#ffe4e6"],
    "categorical": [
        "#6366f1",  # Indigo
        "#22d3ee",  # Cyan
        "#f472b6",  # Pink
        "#fbbf24",  # Amber
        "#34d399",  # Emerald
        "#a78bfa",  # Violet
        "#fb7185",  # Rose
        "#38bdf8",  # Sky
        "#4ade80",  # Green
        "#f97316",  # Orange
    ],
    "diverging": [
        "#ef4444", "#f87171", "#fca5a5", "#fecaca", 
        "#f5f5f5",
        "#bbf7d0", "#86efac", "#4ade80", "#22c55e"
    ],
    "heatmap": [
        "#312e81", "#3730a3", "#4338ca", "#4f46e5", "#6366f1",
        "#818cf8", "#a5b4fc", "#c7d2fe", "#e0e7ff", "#eef2ff"
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# GRID SYSTEM (12-column)
# ═══════════════════════════════════════════════════════════════════════════════

GRID = {
    "columns": 12,
    "gutter": "24px",
    "margin": "32px",
    "container_max_width": {
        "sm": "640px",
        "md": "768px",
        "lg": "1024px",
        "xl": "1280px",
        "2xl": "1536px",
        "full": "100%",
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# ICON SIZES
# ═══════════════════════════════════════════════════════════════════════════════

ICON_SIZE = {
    "xs": "12px",
    "sm": "16px",
    "md": "20px",
    "lg": "24px",
    "xl": "32px",
    "2xl": "40px",
    "3xl": "48px",
}


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIENT PRESETS
# ═══════════════════════════════════════════════════════════════════════════════

GRADIENTS = {
    "primary": "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%)",
    "secondary": "linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%)",
    "accent": "linear-gradient(135deg, #22d3ee 0%, #06b6d4 100%)",
    "success": "linear-gradient(135deg, #10b981 0%, #059669 100%)",
    "warning": "linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%)",
    "error": "linear-gradient(135deg, #f43f5e 0%, #e11d48 100%)",
    "dark": "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)",
    "light": "linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)",
    "mesh_dark": """
        radial-gradient(at 40% 20%, hsla(228,100%,74%,0.15) 0px, transparent 50%),
        radial-gradient(at 80% 0%, hsla(275,100%,70%,0.1) 0px, transparent 50%),
        radial-gradient(at 0% 50%, hsla(339,100%,76%,0.08) 0px, transparent 50%),
        radial-gradient(at 80% 50%, hsla(189,100%,56%,0.08) 0px, transparent 50%),
        radial-gradient(at 0% 100%, hsla(228,100%,74%,0.1) 0px, transparent 50%)
    """,
    "mesh_light": """
        radial-gradient(at 40% 20%, hsla(228,100%,74%,0.08) 0px, transparent 50%),
        radial-gradient(at 80% 0%, hsla(275,100%,70%,0.06) 0px, transparent 50%),
        radial-gradient(at 0% 50%, hsla(339,100%,76%,0.04) 0px, transparent 50%),
        radial-gradient(at 80% 50%, hsla(189,100%,56%,0.04) 0px, transparent 50%)
    """,
}
