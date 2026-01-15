"""
Financial Analytics Suite - Settings Page
User management, preferences, and system configuration
"""

import streamlit as st
from typing import Dict


def get_theme_colors() -> Dict[str, str]:
    """Get current theme colors"""
    theme_mode = st.session_state.get('theme_mode', 'dark')
    if theme_mode == 'dark':
        return {
            'text': '#f8fafc', 'text_muted': '#64748b', 'text_secondary': '#94a3b8',
            'primary': '#6366f1', 'glass_bg': 'rgba(30, 41, 59, 0.7)',
            'glass_border': 'rgba(148, 163, 184, 0.1)', 'bg_card': '#1e293b',
            'success': '#10b981', 'warning': '#f59e0b', 'error': '#ef4444',
        }
    else:
        return {
            'text': '#0f172a', 'text_muted': '#64748b', 'text_secondary': '#475569',
            'primary': '#4f46e5', 'glass_bg': 'rgba(255, 255, 255, 0.8)',
            'glass_border': 'rgba(0, 0, 0, 0.06)', 'bg_card': '#ffffff',
            'success': '#059669', 'warning': '#d97706', 'error': '#dc2626',
        }


def render():
    """Render the Settings page"""
    c = get_theme_colors()
    
    st.title("⚙️ Settings")
    st.markdown(f"<p style='color: {c['text_secondary']};'>Manage your preferences and system configuration</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Profile", "🎨 Appearance", "🔐 Security", "⚙️ System"])
    
    with tab1:
        st.markdown("### Profile Settings")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style="
                width: 120px;
                height: 120px;
                border-radius: 50%;
                background: {c['primary']};
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 48px;
                color: white;
                margin: 0 auto 16px auto;
            ">👤</div>
            """, unsafe_allow_html=True)
            st.button("Change Avatar", use_container_width=True)
        
        with col2:
            st.text_input("Full Name", value="Guest User")
            st.text_input("Email", value="guest@example.com")
            st.selectbox("Role", ["Viewer", "Analyst", "Admin"], index=0)
            st.text_area("Bio", value="Financial analyst passionate about data-driven insights.")
    
    with tab2:
        st.markdown("### Appearance")
        
        st.markdown("#### Theme")
        theme_col = st.columns(2)
        with theme_col[0]:
            if st.button("🌙 Dark Mode", use_container_width=True, type="primary" if st.session_state.get('theme_mode') == 'dark' else "secondary"):
                st.session_state.theme_mode = 'dark'
                st.rerun()
        with theme_col[1]:
            if st.button("☀️ Light Mode", use_container_width=True, type="primary" if st.session_state.get('theme_mode') == 'light' else "secondary"):
                st.session_state.theme_mode = 'light'
                st.rerun()
        
        st.markdown("#### Display")
        st.selectbox("Density", ["Comfortable", "Compact", "Spacious"])
        st.selectbox("Font Size", ["Small", "Medium", "Large"])
        
        st.markdown("#### Charts")
        st.selectbox("Chart Theme", ["Default", "Viridis", "Plasma", "Inferno"])
        st.checkbox("Enable animations", value=True)
        st.checkbox("Show grid lines", value=True)
    
    with tab3:
        st.markdown("### Security")
        
        st.markdown("#### Password")
        st.text_input("Current Password", type="password")
        st.text_input("New Password", type="password")
        st.text_input("Confirm Password", type="password")
        st.button("Update Password")
        
        st.markdown("---")
        
        st.markdown("#### Two-Factor Authentication")
        st.checkbox("Enable 2FA", value=False)
        
        st.markdown("#### API Keys")
        st.code("sk-fas-xxxx-xxxx-xxxx", language="text")
        col1, col2 = st.columns(2)
        with col1:
            st.button("🔄 Regenerate", use_container_width=True)
        with col2:
            st.button("📋 Copy", use_container_width=True)
        
        st.markdown("#### Sessions")
        st.markdown(f"""
        <div style="padding: 12px; background: {c['bg_card']}; border-radius: 8px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="color: {c['text']};">Current Session (This device)</span>
                <span style="color: {c['success']};">Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### System Settings")
        
        st.markdown("#### Defaults")
        st.selectbox("Default Currency", ["USD", "EUR", "GBP", "JPY"])
        st.selectbox("Default Timezone", ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"])
        st.selectbox("Date Format", ["YYYY-MM-DD", "MM/DD/YYYY", "DD/MM/YYYY"])
        
        st.markdown("#### Data Retention")
        st.slider("Keep data for (days)", 30, 365, 90)
        st.checkbox("Auto-delete old exports", value=True)
        
        st.markdown("#### Compute Limits")
        st.number_input("Max concurrent jobs", 1, 10, 3)
        st.number_input("Job timeout (minutes)", 5, 60, 30)
        
        st.markdown("---")
        
        st.markdown("#### Danger Zone")
        st.warning("These actions are irreversible.")
        col1, col2 = st.columns(2)
        with col1:
            st.button("🗑️ Clear All Data", use_container_width=True)
        with col2:
            st.button("❌ Delete Account", use_container_width=True)


if __name__ == "__main__":
    render()
