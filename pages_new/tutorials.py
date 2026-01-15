"""
Financial Analytics Suite - Video Player Page
Dynamic video player that loads content from user-provided URLs
"""

import streamlit as st
from pages_new.theme_utils import get_theme_colors, inject_premium_styles

def render():
    """Render the Video Player page"""
    c = get_theme_colors()
    inject_premium_styles()
    
    # Header
    st.markdown(f"""
    <div style="padding: 20px 0; animation: slideInUp 0.5s ease-out;">
        <h1 class="glass-header" style="font-size: 36px; margin-bottom: 10px;">🎥 Video Player</h1>
        <p style="color: {c['text_secondary']}; font-size: 16px;">Watch tutorials or market updates by pasting a video link below.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # URL Input Section
    st.markdown(f"""
    <div style="background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 12px; padding: 20px; margin-bottom: 30px;">
        <h4 style="color: {c['text']}; margin-bottom: 10px;">🔗 Video Source</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for input and button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input for URL
        video_url = st.text_input(
            "Paste Video URL", 
            placeholder="https://www.youtube.com/watch?v=...", 
            label_visibility="collapsed",
            key="custom_video_url"
        )
    
    with col2:
        # Load Button
        if st.button("▶️ Load Video", type="primary", use_container_width=True):
            if video_url:
                st.session_state.active_video_url = video_url
            else:
                st.warning("Please paste a URL first.")

    # Video Display Section
    current_url = st.session_state.get('active_video_url')
    
    if current_url:
        st.markdown(f"""
        <div style="margin-top: 20px; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
        """, unsafe_allow_html=True)
        
        try:
            st.video(current_url)
            
            st.success(f"✅ Playing: {current_url}")
            
        except Exception as e:
            st.error(f"❌ Could not load video. Error: {str(e)}")
            st.info("Supported formats: YouTube, Vimeo, or direct MP4/Link files.")
            
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Empty state / Placeholder
        st.markdown(f"""
        <div style="
            margin-top: 20px; 
            border: 2px dashed {c['border']}; 
            border-radius: 16px; 
            padding: 60px; 
            text-align: center;
            background: {c['bg_elevated']}40;
        ">
            <div style="font-size: 48px; margin-bottom: 20px; opacity: 0.5;">📺</div>
            <h3 style="color: {c['text_muted']}; font-weight: 500;">No video loaded</h3>
            <p style="color: {c['text_secondary']}; font-size: 14px;">Paste a URL above to start watching.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    render()
