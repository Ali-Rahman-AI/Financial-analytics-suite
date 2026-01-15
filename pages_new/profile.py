"""
Financial Analytics Suite - Profile Page
Premium Digital Portfolio using App Theme System
"""

import streamlit as st
import base64
import os

from pages_new.theme_utils import get_theme_colors


def get_image_base64():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_path = os.path.join(project_root, "www", "mypic.jpeg")
    try:
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return None


def render():
    c = get_theme_colors()
    img_b64 = get_image_base64()
    user_img = f"data:image/jpeg;base64,{img_b64}" if img_b64 else "https://cdn-icons-png.flaticon.com/512/3135/3135768.png"
    
    # Inject Premium CSS using app theme colors
    st.markdown(f"""
    <style>
    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-50px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    @keyframes slideInRight {{
        from {{ opacity: 0; transform: translateX(50px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}
    
    @keyframes scaleIn {{
        from {{ opacity: 0; transform: scale(0.8); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    
    @keyframes shimmer {{
        0% {{ background-position: -1000px 0; }}
        100% {{ background-position: 1000px 0; }}
    }}
    
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes floating {{
        0%, 100% {{ transform: translateY(0) rotate(0deg); }}
        50% {{ transform: translateY(-20px) rotate(5deg); }}
    }}
    
    @keyframes progressFill {{
        from {{ width: 0; }}
    }}
    
    .profile-hero-section {{
        background: {c['gradient']};
        border-radius: 24px;
        padding: 50px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px {c['primary']}30;
        animation: fadeInUp 0.8s ease-out;
    }}
    
    .profile-hero-section::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 200%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }}
    
    .shape {{
        position: absolute;
        filter: blur(60px);
        z-index: 0;
        animation: floating 6s ease-in-out infinite;
        border-radius: 50%;
        opacity: 0.4;
        pointer-events: none;
    }}
    
    .shape-1 {{
        top: -20%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: {c['accent']};
    }}
    
    .shape-2 {{
        bottom: -20%;
        right: -10%;
        width: 250px;
        height: 250px;
        background: {c['secondary']};
        animation-delay: -3s;
    }}
    
    .profile-image-container {{
        position: relative;
        width: 220px;
        height: 220px;
        margin: 0 auto 25px;
        animation: scaleIn 0.8s ease-out 0.2s backwards;
    }}
    
    .profile-image {{
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        border: 6px solid rgba(255,255,255,0.3);
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        transition: transform 0.4s ease;
    }}
    
    .profile-image:hover {{
        transform: scale(1.05) rotate(5deg);
    }}
    
    .profile-image-container::after {{
        content: '';
        position: absolute;
        top: -10px;
        left: -10px;
        right: -10px;
        bottom: -10px;
        border-radius: 50%;
        background: linear-gradient(45deg, {c['accent']}, {c['secondary']}, {c['accent2']}, {c['accent']});
        background-size: 300% 300%;
        z-index: -1;
        opacity: 0.5;
        animation: floating 3s ease-in-out infinite;
    }}
    
    .profile-name {{
        font-size: 42px;
        font-weight: 800;
        color: white;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }}
    
    .profile-title {{
        font-size: 20px;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-bottom: 25px;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }}
    
    .contact-buttons {{
        text-align: center;
        position: relative;
        z-index: 1;
        margin-top: 25px;
    }}
    
    .profile-contact-btn {{
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: white;
        color: {c['primary']};
        padding: 14px 30px;
        border-radius: 999px;
        text-decoration: none;
        font-weight: 700;
        margin: 5px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }}
    
    .profile-contact-btn:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
    }}
    
    .section-card {{
        background: {c['bg_card']};
        backdrop-filter: blur(10px);
        border: 1px solid {c['border']};
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        animation: slideInLeft 0.8s ease-out backwards;
        transition: all 0.4s ease;
    }}
    
    .section-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 35px {c['primary']}20;
        border-color: {c['primary']};
    }}
    
    .section-title {{
        font-size: 24px;
        font-weight: 700;
        color: {c['text']};
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .info-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: {c['primary']}15;
        border: 1px solid {c['primary']}30;
        border-radius: 999px;
        padding: 10px 18px;
        margin: 5px;
        font-size: 14px;
        color: {c['text']};
        transition: all 0.3s ease;
    }}
    
    .info-badge:hover {{
        transform: scale(1.05);
        background: {c['primary']}25;
    }}
    
    .skill-item {{
        background: {c['primary']}15;
        border: 1px solid {c['primary']}20;
        border-radius: 12px;
        padding: 15px 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
        animation: slideInRight 0.6s ease-out backwards;
    }}
    
    .skill-item:hover {{
        transform: translateX(10px);
        background: {c['primary']}25;
        border-color: {c['primary']}40;
    }}
    
    .skill-name {{
        font-weight: 600;
        color: {c['text']};
        margin-bottom: 8px;
        font-size: 14px;
    }}
    
    .skill-bar {{
        height: 8px;
        background: {c['bg_elevated']};
        border-radius: 999px;
        overflow: hidden;
    }}
    
    .skill-progress {{
        height: 100%;
        background: {c['gradient']};
        border-radius: 999px;
        animation: progressFill 1.5s ease-out;
        box-shadow: 0 0 10px {c['primary']}50;
    }}
    
    .project-card {{
        background: {c['primary']}10;
        border: 1px solid {c['primary']}20;
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        transition: all 0.4s ease;
        animation: fadeInUp 0.8s ease-out backwards;
    }}
    
    .project-card:hover {{
        transform: translateY(-5px) translateX(5px);
        box-shadow: 0 20px 40px {c['primary']}20;
        border-color: {c['primary']}40;
    }}
    
    .project-title {{
        font-size: 18px;
        font-weight: 700;
        color: {c['primary']};
        margin-bottom: 10px;
    }}
    
    .project-desc {{
        color: {c['text_secondary']};
        line-height: 1.6;
        font-size: 14px;
    }}
    
    .cv-download-section {{
        background: linear-gradient(135deg, {c['success']}, #059669);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        color: white;
        margin: 30px 0;
        box-shadow: 0 15px 35px {c['success']}30;
        animation: fadeInUp 0.8s ease-out 0.4s backwards;
    }}
    
    .edu-item {{
        margin: 15px 0;
        padding: 15px;
        background: {c['primary']}10;
        border-radius: 12px;
        border-left: 4px solid {c['primary']};
    }}
    
    .edu-title {{
        color: {c['primary']};
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 8px;
    }}
    
    .edu-details {{
        color: {c['text_secondary']};
        font-size: 13px;
        line-height: 1.6;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown(f"""
    <div class="profile-hero-section">
        <div class="shape shape-1"></div>
        <div class="shape shape-2"></div>
        <div class="profile-image-container">
            <img src="{user_img}" class="profile-image" alt="Ali Rahman">
        </div>
        <div class="profile-name">Ali Rahman</div>
        <div class="profile-title">Data Scientist and Analytics Developer</div>
        <div class="contact-buttons">
            <a href="mailto:ali.m.rahman369@gmail.com" class="profile-contact-btn">Email Me</a>
            <a href="tel:+923223278356" class="profile-contact-btn">Call Me</a>
            <a href="https://linkedin.com/in/ali-rahman-ai" class="profile-contact-btn" target="_blank">LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Personal Info and Education - Two Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">Personal Information</div>
            <div>
                <span class="info-badge">Lahore, Pakistan</span>
                <span class="info-badge">March 3, 2005</span>
                <span class="info-badge">Pakistani</span>
                <span class="info-badge">ali.m.rahman369@gmail.com</span>
                <span class="info-badge">+92-322-3278356</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="section-card">
            <div class="section-title">Education</div>
            <div class="edu-item">
                <div class="edu-title">BS Statistics (Data Science)</div>
                <div class="edu-details">
                    COMSATS University Lahore<br>
                    2024 - 2028 (Expected)<br>
                    Current Semester: 4 | CGPA: 3.64/4.00
                </div>
            </div>
            <div class="edu-item">
                <div class="edu-title">Intermediate (Pre-Medical)</div>
                <div class="edu-details">
                    BISE Lahore (2023)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Skills Section
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Technical Skills</div>
    </div>
    """, unsafe_allow_html=True)
    
    skill_col1, skill_col2 = st.columns(2)
    
    with skill_col1:
        st.markdown(f"""
        <div class="skill-item" style="animation-delay: 0.1s;">
            <div class="skill-name">R Programming</div>
            <div class="skill-bar"><div class="skill-progress" style="width: 90%;"></div></div>
        </div>
        <div class="skill-item" style="animation-delay: 0.2s;">
            <div class="skill-name">Python</div>
            <div class="skill-bar"><div class="skill-progress" style="width: 85%;"></div></div>
        </div>
        <div class="skill-item" style="animation-delay: 0.3s;">
            <div class="skill-name">Data Visualization</div>
            <div class="skill-bar"><div class="skill-progress" style="width: 88%;"></div></div>
        </div>
        """, unsafe_allow_html=True)
    
    with skill_col2:
        st.markdown(f"""
        <div class="skill-item" style="animation-delay: 0.4s;">
            <div class="skill-name">SQL and Databases</div>
            <div class="skill-bar"><div class="skill-progress" style="width: 80%;"></div></div>
        </div>
        <div class="skill-item" style="animation-delay: 0.5s;">
            <div class="skill-name">Statistical Analysis</div>
            <div class="skill-bar"><div class="skill-progress" style="width: 92%;"></div></div>
        </div>
        <div class="skill-item" style="animation-delay: 0.6s;">
            <div class="skill-name">Machine Learning</div>
            <div class="skill-bar"><div class="skill-progress" style="width: 83%;"></div></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Featured Projects
    st.markdown(f"""
    <div class="section-card">
        <div class="section-title">Featured Projects</div>
        <div class="project-card" style="animation-delay: 0.1s;">
            <div class="project-title">Titanic Dataset Analysis</div>
            <div class="project-desc">Comprehensive exploratory data analysis and predictive modeling using Pandas, Matplotlib, and Scikit-learn. Achieved 85% accuracy in survival prediction using ensemble methods.</div>
        </div>
        <div class="project-card" style="animation-delay: 0.2s;">
            <div class="project-title">Pakistan Population Data Analysis</div>
            <div class="project-desc">Statistical analysis of demographic trends using Python and Seaborn. Created interactive visualizations showcasing population growth patterns across provinces.</div>
        </div>
        <div class="project-card" style="animation-delay: 0.3s;">
            <div class="project-title">COVID-19 Interactive Dashboard</div>
            <div class="project-desc">Built a comprehensive Power BI dashboard with DAX calculations for tracking pandemic metrics. Integrated multiple data sources and implemented real-time updates.</div>
        </div>
        <div class="project-card" style="animation-delay: 0.4s;">
            <div class="project-title">Financial Analytics Suite</div>
            <div class="project-desc">Enterprise-grade financial analytics platform with ML models, portfolio optimization, time series forecasting, and interactive dashboards using Streamlit and Python.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # CV Download Section
    st.markdown("""
    <div class="cv-download-section">
        <h2 style="margin-bottom: 15px; font-weight: 800;">Download My CV</h2>
        <p style="font-size: 16px; margin-bottom: 25px; opacity: 0.9;">Get a comprehensive overview of my skills, experience, and projects</p>
    </div>
    """, unsafe_allow_html=True)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resume_path = os.path.join(project_root, "www", "Ali_Rahman_Resume.html")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists(resume_path):
            with open(resume_path, "rb") as f:
                st.download_button(
                    "📄 Download My Resume",
                    f,
                    "Ali_Rahman_Resume.html",
                    "text/html",
                    use_container_width=True
                )
        else:
            st.info("Resume file not found in www folder.")
    
    # Professional Summary
    st.markdown(f"""
    <div class="section-card" style="animation-delay: 0.5s;">
        <div class="section-title">Professional Summary</div>
        <p style="color: {c['text_secondary']}; line-height: 1.8; font-size: 15px;">
            Passionate Data Science student with a strong foundation in statistical analysis, machine learning, and data visualization.
            Experienced in developing analytical solutions using R and Python, with a focus on financial analytics and predictive modeling.
            Demonstrated ability to transform complex datasets into actionable insights through interactive dashboards and comprehensive reports.
            Strong problem-solving skills combined with effective communication abilities to bridge technical and business stakeholders.
            Currently seeking opportunities to apply data science expertise to solve real-world challenges in finance and technology sectors.
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    render()
