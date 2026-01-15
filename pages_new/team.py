"""
Financial Analytics Suite - Team Page
Premium team showcase with 2026 fintech design
"""

import streamlit as st
from typing import Dict
import base64
import os

# Import theme colors from central theme_utils
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
    
    # CSS
    st.markdown(f"""
    <style>
    @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    @keyframes float {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-8px); }} }}
    @keyframes shimmer {{ 0% {{ background-position: -500px 0; }} 100% {{ background-position: 500px 0; }} }}
    @keyframes gradientFlow {{ 0% {{ background-position: 0% 50%; }} 50% {{ background-position: 100% 50%; }} 100% {{ background-position: 0% 50%; }} }}
    @keyframes pulse {{ 0%,100% {{ box-shadow: 0 0 20px {c['primary']}40; }} 50% {{ box-shadow: 0 0 35px {c['primary']}60; }} }}
    
    .page-header {{ text-align: center; margin-bottom: 40px; animation: fadeInUp 0.6s ease; }}
    .page-title {{ font-size: 42px; font-weight: 800; background: {c['gradient']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }}
    .page-subtitle {{ color: {c['text_secondary']}; font-size: 16px; }}
    
    .member-card {{ background: {c['glass_bg']}; backdrop-filter: blur(20px); border: 1px solid {c['glass_border']}; border-radius: 24px; padding: 35px; margin-bottom: 30px; animation: fadeInUp 0.6s ease; transition: all 0.4s ease; position: relative; overflow: hidden; }}
    .member-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: {c['gradient']}; }}
    .member-card:hover {{ transform: translateY(-6px); box-shadow: 0 25px 50px {c['primary']}20; border-color: {c['primary']}50; }}
    
    .member-header {{ display: flex; align-items: center; gap: 24px; margin-bottom: 24px; }}
    .member-avatar {{ width: 100px; height: 100px; border-radius: 20px; object-fit: cover; border: 3px solid {c['primary']}40; box-shadow: 0 10px 30px rgba(0,0,0,0.2); animation: pulse 3s infinite; }}
    .member-info h2 {{ font-size: 26px; font-weight: 700; color: {c['text']}; margin-bottom: 6px; }}
    .member-info p {{ color: {c['primary']}; font-weight: 600; font-size: 14px; }}
    
    .role-badge {{ display: inline-block; background: {c['gradient']}; color: white; padding: 8px 18px; border-radius: 999px; font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 10px; }}
    
    .details-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .detail-item {{ background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 14px; padding: 18px; transition: all 0.3s; }}
    .detail-item:hover {{ border-color: {c['primary']}; transform: translateY(-2px); }}
    .detail-label {{ color: {c['text_muted']}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }}
    .detail-value {{ color: {c['text']}; font-weight: 600; font-size: 14px; }}
    
    .section-title {{ font-size: 18px; font-weight: 700; color: {c['text']}; margin: 24px 0 16px; display: flex; align-items: center; gap: 10px; }}
    .section-title::before {{ content: ''; width: 4px; height: 20px; background: {c['gradient']}; border-radius: 2px; }}
    
    .contrib-list {{ list-style: none; padding: 0; margin: 0; }}
    .contrib-list li {{ background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; color: {c['text_secondary']}; font-size: 14px; transition: all 0.3s; display: flex; align-items: center; gap: 10px; }}
    .contrib-list li:hover {{ border-color: {c['primary']}; transform: translateX(6px); background: {c['primary']}08; }}
    .contrib-list li::before {{ content: '✓'; color: {c['success']}; font-weight: 700; }}
    
    .timeline-card {{ background: {c['glass_bg']}; backdrop-filter: blur(16px); border: 1px solid {c['glass_border']}; border-radius: 20px; padding: 32px; margin-bottom: 24px; }}
    .timeline-item {{ display: flex; gap: 16px; margin-bottom: 16px; }}
    .timeline-dot {{ width: 12px; height: 12px; background: {c['gradient']}; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }}
    .timeline-content {{ flex: 1; }}
    .timeline-date {{ color: {c['primary']}; font-weight: 600; font-size: 13px; margin-bottom: 4px; }}
    .timeline-text {{ color: {c['text_secondary']}; font-size: 14px; }}
    
    .ack-card {{ background: linear-gradient(135deg, {c['success']}15, {c['success']}05); border: 1px solid {c['success']}30; border-radius: 18px; padding: 28px; margin: 30px 0; text-align: center; }}
    .ack-title {{ color: {c['success']}; font-size: 20px; font-weight: 700; margin-bottom: 12px; }}
    .ack-text {{ color: {c['text_secondary']}; font-size: 14px; line-height: 1.7; }}
    
    .cert-card {{ background: {c['bg_card']}; border: 2px solid {c['primary']}30; border-radius: 18px; padding: 28px; text-align: center; }}
    .cert-icon {{ font-size: 48px; margin-bottom: 16px; animation: float 3s ease-in-out infinite; }}
    .cert-title {{ color: {c['text']}; font-size: 18px; font-weight: 700; margin-bottom: 10px; }}
    .cert-text {{ color: {c['text_secondary']}; font-size: 13px; line-height: 1.6; }}
    </style>
    """, unsafe_allow_html=True)
    
    # Page Header
    st.markdown("""
    <div class="page-header">
        <div class="page-title">👥 Project Team</div>
        <div class="page-subtitle">Meet the talented individuals behind the Financial Analytics Suite</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Supervisor Card
    st.markdown(f"""
    <div class="member-card">
        <div class="member-header">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" class="member-avatar" alt="Supervisor">
            <div class="member-info">
                <h2>Dr. Muhammad Farooq</h2>
                <p>🎓 Professor & Research Supervisor</p>
                <span class="role-badge">👑 Project Supervisor</span>
            </div>
        </div>
        <div class="details-grid">
            <div class="detail-item"><div class="detail-label">Position</div><div class="detail-value">Professor, Department of Statistics</div></div>
            <div class="detail-item"><div class="detail-label">Institution</div><div class="detail-value">COMSATS University Lahore</div></div>
            <div class="detail-item"><div class="detail-label">Expertise</div><div class="detail-value">Statistical Modeling, Financial Research</div></div>
            <div class="detail-item"><div class="detail-label">Email</div><div class="detail-value">muhammadfarooq@cuilahore.edu.pk</div></div>
        </div>
        <div class="section-title">🏆 Supervision Role</div>
        <ul class="contrib-list">
            <li>Provided guidance on financial analytics methodology</li>
            <li>Reviewed statistical models and validation approaches</li>
            <li>Ensured financial relevance and practical applicability</li>
            <li>Mentored in research design and implementation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Lead Developer Card
    st.markdown(f"""
    <div class="member-card">
        <div class="member-header">
            <img src="{user_img}" class="member-avatar" alt="Developer">
            <div class="member-info">
                <h2>Ali Rahman</h2>
                <p>👨‍💻 Data Scientist & Developer</p>
                <span class="role-badge">💻 Lead Developer</span>
            </div>
        </div>
        <div class="details-grid">
            <div class="detail-item"><div class="detail-label">Degree</div><div class="detail-value">BS Statistics (Data Science)</div></div>
            <div class="detail-item"><div class="detail-label">Institution</div><div class="detail-value">COMSATS University Lahore</div></div>
            <div class="detail-item"><div class="detail-label">Semester</div><div class="detail-value">4th Semester • CGPA: 3.64</div></div>
            <div class="detail-item"><div class="detail-label">Email</div><div class="detail-value">ali.m.rahman369@gmail.com</div></div>
        </div>
        <div class="section-title">💻 Development Responsibilities</div>
        <ul class="contrib-list">
            <li>Full-stack development of the Financial Analytics Suite</li>
            <li>Implementation of statistical algorithms and ML models</li>
            <li>User interface design and UX optimization</li>
            <li>Data preprocessing and cleaning module development</li>
            <li>Financial validation and testing of all modules</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Timeline & Objectives
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="timeline-card">
            <div class="section-title">📅 Project Timeline</div>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-content"><div class="timeline-date">Sep 2025</div><div class="timeline-text">Requirements analysis and planning</div></div></div>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-content"><div class="timeline-date">Oct 2025</div><div class="timeline-text">Core module development</div></div></div>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-content"><div class="timeline-date">Nov 2025</div><div class="timeline-text">Advanced analytics implementation</div></div></div>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-content"><div class="timeline-date">Dec 2025</div><div class="timeline-text">Testing and validation</div></div></div>
            <div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-content"><div class="timeline-date">Jan 2026</div><div class="timeline-text">Deployment and documentation</div></div></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="timeline-card">
            <div class="section-title">🎯 Project Objectives</div>
            <ul class="contrib-list">
                <li>Develop accessible platform for financial data analysis</li>
                <li>Integrate statistical methods with financial relevance</li>
                <li>Create educational tool for financial researchers</li>
                <li>Implement robust validation for financial applications</li>
                <li>Build enterprise-grade analytics capabilities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Acknowledgements
    st.markdown("""
    <div class="ack-card">
        <div class="ack-title">❤️ Acknowledgements</div>
        <div class="ack-text">
            Special thanks to Dr. Muhammad Farooq for invaluable guidance and mentorship throughout this project.<br>
            This project represents the culmination of academic learning and practical application in financial data science.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Certification
    st.markdown(f"""
    <div class="cert-card">
        <div class="cert-icon">📜</div>
        <div class="cert-title">Project Certification</div>
        <div class="cert-text">
            This project has been developed as part of academic coursework at COMSATS University Lahore.<br>
            All methodologies and implementations have been reviewed for financial analysis and statistical validity.
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    render()
