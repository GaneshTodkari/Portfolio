import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Ganesh Todkari | Data & Business Analytics Portfolio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

current_dir = Path(__file__).resolve().parent
resume_path = current_dir / "assets" / "resume.pdf"
profile_pic_path = current_dir / "assets" / "profile.png"

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    :root {
        --accent: #1f77b4;
        --accent-soft: #4a90e2;
        --text-strong: var(--text-color);
        --card-border: rgba(49, 51, 63, 0.16);
        --card-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
        --card-shadow-hover: 0 14px 26px rgba(15, 23, 42, 0.12);
    }

    [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 1.1rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--card-border);
    }

    [data-testid="stSidebarNav"] a {
        font-weight: 500;
    }

    [data-testid="stSidebarNav"] a[aria-current="page"] {
        color: var(--accent) !important;
        font-weight: 700;
    }

    h1, h2, h3, h4, .main-header, .section-header, .card-title {
        color: var(--text-strong);
        letter-spacing: -0.02em;
    }

    p {
        line-height: 1.65;
    }

    .hero-wrap {
        margin: 0.5rem 0 1.6rem 0;
    }

    .hero-kicker {
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.8rem;
    }

    .hero-subtitle {
        font-size: 1.12rem;
        opacity: 0.86;
        margin-bottom: 0;
        max-width: 900px;
    }

    .section-header {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 2.4rem 0 1.2rem 0;
        padding-bottom: 0.45rem;
        border-bottom: 2px solid var(--card-border);
    }

    .card,
    .project-card {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        box-shadow: var(--card-shadow);
    }

    .card {
        padding: 1.4rem 1.5rem;
    }

    .project-card {
        padding: 1.4rem;
        height: 100%;
        transition: 0.25s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .project-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--card-shadow-hover);
        border-color: rgba(31, 119, 180, 0.38);
    }

    .card-subtitle {
        color: var(--accent);
        font-size: 0.86rem;
        font-weight: 600;
        margin-bottom: 0.55rem;
    }

    .metric-box {
        border: 1px solid var(--card-border);
        border-left: 3px solid var(--accent-soft);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0.9rem 0 0.8rem 0;
        font-size: 0.9rem;
        opacity: 0.95;
    }

    .skill-tag {
        background: rgba(31, 119, 180, 0.1);
        color: var(--accent);
        border: 1px solid rgba(31, 119, 180, 0.28);
        border-radius: 999px;
        display: inline-block;
        padding: 0.28rem 0.78rem;
        margin: 0.2rem;
        font-size: 0.74rem;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        padding-bottom: 0.4rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        border: 1px solid var(--card-border);
        font-weight: 600;
        height: 44px;
        padding: 0 18px;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--accent) !important;
        color: #fff !important;
        border: none !important;
    }

    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] svg {
        color: #fff !important;
        fill: #fff !important;
    }

    .stButton button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }

    .profile-name {
        text-align: center;
        margin: 1.1rem 0 0.2rem 0;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .profile-role {
        text-align: center;
        margin-top: 0;
        opacity: 0.8;
        font-size: 0.95rem;
        font-weight: 500;
    }

    [data-testid="stSidebar"] img {
        border-radius: 50%;
        border: 3px solid rgba(31, 119, 180, 0.65);
        padding: 3px;
    }

    .footer-wrap {
        text-align: center;
        padding: 1.6rem 0 0.7rem 0;
        opacity: 0.92;
    }

    .footer-links {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.9rem 1.3rem;
        margin: 0.95rem 0 0.7rem 0;
    }

    .footer-link {
        color: var(--accent);
        text-decoration: none;
        font-weight: 600;
    }

    @media (max-width: 900px) {
        .main-header { font-size: 2.05rem; }
        .section-header { font-size: 1.45rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.write("")
    if profile_pic_path.exists():
        _, center_col, _ = st.columns([0.5, 2, 0.5])
        with center_col:
            st.image(str(profile_pic_path), width=180)
    else:
        st.markdown(
            """
            <div style="width:160px;height:160px;border-radius:50%;
                        background:linear-gradient(135deg,#1f77b4,#4a90e2);
                        display:flex;align-items:center;justify-content:center;
                        color:#fff;font-size:2.8rem;margin:0 auto;">
                GT
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<h1 class='profile-name'>Ganesh Todkari</h1>", unsafe_allow_html=True)
    st.markdown("<p class='profile-role'>MBA-IT Candidate | Data and Business Analytics</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Contact")
    st.markdown(
        """
        <div style="font-size:0.92rem; line-height:1.7;">
            <div><a href="mailto:ganesh697todkari@gmail.com" style="color:inherit; text-decoration:none;">Email</a></div>
            <div><a href="https://linkedin.com/in/GaneshTodkari" style="color:inherit; text-decoration:none;">LinkedIn</a></div>
            <div><a href="https://github.com/GaneshTodkari" style="color:inherit; text-decoration:none;">GitHub</a></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

hero_col, skills_col = st.columns([2, 1], gap="large")

with hero_col:
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-kicker">Portfolio Overview</div>
            <div class="main-header">Turning Data Into Measurable Business Outcomes</div>
            <p class="hero-subtitle">
                I build analytics and machine learning solutions that improve decisions,
                streamline operations, and create measurable impact.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <h3 style="margin-top:0; margin-bottom:0.8rem;">Professional Summary</h3>
            <p>
                I am a Computer Science graduate currently pursuing an MBA in IT, with a focus on
                combining technical depth with business execution. My work spans data science,
                business intelligence, and process optimization.
            </p>
            <p>
                I specialize in translating complex datasets into clear decisions through
                forecasting, fraud analytics, and operational workflow redesign.
            </p>
            <ul style="line-height:1.7; margin-bottom:0.4rem;">
                <li><strong>Analytics Delivery:</strong> Built end-to-end solutions using Python, SQL, and Power BI.</li>
                <li><strong>Business Impact:</strong> Designed automation initiatives that reduced cycle time and manual effort.</li>
                <li><strong>Collaboration:</strong> Worked across technical and non-technical stakeholders to align outcomes with business goals.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with skills_col:
    st.markdown("### Core Competencies")
    st.markdown("**Data Science and Machine Learning**")
    st.markdown(
        """
        <span class='skill-tag'>Python</span>
        <span class='skill-tag'>XGBoost</span>
        <span class='skill-tag'>Scikit-Learn</span>
        <span class='skill-tag'>Random Forest</span>
        <span class='skill-tag'>NLP</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("**Analytics and BI**")
    st.markdown(
        """
        <span class='skill-tag'>SQL</span>
        <span class='skill-tag'>Power BI</span>
        <span class='skill-tag'>Pandas</span>
        <span class='skill-tag'>NumPy</span>
        <span class='skill-tag'>EDA</span>
        <span class='skill-tag'>ETL</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("**Business Analysis**")
    st.markdown(
        """
        <span class='skill-tag'>Requirement Analysis</span>
        <span class='skill-tag'>BPMN</span>
        <span class='skill-tag'>Risk Analytics</span>
        <span class='skill-tag'>Stakeholder Management</span>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    if resume_path.exists():
        with open(resume_path, "rb") as pdf_file:
            st.download_button(
                label="Download Resume",
                data=pdf_file,
                file_name="Ganesh_Todkari_Resume.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )

st.markdown('<div class="section-header">Featured Projects</div>', unsafe_allow_html=True)

tab_ds, tab_da, tab_ba = st.tabs(
    ["Data Science and ML", "Data Analytics and BI", "Business Analysis and Strategy"]
)

with tab_ds:
    ds_col1, ds_col2, ds_col3 = st.columns(3, gap="medium")
    projects_ds = [
        {
            "title": "Retail Demand Forecasting",
            "subtitle": "XGBoost | Time Series Modeling",
            "description": "Developed a forecasting pipeline for store-level demand prediction to improve replenishment planning.",
            "metric_label": "Key Result",
            "metric_value": "R²: 0.85 with 15% lower forecast error",
            "skills": ["Python", "XGBoost", "Time Series", "Feature Engineering"],
            "link": "pages/1_Retail_Demand_Forecasting.py",
        },
        {
            "title": "Credit Card Fraud Detection",
            "subtitle": "Anomaly Detection | Imbalanced Learning",
            "description": "Implemented a high-precision fraud detection workflow for low-incidence transaction monitoring.",
            "metric_label": "Key Result",
            "metric_value": "AUC-ROC: 0.998",
            "skills": ["Scikit-Learn", "SMOTE", "XGBClassifier", "Risk Analytics"],
            "link": "pages/2_Credit_Card_Security_Analysis.py",
        },
        {
            "title": "House Price Prediction Engine",
            "subtitle": "Regression | Spatial Features",
            "description": "Built a property valuation model using location intelligence and structured feature engineering.",
            "metric_label": "Key Result",
            "metric_value": "Model accuracy: 78%",
            "skills": ["Python", "K-Means", "Predictive Modeling", "EDA"],
            "link": "pages/5_House_Price.py",
        },
        {
            "title": "Resume Match System",
            "subtitle": "NLP | Semantic + Skill Matching",
            "description": "Production-style resume-to-JD matching with semantic, skill, and experience scoring for hiring-fit analysis.",
            "metric_label": "Model Output",
            "metric_value": "Final score with matched and missing skill breakdown",
            "skills": ["NLP", "Semantic Similarity", "Skill Extraction", "Information Retrieval"],
            "link": "pages/6_Resume_Match_System.py",
        },
    ]

    for i, project in enumerate(projects_ds):
        with [ds_col1, ds_col2, ds_col3][i % 3]:
            st.markdown(
                f"""
                <div class='project-card'>
                    <div>
                        <h4 class='card-title' style='margin-top:0; margin-bottom:0.2rem;'>{project['title']}</h4>
                        <p class='card-subtitle'>{project['subtitle']}</p>
                        <p style='font-size:0.94rem;'>{project['description']}</p>
                        <div class='metric-box'><strong>{project['metric_label']}:</strong> {project['metric_value']}</div>
                    </div>
                    <div>
                        {"".join([f"<span class='skill-tag'>{skill}</span>" for skill in project['skills']])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            if st.button("Open Case Study", key=f"btn_ds_{i}"):
                st.switch_page(project["link"])

with tab_da:
    da_col1, da_col2 = st.columns(2, gap="medium")
    projects_da = [
        {
            "title": "Insurance Fraud Intelligence Dashboard",
            "subtitle": "Power BI | Risk Monitoring",
            "description": "Designed an interactive dashboard to surface fraud patterns, investigator priorities, and risk segments.",
            "metric_label": "Business Impact",
            "metric_value": "Investigation workflow accelerated by 60%",
            "skills": ["Power BI", "DAX", "SQL", "Data Modeling"],
            "link": "pages/3_Insurance_Fraud_Claim_Analysis.py",
        },
        {
            "title": "Retail Sales and Inventory Analytics",
            "subtitle": "SQL | Business Intelligence",
            "description": "Analyzed regional and product-level trends to support inventory optimization and margin planning.",
            "metric_label": "Business Impact",
            "metric_value": "Identified 25% potential inventory cost reduction",
            "skills": ["SQL", "Power BI", "Excel", "Strategic Analysis"],
            "link": "pages/4_Contoso_Retail.py",
        },
    ]

    for i, project in enumerate(projects_da):
        with [da_col1, da_col2][i]:
            st.markdown(
                f"""
                <div class='project-card'>
                    <div>
                        <h4 class='card-title' style='margin-top:0; margin-bottom:0.2rem;'>{project['title']}</h4>
                        <p class='card-subtitle'>{project['subtitle']}</p>
                        <p style='font-size:0.94rem;'>{project['description']}</p>
                        <div class='metric-box'><strong>{project['metric_label']}:</strong> {project['metric_value']}</div>
                    </div>
                    <div>
                        {"".join([f"<span class='skill-tag'>{skill}</span>" for skill in project['skills']])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            if st.button("Open Dashboard", key=f"btn_da_{i}"):
                st.switch_page(project["link"])

with tab_ba:
    ba_col1, ba_col2 = st.columns(2, gap="medium")
    projects_ba = [
        {
            "title": "VybeRiders Process Automation",
            "subtitle": "BPMN | Service Operations",
            "description": "Mapped the checkout process and redesigned billing operations to eliminate manual handoffs.",
            "metric_label": "Business Impact",
            "metric_value": "Checkout time reduced from 10 minutes to 15 seconds",
            "skills": ["Process Mapping", "Requirement Gathering", "Automation Design"],
            "link": None,
        }
    ]

    for i, project in enumerate(projects_ba):
        with [ba_col1, ba_col2][i]:
            st.markdown(
                f"""
                <div class='project-card'>
                    <div>
                        <h4 class='card-title' style='margin-top:0; margin-bottom:0.2rem;'>{project['title']}</h4>
                        <p class='card-subtitle'>{project['subtitle']}</p>
                        <p style='font-size:0.94rem;'>{project['description']}</p>
                        <div class='metric-box'><strong>{project['metric_label']}:</strong> {project['metric_value']}</div>
                    </div>
                    <div>
                        {"".join([f"<span class='skill-tag'>{skill}</span>" for skill in project['skills']])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            if project.get("link"):
                if st.button("Open Details", key=f"btn_ba_{i}"):
                    st.switch_page(project["link"])
            else:
                st.button("Detailed Case Study Coming Soon", key=f"btn_ba_{i}", disabled=True)

st.markdown("---")
st.markdown(
    """
    <div class="footer-wrap">
        <h4 style="margin-bottom:0.25rem;">Open to Analytics and Data Science Opportunities</h4>
        <p style="margin-bottom:0.35rem;">Interested in roles focused on business impact, decision science, and automation.</p>
        <div class="footer-links">
            <a href="mailto:ganesh697todkari@gmail.com" class="footer-link">Email</a>
            <a href="https://linkedin.com/in/GaneshTodkari" class="footer-link">LinkedIn</a>
            <a href="https://github.com/GaneshTodkari" class="footer-link">GitHub</a>
        </div>
        <p style="font-size:0.75rem; opacity:0.68;">Copyright 2026 Ganesh Todkari</p>
    </div>
    """,
    unsafe_allow_html=True,
)
