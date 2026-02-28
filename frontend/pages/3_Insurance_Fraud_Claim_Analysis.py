from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Insurance Fraud Claim Analysis", page_icon="📊", layout="wide")

current_dir = Path(__file__).resolve().parent
assets_dir = (current_dir / ".." / "assets" / "fraud_car").resolve()
pbix_path = assets_dir / "Fraud Claim Analysis.pbix"


def render_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.error(f"Missing file: {path.name}")


st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 1.1rem; }
    .page-kicker { font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.08em; color: #1f77b4; font-weight: 700; }
    .panel {
        background: var(--secondary-background-color);
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        padding: 1rem 1.1rem;
        margin-bottom: 0.9rem;
    }
    .stDownloadButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='page-kicker'>Business Intelligence Case Study</div>", unsafe_allow_html=True)
st.title("Insurance Fraud Claim Analysis Dashboard")
st.markdown(
    """
    <div class="panel">
      <strong>Project Objective</strong><br>
      Designed a Power BI dashboard to convert model outputs into operational intelligence,
      helping investigators prioritize suspicious claims more efficiently.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Dashboard Views")
tab1, tab2, tab3 = st.tabs(["Executive Summary", "Demographics and Policy", "Evidence Analysis"])

with tab1:
    render_image(assets_dir / "1.png", "Executive performance and fraud overview")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        render_image(assets_dir / "gender.png", "Risk distribution by gender segment")
    with c2:
        render_image(assets_dir / "Age.png", "Age group claim-risk distribution")

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        render_image(assets_dir / "2.png", "Claim evidence trend view")
        render_image(assets_dir / "4.png", "Policy and claim cross-analysis")
    with c2:
        render_image(assets_dir / "3.png", "Fraud indicators by category")
        render_image(assets_dir / "5.png", "Witness and police report discrepancy view")

st.divider()

left, right = st.columns([2, 1], gap="large")
with left:
    st.info(
        "This dashboard was built in Power BI Desktop. Download the PBIX file to inspect data "
        "relationships, DAX measures, and interactive filters."
    )
with right:
    if pbix_path.exists():
        with open(pbix_path, "rb") as file_obj:
            st.download_button(
                label="Download PBIX Source",
                data=file_obj,
                file_name="Fraud_Claim_Analysis.pbix",
                mime="application/octet-stream",
                use_container_width=True,
            )
    else:
        st.error("PBIX source file not found in assets/fraud_car.")
