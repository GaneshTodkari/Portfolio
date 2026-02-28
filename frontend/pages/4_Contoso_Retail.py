from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Contoso Retail Analysis", page_icon="🏬", layout="wide")

current_dir = Path(__file__).resolve().parent
assets_dir = (current_dir / ".." / "assets" / "contoso").resolve()
pbix_path = assets_dir / "contoso.pbix"


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
st.title("Contoso Retail Analysis Dashboard")
st.markdown(
    """
    <div class="panel">
      <strong>Project Objective</strong><br>
      Built a retail performance dashboard to analyze revenue quality, product contribution,
      and customer distribution across regions for better commercial planning.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Dashboard Views")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Dashboard", "Sales Overview", "Product Analysis", "Geo and Demographics"]
)

with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        render_image(assets_dir / "dashbord_1.png", "Executive KPI dashboard")
    with c2:
        render_image(assets_dir / "dashbord_2.png", "Supporting management view")

with tab2:
    c1, c2 = st.columns([2, 1])
    with c1:
        render_image(assets_dir / "product.png", "Revenue and category performance")
    with c2:
        render_image(assets_dir / "product2.png", "Sales trend and mix detail")

with tab3:
    c1, c2 = st.columns([2, 1])
    with c1:
        render_image(assets_dir / "sub (1).png", "Product segment contribution")
    with c2:
        render_image(assets_dir / "sub (2).png", "Sub-category trend analysis")

with tab4:
    c1, c2 = st.columns([2, 1])
    with c1:
        render_image(assets_dir / "geo (1).png", "Regional performance map")
    with c2:
        render_image(assets_dir / "geo (2).png", "Customer demographic profile")

st.divider()

left, right = st.columns([2, 1], gap="large")
with left:
    st.info(
        "This dashboard was developed in Power BI Desktop. Download the PBIX file to inspect data "
        "modeling logic, DAX measures, and interactive drill paths."
    )
with right:
    if pbix_path.exists():
        with open(pbix_path, "rb") as file_obj:
            st.download_button(
                label="Download PBIX Source",
                data=file_obj,
                file_name="Contoso_Retail.pbix",
                mime="application/octet-stream",
                use_container_width=True,
            )
    else:
        st.error("PBIX source file not found in assets/contoso.")
