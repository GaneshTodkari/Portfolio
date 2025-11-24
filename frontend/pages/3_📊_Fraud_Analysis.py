import streamlit as st
import os

st.set_page_config(page_title="Fraud Analysis Dashboard", page_icon="📊", layout="wide")

st.title("📊 Insurance Fraud Claim Analysis")
st.markdown("""
**Project Overview:**
This Power BI dashboard complements the Machine Learning model by providing visual insights into insurance fraud patterns.
It analyzes demographic data, claim details, and vehicle information to identify high-risk segments.
""")

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(current_dir, "..", "assets", "fraud_car")
pbix_path = os.path.join(assets_dir, "Fraud Claim Analysis.pbix")

# --- DASHBOARD GALLERY ---
st.markdown("### 🖼️ Dashboard Insights")

# Create tabs for organized viewing
tab1, tab2, tab3 = st.tabs(["Dashbord","Demographics & Policy", "Evidence Analysis"])

with tab1:
    img_path = os.path.join(assets_dir,'1.png')
    if os.path.exists(img_path):
        st.image(img_path, caption="Gender Distribution: Analyzing Demographic Risk Factors", use_column_width=True)
    else:
        st.error("Missing: gender.png")
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        # Gender Distribution
        img_path = os.path.join(assets_dir, "gender.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Gender Distribution: Analyzing Demographic Risk Factors", use_column_width=True)
        else:
            st.error("Missing: gender.png")
            


with tab3:
        # Witness Reports
        img_path = os.path.join(assets_dir, "5.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Evidence Analysis: Discrepancies in Witness vs. Police Reports", use_column_width=True)
        else:
            st.error("Missing: 5.png")

st.divider()

# --- DOWNLOAD ---
col1, col2 = st.columns([2, 1])

with col1:
    st.info("""
    ℹ️ **Note:** This analysis was built using **Power BI Desktop**. 
    You can download the source file to interact with the filters and drill-downs.
    """)

with col2:
    if os.path.exists(pbix_path):
        with open(pbix_path, "rb") as f:
            st.download_button(
                label="📥 Download .pbix Source File",
                data=f,
                file_name="Fraud_Claim_Analysis.pbix",
                mime="application/octet-stream"
            )
    else:
        st.error("⚠️ Missing file: assets/Fraud Claim Analysis.pbix")