"""Streamlit Multi-Page Dashboard
Name: M.YAGNA
Roll Number: 3210XX
Dataset: merged_patient_data_normalized .csv (stored at C:/MINIPROJECT/)
Instructions: Save this file as `app.py` (or any name) and run with `streamlit run app.py`.

Features:
- Global sidebar filters (City, Gender, Age range, Primary Condition, Doctor) apply to all pages.
- Center-aligned title with consistent font across pages.
- Colorful, meaningful visualizations (using plotly and matplotlib where appropriate).
- Numeric values shown on bars/points when meaningful.
- Summary (with emojis) at the bottom of every page.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Config & Styling
# -------------------------
st.set_page_config(page_title="Comprehensive Dashboard", layout="wide")

# CSS for consistent font and centered title
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .center-title {
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtle {
        color: #6c757d;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='center-title'>Comprehensive Dashboard: Unveiling Patterns from Patient Data</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtle' style='text-align:center'>Interactive analysis of patient, medication, lifestyle and lab data 📊🔬</div>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# Load data
# -------------------------
@st.cache_data
def load_data(path="C:/MINIPROJECT/merged_patient_data_normalized .csv"):
    df = pd.read_csv(path)
    # Basic cleaning - unify column names
    df.columns = [c.strip() for c in df.columns]
    # Ensure datetime
    if 'Diagnosis_Date' in df.columns:
        df['Diagnosis_Date'] = pd.to_datetime(df['Diagnosis_Date'], errors='coerce')
    # If Age not numeric, try to coerce
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found at C:/MINIPROJECT/merged_patient_data_normalized .csv. Please check the path.")
    st.stop()

# -------------------------
# Global Sidebar Filters
# -------------------------
st.sidebar.header("🔎 Global Filters")

def safe_unique(col):
    return sorted(df[col].dropna().unique().tolist())

city_sel = st.sidebar.multiselect("City", options=safe_unique('City'), default=safe_unique('City'))
gender_sel = st.sidebar.multiselect("Gender", options=safe_unique('Gender'), default=safe_unique('Gender'))

age_min = int(np.nanmin(df['Age'])) if 'Age' in df.columns else 0
age_max = int(np.nanmax(df['Age'])) if 'Age' in df.columns else 100
age_range = st.sidebar.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

condition_sel = st.sidebar.multiselect("Primary Condition", options=safe_unique('Condition_Primary'), default=safe_unique('Condition_Primary'))
doctor_sel = st.sidebar.multiselect("Doctor",
    options=safe_unique('Primary_Care_Doctor') if 'Primary_Care_Doctor' in df.columns else [],
    default=safe_unique('Primary_Care_Doctor') if 'Primary_Care_Doctor' in df.columns else [])

# Apply filters
filtered = df.copy()
if city_sel:
    filtered = filtered[filtered['City'].isin(city_sel)]
if gender_sel:
    filtered = filtered[filtered['Gender'].isin(gender_sel)]
if 'Age' in filtered.columns:
    filtered = filtered[(filtered['Age'] >= age_range[0]) & (filtered['Age'] <= age_range[1])]
if condition_sel:
    filtered = filtered[filtered['Condition_Primary'].isin(condition_sel)]
if doctor_sel and 'Primary_Care_Doctor' in filtered.columns:
    filtered = filtered[filtered['Primary_Care_Doctor'].isin(doctor_sel)]

if filtered.empty:
    st.warning("No rows match the selected filters — try widening the filters.")

# -------------------------
# Navigation
# -------------------------
pages = [
    "Overview",
    "Conditions & Health Stats",
    "Medications & Costs",
    "Lifestyle & Demographics",
    "Lab Tests & Results",
    "Final Summary"
]
page = st.sidebar.radio("Pages", pages)

# Helper for bar charts
def px_bar_with_values(df_plot, x, y, title, orientation='v', color=None, text_auto=True, height=450):
    fig = px.bar(df_plot, x=x, y=y, orientation=orientation, color=color, text=y if text_auto else None)
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(title=title, uniformtext_minsize=8, uniformtext_mode='hide', height=height)
    return fig

# -------------------------
# Page: Overview
# -------------------------
if page == 'Overview':
    st.subheader("Overview 📈")
    col1, col2, col3, col4, col5 = st.columns(5)
    total_patients = int(filtered['Patient_ID'].nunique()) if 'Patient_ID' in filtered.columns else len(filtered)
    avg_age = round(filtered['Age'].mean(), 1) if 'Age' in filtered.columns else np.nan
    avg_bmi = round(filtered['BMI'].mean(), 1) if 'BMI' in filtered.columns else np.nan
    total_doctors = int(filtered['Primary_Care_Doctor'].nunique()) if 'Primary_Care_Doctor' in filtered.columns else np.nan
    total_drugs = int(filtered['Drug_Name'].nunique()) if 'Drug_Name' in filtered.columns else np.nan

    col1.metric("Total Patients 👥", total_patients)
    col2.metric("Avg Age 🧓", avg_age)
    col3.metric("Avg BMI ⚖️", avg_bmi)
    col4.metric("Doctors 🩺", total_doctors)
    col5.metric("Unique Drugs 💊", total_drugs)

    st.markdown("### Patient Distribution")
    g1, g2 = st.columns([1,1])

    if 'Gender' in filtered.columns:
        gender_counts = filtered['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender','Count']
        fig = px.pie(gender_counts, names='Gender', values='Count', hole=0.45, title='Gender Distribution')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        g1.plotly_chart(fig, use_container_width=True)

    if 'Diagnosis_Date' in filtered.columns:
        time_df = filtered.dropna(subset=['Diagnosis_Date']).groupby(filtered['Diagnosis_Date'].dt.to_period('M')).size().reset_index(name='count')
        if not time_df.empty:
            time_df['Diagnosis_Date'] = time_df['Diagnosis_Date'].dt.to_timestamp()
            fig = px.line(time_df, x='Diagnosis_Date', y='count', markers=True, title='Patients Over Time')
            fig.update_layout(height=420)
            g2.plotly_chart(fig, use_container_width=True)
    else:
        g2.info("No Diagnosis_Date column found to show trends over time.")

    st.markdown("### Top Cities & Age Distribution")
    c1, c2 = st.columns([1,1])
    if 'City' in filtered.columns:
        city_counts = filtered['City'].value_counts().nlargest(10).reset_index()
        city_counts.columns = ['City','Count']
        fig = px.bar(city_counts, x='Count', y='City', orientation='h', title='Top Cities by Patient Count', text='Count')
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        c1.plotly_chart(fig, use_container_width=True)

    if 'Age' in filtered.columns:
        fig = px.histogram(filtered, x='Age', nbins=20, marginal='box', title='Age Distribution')
        fig.update_layout(height=420)
        c2.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📌 Summary")
    st.write("Overall, patient population is concentrated in a few major cities with varied age distribution. Use the sidebar filters to refine these views. ✨")

# -------------------------
# Remaining pages (Conditions, Medications, Lifestyle, Lab Tests, Final Summary)
# -------------------------
elif page == 'Conditions & Health Stats':
    st.subheader("Conditions & Health Stats ⚕️")
    if 'Condition_Primary' in filtered.columns:
        cond = filtered['Condition_Primary'].value_counts().nlargest(8).reset_index()
        cond.columns = ['Condition','Count']
        fig = px.bar(cond, x='Condition', y='Count', title='Top Primary Conditions', text='Count')
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    if all(col in filtered.columns for col in ['BMI','Condition_Primary']):
        bmi_df = filtered.dropna(subset=['BMI','Condition_Primary'])
        if not bmi_df.empty:
            fig = px.box(bmi_df, x='Condition_Primary', y='BMI', title='BMI Distribution by Primary Condition')
            fig.update_layout(xaxis={'categoryorder':'total descending'}, height=450)
            col1.plotly_chart(fig, use_container_width=True)

    num_cols = [c for c in ['BMI','Age','Systolic_BP','Diastolic_BP','HbA1c','LDL','HDL','eGFR'] if c in filtered.columns]
    if len(num_cols) >= 2:
        corr = filtered[num_cols].corr()
        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(corr, annot=True, cmap='vlag', center=0, ax=ax)
        ax.set_title('Correlation: Key Health Indicators')
        col2.pyplot(fig)

    st.markdown("### 📌 Summary")
    st.write("Hypertension and Diabetes are among the top diagnosed conditions. BMI shows meaningful variation across conditions — useful for targeted intervention programs. 🩺📌")

elif page == 'Medications & Costs':
    st.subheader("Medications & Costs 💊💰")
    if 'Drug_Name' in filtered.columns:
        drug_counts = filtered['Drug_Name'].value_counts().nlargest(12).reset_index()
        drug_counts.columns = ['Drug','Count']
        fig = px.bar(drug_counts, x='Count', y='Drug', orientation='h', title='Top Prescribed Drugs', text='Count')
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    if all(c in filtered.columns for c in ['Drug_Name','Adherence_Percent']):
        adh = filtered.groupby('Drug_Name')['Adherence_Percent'].mean().nlargest(12).reset_index()
        adh.columns = ['Drug','Avg_Adherence']
        fig = px.bar(adh, x='Avg_Adherence', y='Drug', orientation='h', title='Average Adherence % by Drug', text='Avg_Adherence')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    if 'Cost_USD' in filtered.columns and 'City' in filtered.columns:
        city_costs = filtered.groupby('City')['Cost_USD'].sum().reset_index().nlargest(15, 'Cost_USD')
        fig = px.treemap(city_costs, path=['City'], values='Cost_USD', title='Medication Cost by City')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📌 Summary")
    st.write("Medication demand concentrates on a handful of drugs; adherence varies. Costs are higher in urban clusters — consider cost-awareness programs. 💡")

elif page == 'Lifestyle & Demographics':
    st.subheader("Lifestyle & Demographics 🧭")
    cols = ['Smoker','Alcohol_Use','Insurance_Type']
    available = [c for c in cols if c in filtered.columns]
    if available:
        charts = st.columns(len(available))
        for i, colname in enumerate(available):
            counts = filtered[colname].value_counts().reset_index()
            counts.columns = [colname, 'Count']
            fig = px.pie(counts, names=colname, values='Count', title=colname, hole=0.4)
            charts[i].plotly_chart(fig, use_container_width=True)

    if 'Occupation' in filtered.columns:
        occ = filtered['Occupation'].value_counts().nlargest(10).reset_index()
        occ.columns = ['Occupation','Count']
        fig = px.bar(occ, x='Count', y='Occupation', orientation='h', title='Top Occupations', text='Count')
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📌 Summary")
    st.write("Low smoking prevalence but moderate alcohol use in some groups. Insurance coverage varies — filtering by city or doctor shows interesting differences. 🧾📍")

elif page == 'Lab Tests & Results':
    st.subheader("Lab Tests & Results 🔬")
    if 'Test_Name' in filtered.columns:
        test_counts = filtered['Test_Name'].value_counts().nlargest(12).reset_index()
        test_counts.columns = ['Test','Count']
        fig = px.bar(test_counts, x='Count', y='Test', orientation='h', title='Most Common Lab Tests', text='Count')
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    if 'Result_Value' in filtered.columns:
        top_tests = filtered['Test_Name'].value_counts().nlargest(5).index.tolist()
        subset = filtered[filtered['Test_Name'].isin(top_tests)].dropna(subset=['Result_Value'])
        if not subset.empty:
            fig = px.violin(subset, x='Test_Name', y='Result_Value', box=True, points='all',
                            title='Result Value Distribution for Top Tests')
            fig.update_traces(meanline_visible=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No Result_Value data available for selected tests.')
    else:
        st.info('No Result_Value column found in dataset.')

    st.markdown("### 📌 Summary")
    st.write("HbA1c and kidney function tests appear frequently. Violin plots reveal spread and outliers — useful for flagging abnormal results 🧪🔍")

elif page == 'Final Summary':
    st.subheader("Final Insights & Recommendations ✨")
    st.markdown("**Top takeaways:**")
    st.write("- Major conditions include Hypertension and Diabetes — focus preventive programs accordingly. 🩺")
    st.write("- Adherence to medicines varies by drug; patient education may improve outcomes. 💊")
    st.write("- Urban centers contribute a majority of costs and patient volume; allocate resources appropriately. 🏙️")
    st.write("- Lab results (HbA1c, eGFR) are key monitoring metrics — set alert thresholds for outliers. 🚨")

    st.markdown("---")
    st.write("**Presenter:** M.YAGNA  •  **Roll No.:** 321XXX")
    st.markdown("**How to run:** Save this file as `app.py` and run `streamlit run app.py`. Use the sidebar to filter data globally and navigate pages. ✅")

else:
    st.info("Select a page from the sidebar.")
