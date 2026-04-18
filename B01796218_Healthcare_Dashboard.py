#Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Attempt to import advanced models, fallback to sklearn if unavailable
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    ADV_MODELS = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier as LGBMClassifier
    ADV_MODELS = False

# ============================================================
# 1. PAGE CONFIG & BRANDING
# ============================================================
st.set_page_config(
    page_title="AI-Driven Cloud-IoT Healthcare Ecosystem",
    page_icon="🏥",
    layout="wide"
)

# Professional Dissertation Header
st.markdown("""
    <div style="background-color:#002b36; padding:25px; border-radius:15px; margin-bottom:25px; border-left: 10px solid #2aa198; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        <h1 style="color:white; text-align:center; font-size:32px; margin:0; font-family: 'Helvetica Neue', sans-serif;">
            Architecting AI-Driven Cloud-IoT Ecosystems for Global Healthcare
        </h1>
        <p style="color:#2aa198; text-align:center; font-size:18px; margin:10px 0 0 0; font-weight:300;">
            A Multi-Disease Predictive Analytics Approach to Clinical Decision Support and Strategic Management
        </p>
        <hr style="border: 0.5px solid #586e75; margin: 15px 0;">
        <div style="display: flex; justify-content: space-around; color:#859900; font-size:14px; font-weight:bold;">
            <span><b>Researcher:</b> Darpan Narayan Chaudhary</span>
            <span><b>Student ID:</b> B01796218</span>
            <span><b>© 2026 Academic Research | Copyright Protected</b></span>
        </div>
    </div>
""", unsafe_allow_html=True)

# Custom CSS for KPI Cards
st.markdown("""
<style>
    .kpi-box {
        background-color: #ffffff; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05); text-align: center;
        border-top: 4px solid #268bd2; transition: transform 0.3s;
    }
    .kpi-box:hover { transform: translateY(-5px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
    .kpi-val { font-size: 30px; font-weight: bold; color: #073642; }
    .kpi-label { font-size: 12px; color: #586e75; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 2. DATA & AI ENGINE
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Healthcare_Data_Multi_Disease.csv")
    # Basic data cleaning
    df['Mortality_Event'] = df['Mortality_Event'].fillna(0)
    return df

@st.cache_resource
def train_models(_df):
    features = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Fasting_Blood_Glucose', 
                'HbA1c', 'LDL_Cholesterol', 'CRP_mg_L', 'Resting_HR', 'Comorbidity_Index']
    
    # Use a sample for training speed in the dashboard
    train_sample = _df.sample(min(20000, len(_df)))
    X = train_sample[features].fillna(train_sample[features].median())
    y = train_sample['Mortality_Event']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. XGBoost
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb.fit(X, y)
    
    # 2. LightGBM
    lgbm = LGBMClassifier(n_estimators=100, random_state=42)
    lgbm.fit(X, y)
    
    # 3. SVM
    svm = SVC(probability=True, kernel='rbf', C=1.0)
    svm.fit(X_scaled, y)
    
    # Unsupervised: Anomalies and Clusters
    iso = IsolationForest(contamination=0.05, random_state=42)
    full_X_scaled = scaler.transform(_df[features].fillna(_df[features].median()))
    anomalies = iso.fit_predict(full_X_scaled)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(full_X_scaled)
    
    return {"XGBoost": xgb, "LightGBM": lgbm, "SVM": svm}, scaler, features, anomalies, clusters

data = load_data()
model_dict, data_scaler, feature_list, anomaly_labels, cluster_labels = train_models(data)

# Inject AI layers back into dataframe
data["Anomaly_Status"] = anomaly_labels
data["Health_Cluster"] = cluster_labels

# ============================================================
# 3. STRATEGIC SIDEBAR CONTROL
# ============================================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2785/2785482.png", width=80)
st.sidebar.title("Cloud-IoT Control")

menu = st.sidebar.radio("Module Navigation", [
    "📊 Strategic Analytics", 
    "🌐 Global Health Determinants", # NEW VISUAL INSIGHT PAGE
    "🔎 Clinical AI Diagnostic"
])

st.sidebar.divider()
st.sidebar.subheader("🌍 Population Filters")
eth_filter = st.sidebar.multiselect("Ethnicity demographic", options=sorted(data["Ethnicity"].unique()), default=data["Ethnicity"].unique())
diag_filter = st.sidebar.multiselect("Disease Focus", options=sorted(data["Primary_Diagnosis"].unique()), default=data["Primary_Diagnosis"].unique())

filtered_data = data[(data["Ethnicity"].isin(eth_filter)) & (data["Primary_Diagnosis"].isin(diag_filter))]



# ============================================================
# PAGE 1: STRATEGIC ANALYTICS
# ============================================================
if menu == "📊 Strategic Analytics":
    st.subheader("📋 Population Health Intelligence & Registry")

    # --- ROW 1: CLINICAL RISK TELEMETRY ---
    st.markdown("##### 🩺 Clinical Risk & IoT Telemetry")
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1: 
        st.markdown(f'<div class="kpi-box"><div class="kpi-val">{len(filtered_data):,}</div><div class="kpi-label">Active Nodes</div></div>', unsafe_allow_html=True)

    with k2: 
        st.markdown(f'<div class="kpi-box"><div class="kpi-val" style="color:#dc322f;">{(filtered_data["Mortality_Event"]==1).sum():,}</div><div class="kpi-label">Mortality Count</div></div>', unsafe_allow_html=True)

    with k3: 
        st.markdown(f'<div class="kpi-box"><div class="kpi-val" style="color:#2aa198;">{filtered_data["Medication_Adherence_Rate"].mean():.1f}%</div><div class="kpi-label">Avg Adherence</div></div>', unsafe_allow_html=True)

    with k4: 
        st.markdown(f'<div class="kpi-box"><div class="kpi-val" style="color:#b58900;">{filtered_data["Comorbidity_Index"].mean():.1f}</div><div class="kpi-label">Mean Comorb.</div></div>', unsafe_allow_html=True)

    with k5: 
        st.markdown(f'<div class="kpi-box"><div class="kpi-val">{(filtered_data["Anomaly_Status"]==-1).sum()}</div><div class="kpi-label">AI Anomalies</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ROW 2: DEMOGRAPHIC & POPULATION HEALTH ---
    st.markdown("##### 👥 Demographic & Structural Metrics")
    k6, k7, k8, k9 = st.columns(4)

    with k6:
        female_count = (filtered_data["Gender"]=="Female").sum()
        st.markdown(f'<div class="kpi-box"><div class="kpi-val" style="color: #d33682;">{female_count:,}</div><div class="kpi-label">Female Pop.</div></div>', unsafe_allow_html=True)

    with k7:
        male_count = (filtered_data["Gender"]=="Male").sum()
        st.markdown(f'<div class="kpi-box"><div class="kpi-val" style="color: #268bd2;">{male_count:,}</div><div class="kpi-label">Male Pop.</div></div>', unsafe_allow_html=True)

    with k8:
        geriatric_count = (filtered_data["Age"] > 60).sum()
        st.markdown(f'<div class="kpi-box"><div class="kpi-val" style="color: #859900;">{geriatric_count:,}</div><div class="kpi-label">Geriatric (60+)</div></div>', unsafe_allow_html=True)

    with k9:
        st.markdown(f'<div class="kpi-box"><div class="kpi-val" style="color: #6c71c4;">{filtered_data["BMI"].mean():.1f}</div><div class="kpi-label">Mean BMI</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.write("**Multi-Disease Prevalence Analysis**")
        diag_counts = filtered_data["Primary_Diagnosis"].value_counts().reset_index()
        fig_diag = px.bar(
            diag_counts,
            x="Primary_Diagnosis",
            y="count",
            color="count",
            color_continuous_scale="Viridis",
            template="plotly_white"
        )
        st.plotly_chart(fig_diag, use_container_width=True)

    with c2:
        st.write("**Vital Status Distribution**")
        fig_pie = px.pie(
            filtered_data,
            names="Vital_Status",
            hole=0.6,
            color_discrete_sequence=["#2aa198", "#dc322f"]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Clinical Correlation Chart ---
    st.subheader("🔬 Clinical Correlation: Metabolism vs. Inflammation")

    scatter_fig = px.scatter(
        filtered_data.sample(min(1500, len(filtered_data))),
        x="BMI",
        y="CRP_mg_L",
        color="Primary_Diagnosis",
        size="Fasting_Blood_Glucose",
        hover_data=["Age", "HbA1c"],
        labels={"CRP_mg_L": "C-Reactive Protein (Inflammation)"},
        template="plotly_white"
    )

    st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("---")

    st.write("**Master Synchronized Cloud Registry**")
    st.dataframe(filtered_data.head(100), use_container_width=True)

# ============================================================
# PAGE 2: GLOBAL HEALTH DETERMINANTS (NEW)
# ============================================================
elif menu == "🌐 Global Health Determinants":
    st.subheader("🌐 Strategic Determinants of Health & Equity")
    st.markdown("Analyzing the intersection of Socioeconomics, Geography, and Clinical Outcomes in the Cloud-IoT Ecosystem.")

    # --- ROW 1: Geographic & Economic Access ---
    r1_c1, r1_c2 = st.columns(2)
    with r1_c1:
        st.write("**🏥 Distance to Care vs. Mortality Outcomes**")
        fig_dist = px.box(filtered_data, x="Socioeconomic_Index", y="Distance_to_Clinic_km", color="Vital_Status",
                         notched=True, color_discrete_map={"Alive": "#2aa198", "Deceased": "#dc322f"}, template="plotly_white")
        st.plotly_chart(fig_dist, use_container_width=True)
    with r1_c2:
        st.write("**💰 Economic Hierarchy & Medication Adherence**")
        fig_sun = px.sunburst(filtered_data, path=['Ethnicity', 'Socioeconomic_Index'], values='Medication_Adherence_Rate',
                             color='Medication_Adherence_Rate', color_continuous_scale='RdYlGn', template="plotly_white")
        st.plotly_chart(fig_sun, use_container_width=True)

    st.markdown("---")

    # --- ROW 2: High-Dimensional Biomarker Variance ---
    st.write("**🧬 Population Biomarker Distribution by Demographic**")
    biomarker_choice = st.selectbox("Select Biomarker for Deep-Dive Analysis", ["HbA1c", "CRP_mg_L", "LDL_Cholesterol", "NT_proBNP_pg_mL"])
    fig_violin = px.violin(filtered_data, y=biomarker_choice, x="Ethnicity", color="Gender", box=True, points=None,
                          template="plotly_white", color_discrete_sequence=["#d33682", "#268bd2"])
    st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("---")

    # --- ROW 3: Environment & Lifestyle Correlations ---
    r3_c1, r3_c2 = st.columns(2)
    with r3_c1:
        st.write("**🌬️ Air Quality vs. Systemic Inflammation (CRP)**")
        fig_air = px.scatter(filtered_data.sample(min(2000, len(filtered_data))), x="Air_Quality_Index_Residence", y="CRP_mg_L", 
                            color="Smoking_Status", trendline="ols", template="plotly_white")
        st.plotly_chart(fig_air, use_container_width=True)
    with r3_c2:
        st.write("**📊 3D Metabolic Risk Space**")
        fig_3d = px.scatter_3d(filtered_data.sample(min(1000, len(filtered_data))), x='BMI', y='HbA1c', z='CRP_mg_L', 
                               color='Health_Cluster', size='Age', opacity=0.7, template="plotly_dark")
        st.plotly_chart(fig_3d, use_container_width=True)

# ============================================================
# PAGE 3: CLINICAL AI DIAGNOSTIC
# ============================================================
elif menu == "🔎 Clinical AI Diagnostic":
    st.title("🔎 Precision CDS & AI Telemetry")
    
    # Active Inference Selection
    selected_engine = st.selectbox("Active Inference Engine", options=["XGBoost", "LightGBM", "SVM"])
    
    search_id = st.text_input("📡 Cloud-IoT Node Inquiry: Search Patient ID", placeholder="e.g., P000001")
    p_rec = data[data["Patient_ID"] == search_id].iloc[0] if search_id and not data[data["Patient_ID"] == search_id].empty else None

    if p_rec is not None:
        st.success(f"Verified Record: {p_rec['Ethnicity']} {p_rec['Gender']} | {p_rec['Age']} yrs | {p_rec['Primary_Diagnosis']}")

    # Telemetry Input Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 0, 130, int(p_rec['Age']) if p_rec is not None else 50)
        sys_bp = st.number_input("Systolic BP", 80, 250, int(p_rec['Systolic_BP']) if p_rec is not None else 120)
        glucose = st.number_input("Glucose", 40, 500, int(p_rec['Fasting_Blood_Glucose']) if p_rec is not None else 100)
    with col2:
        bmi = st.number_input("BMI", 10.0, 80.0, float(p_rec['BMI']) if p_rec is not None else 25.0)
        dia_bp = st.number_input("Diastolic BP", 40, 160, int(p_rec['Diastolic_BP']) if p_rec is not None else 80)
        hba1c = st.number_input("HbA1c (%)", 3.0, 20.0, float(p_rec['HbA1c']) if p_rec is not None else 5.5)
    with col3:
        ldl = st.number_input("LDL Chol", 20, 400, int(p_rec['LDL_Cholesterol']) if p_rec is not None else 130)
        crp = st.number_input("CRP (Inflammation)", 0.0, 150.0, float(p_rec['CRP_mg_L']) if p_rec is not None else 1.0)
        hr = st.number_input("Resting HR", 30, 220, int(p_rec['Resting_HR']) if p_rec is not None else 72)

    if st.button("🧠 Compute High-Fidelity Inference", type="primary", use_container_width=True):
        comorb = int(p_rec['Comorbidity_Index']) if p_rec is not None else 2
        raw_v = np.array([[age, bmi, sys_bp, dia_bp, glucose, hba1c, ldl, crp, hr, comorb]])
        
        engine = model_dict[selected_engine]
        input_v = data_scaler.transform(raw_v) if selected_engine == "SVM" else raw_v
        prob = engine.predict_proba(input_v)[0][1]
        risk = int(prob * 100)

        # Risk Gauge
        r_col, m_col = st.columns([1, 2])
        with r_col:
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=risk, title={'text': "Mortality Risk %"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#073642"},
                       'steps': [{'range': [0, 35], 'color': "green"}, {'range': [35, 75], 'color': "orange"}, {'range': [75, 100], 'color': "red"}]}))
            st.plotly_chart(fig_g, use_container_width=True)
        
        with m_col:
            st.write("**AI Strategic Clinical Recommendation:**")
            if risk > 75: st.error("🚨 CRITICAL: High correlation with adverse events. Immediate hospitalization required.")
            elif risk > 40: st.warning("⚠️ ELEVATED: Risk identified. Adjust medication and increase IoT monitoring frequency.")
            else: st.success("✅ STABLE: Low risk detected. Maintain routine preventative care trajectory.")
            
            # Intervention Simulation
            st.info(f"💡 Potential Risk Reduction: **{max(0, risk-15)}%** if BMI and Blood Glucose are reduced by 10%.")

    with st.expander("📡 RAW CLOUD-IOT DATA TELEMETRY"):
        if p_rec is not None:
            st.json(p_rec.to_dict())
        else:
            st.info("Awaiting Node ID for live sync...")































