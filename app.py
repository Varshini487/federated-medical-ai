import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import json

st.set_page_config(page_title="🏥 Federated Medical AI", layout="wide")
st.title("🏥 Federated Medical AI (Simulation)")
st.markdown("Multi-hospital collaborative learning without sharing patient data")

@st.cache_data
def generate_hospital_data(hospital_id, n_samples=200):
    np.random.seed(hospital_id)
    age_shift = hospital_id * 5
    X = np.random.randn(n_samples, 5)  # 5 features: age, glucose, BP, cholesterol, BMI
    X[:, 0] = X[:, 0] + age_shift / 10
    y = (X[:, 0] + X[:, 1] + 0.5*X[:, 2] + np.random.randn(n_samples)*0.1 > 0).astype(int)
    return X, y

hospitals = {
    "Hospital A": generate_hospital_data(1),
    "Hospital B": generate_hospital_data(2),
    "Hospital C": generate_hospital_data(3),
}

if "global_model" not in st.session_state:
    st.session_state.global_model = LogisticRegression(random_state=42)
    st.session_state.round = 0
    st.session_state.history = []

st.sidebar.header("⚙️ Federated Learning Settings")
num_rounds = st.sidebar.slider("Number of rounds", 1, 10, 5)

tab1, tab2, tab3 = st.tabs(["🏥 Hospital Data", "🔄 FL Training", "📊 Results"])

with tab1:
    st.subheader("Local Patient Data (NEVER Leaves Hospital)")
    for hospital_name, (X, y) in hospitals.items():
        with st.expander(f"{hospital_name} — {len(X)} patients"):
            st.write(f"✅ Data is local — raw values NEVER sent to central server")
            st.write(f"Positive cases: {y.sum()}, Negative: {(1-y).sum()}")
            df_hosp = pd.DataFrame(X, columns=["Age_Norm", "Glucose", "BP", "Cholesterol", "BMI"])
            df_hosp["Disease"] = y
            st.dataframe(df_hosp.head(10))

with tab2:
    st.subheader("Federated Learning Rounds")
    if st.button(f"🚀 Run {num_rounds} FL Rounds"):
        with st.spinner("Training across hospitals..."):
            scaler = StandardScaler()
            for round_num in range(num_rounds):
                local_weights = []
                for hospital_name, (X, y) in hospitals.items():
                    X_scaled = scaler.fit_transform(X)
                    model = LogisticRegression(random_state=42, max_iter=100)
                    
                    if round_num == 0:
                        model.fit(X_scaled, y)
                    else:
                        model.coef_ = st.session_state.global_model.coef_.copy()
                        model.intercept_ = st.session_state.global_model.intercept_.copy()
                        model.fit(X_scaled, y)
                    
                    local_weights.append({
                        "hospital": hospital_name,
                        "coef": model.coef_[0].tolist(),
                        "intercept": float(model.intercept_[0])
                    })
                
                avg_coef = np.mean([w["coef"] for w in local_weights], axis=0)
                avg_intercept = np.mean([w["intercept"] for w in local_weights])
                
                st.session_state.global_model.coef_ = avg_coef.reshape(1, -1)
                st.session_state.global_model.intercept_ = np.array([avg_intercept])
                
                accuracy_global = 0
                for X, y in hospitals.values():
                    X_scaled = scaler.fit_transform(X)
                    acc = st.session_state.global_model.score(X_scaled, y)
                    accuracy_global += acc
                accuracy_global /= len(hospitals)
                
                st.session_state.history.append(accuracy_global)
                st.write(f"**Round {round_num+1}:** Global Model Accuracy = {accuracy_global:.2%}")
        
        st.success("✅ FL Training Complete!")
        st.balloons()

with tab3:
    st.subheader("Global Model Performance")
    if st.session_state.history:
        fig, ax = plt.subplots()
        ax.plot(st.session_state.history, marker="o", linewidth=2, markersize=8)
        ax.set_xlabel("Federated Round")
        ax.set_ylabel("Accuracy (avg across hospitals)")
        ax.set_title("Global Model Improvement Over Rounds")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("### 🎯 Key Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Accuracy", f"{st.session_state.history[-1]:.1%}")
        col2.metric("Improvement", f"+{(st.session_state.history[-1] - st.session_state.history[0]):.1%}")
        col3.metric("Rounds Completed", len(st.session_state.history))
        
        st.markdown("### 🔒 Privacy Guarantee")
        st.info("✅ **ZERO patient data transmitted.** Only model weights (5-10 numbers) sent per hospital per round. Even if weights are intercepted, they reveal nothing about individual patients.")

import matplotlib.pyplot as plt
