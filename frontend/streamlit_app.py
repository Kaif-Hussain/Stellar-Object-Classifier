import streamlit as st
import requests

API_URL = "https://stellar-object-classifier-ofph.onrender.com/predict"  

st.set_page_config(page_title="Stellar Object Classifier", page_icon="🌌", layout="centered")

st.title("🌌 Stellar Object Classifier")
st.write("Enter photometric and spectral features to predict whether the object is a **Galaxy**, **Quasar (QSO)**, or **Star**.")

st.divider()

with st.form("prediction_form"):
    st.subheader("Object Features")

    col1, col2 = st.columns(2)

    with col1:
        alpha = st.number_input("Alpha (Right Ascension)", value=150.0, format="%.6f")
        delta = st.number_input("Delta (Declination)", value=20.0, format="%.6f")
        u = st.number_input("u (ultraviolet magnitude)", value=22.0, format="%.6f")
        g = st.number_input("g (green magnitude)", value=20.0, format="%.6f")
        r = st.number_input("r (red magnitude)", value=19.0, format="%.6f")

    with col2:
        i = st.number_input("i (infrared magnitude)", value=18.5, format="%.6f")
        z = st.number_input("z (infrared magnitude)", value=18.0, format="%.6f")
        redshift = st.number_input("Redshift", value=0.5, format="%.6f")
        spectral_type = st.selectbox("Spectral Type", options=["M", "O/B", "K", "G", "F", "A"])
        galaxy_population = st.selectbox("Galaxy Population", options=["Red_Sequence", "Blue_Cloud", "Green_Valley"])

    submitted = st.form_submit_button("🔮 Predict Class")

if submitted:
    payload = {
        "alpha": alpha,
        "delta": delta,
        "u": u,
        "g": g,
        "r": r,
        "i": i,
        "z": z,
        "redshift": redshift,
        "spectral_type": spectral_type,
        "galaxy_population": galaxy_population
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.success(f"### Predicted Class: **{result['predicted_class']}**")

        if "probabilities" in result:
            import pandas as pd
            proba_df = pd.DataFrame({
                "Class": list(result["probabilities"].keys()),
                "Probability": list(result["probabilities"].values())
            }).sort_values("Probability", ascending=False)
            st.bar_chart(proba_df.set_index("Class"))

    except requests.exceptions.ConnectionError:
        st.error("Could not reach the backend API. Is it running?")
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.divider()
st.caption("Developed by [Kaif Hussain]. Powered by FastAPI and Streamlit.")