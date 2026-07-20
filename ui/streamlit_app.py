"""
Minimal Streamlit UI for the Counterfeit Currency Identification Agent.
Run with:
    streamlit run ui/streamlit_app.py

Assumes the FastAPI backend is already running (default: http://localhost:8000).
"""

import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Counterfeit Currency Agent", layout="centered")

st.title("💵 Counterfeit Currency Identification Agent")
st.caption(
    "Prototype for academic demonstration only — not for legal-grade authentication."
)

uploaded_file = st.file_uploader(
    "Upload a currency note image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    if st.button("Analyze Note"):
        with st.spinner("Running analysis pipeline..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = requests.post(f"{API_BASE_URL}/analyze", files=files, timeout=60)
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to backend. Is the FastAPI server running "
                    "(uvicorn app.main:app --reload)?"
                )
                st.stop()

        if response.status_code != 200:
            st.error(f"Analysis failed: {response.text}")
        else:
            result = response.json()

            verdict = result["verdict"]
            score = result["overall_score"]

            verdict_color = {
                "likely genuine": "green",
                "suspicious": "red",
                "unclear": "orange",
            }.get(verdict, "gray")

            st.markdown(
                f"### Verdict: :{verdict_color}[{verdict.upper()}]  "
                f"(score: {score:.1f}/100)"
            )

            if result.get("feature_scores"):
                st.subheader("Per-Feature Scores")
                st.json(result["feature_scores"])

            if result.get("explanations"):
                st.subheader("Decision Explanation")
                for line in result["explanations"]:
                    st.write(f"- {line}")

            st.subheader("Image Quality")
            st.json(result["image_quality"])

            st.subheader("Detection")
            st.json(result["detection"])

            st.subheader("Denomination")
            st.json(result["denomination"])

            if result.get("checks"):
                st.subheader("Check Breakdown")
                st.json(result["checks"])

            if result.get("annotated_image_path"):
                st.subheader("Annotated Output")
                st.text(f"Saved at: {result['annotated_image_path']}")

            if result.get("notes"):
                st.subheader("Notes / Disclaimers")
                for note in result["notes"]:
                    st.info(note)
