import streamlit as st
import pandas as pd
from datetime import datetime
from adaptive_scanner import run_adaptive_vulnerability_scan
from llama_guard import test_llama_guard_connection
from dotenv import load_dotenv
import os
import plotly.express as px

# Load environment variables
load_dotenv()
default_openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit Config
st.set_page_config(page_title="🗁️ LLM Vulnerability Scanner", layout="wide")
st.title("🗁️ Adaptive LLM Vulnerability Scanner")

# --- Provider Configuration ---
provider_defaults = {
    "Gemini": {
        "model": "gemini-pro",
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    },
    "OpenAI": {
        "model": "gpt-3.5-turbo",
        "endpoint": "https://api.openai.com/v1/chat/completions",
    },
    "Claude": {
        "model": "claude-3-opus-20240229",
        "endpoint": "https://api.anthropic.com/v1/messages",
    },
    "Mistral": {
        "model": "mistral-medium",
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
    },
    "Ollama": {"model": "tinyllama", "endpoint": "http://localhost:11434/api/generate"},
    "Together": {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "endpoint": "https://api.together.xyz/v1/chat/completions",
    },
}

# --- Sidebar Config ---
with st.sidebar:
    st.header("🎯 Target LLM Configuration")
    provider = st.selectbox("LLM Provider", list(provider_defaults.keys()))
    model = st.text_input("Model", value=provider_defaults[provider]["model"])
    endpoint = st.text_input(
        "API Endpoint", value=provider_defaults[provider]["endpoint"]
    )
    target_api_key = st.text_input("API Key (if required)", type="password")

    # Custom API Configuration
    st.markdown("---")
    st.subheader("🔧 Custom API Configuration")
    custom_api_url = st.text_input("Custom API Endpoint (Optional)")
    custom_api_key = st.text_input("Custom API Key (Optional)", type="password")

    st.markdown("---")
    st.subheader("⚙️ Attack Settings")
    multilingual = st.checkbox("🌍 Enable Multilingual Attack Testing", value=False)
    selected_languages = []
    if multilingual:
        selected_languages = st.multiselect(
            "Select Languages", ["fr", "de", "te", "es"], default=["fr", "de"]
        )

    attack_types = st.multiselect(
        "Attack Categories",
        [
            "Prompt Injection",
            "Jailbreak",
            "Privacy Leak",
            "Denial of Service",
            "Command Execution",
            "System Takeover",
        ],
        default=["Prompt Injection", "Privacy Leak"],
    )
    iterations = st.slider("Adaptive Iterations", 1, 10, 5)
    risk_threshold = st.slider("Risk Threshold", 1, 10, 6)

    # JSON File Upload for Templates/Prompts
    st.markdown("---")
    st.subheader("📂 Upload JSON Templates")
    uploaded_file = st.file_uploader(
        "Upload a JSON file with templates/prompts", type="json"
    )
    custom_templates = None
    if uploaded_file is not None:
        try:
            import json

            custom_templates = json.load(uploaded_file)
            st.success("✅ JSON file loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load JSON file: {e}")

    st.markdown("---")
    st.subheader("🗁️ Llama Guard")
    use_llama_guard = st.checkbox("Enable Llama Guard Safety Analysis", value=True)
    llama_guard_endpoint = st.text_input(
        "Llama Guard Endpoint",
        value="http://localhost:11434/api/generate",
        disabled=not use_llama_guard,
    )

    if use_llama_guard:
        st.info(
            "Llama Guard uses Ollama to analyze prompts and responses for safety concerns."
        )
        if st.button("Test Llama Guard Connection"):
            with st.spinner("Testing Llama Guard connection..."):
                llama_guard_available = test_llama_guard_connection(
                    llama_guard_endpoint
                )
                if llama_guard_available:
                    st.success("✅ Llama Guard is available and ready to use!")
                else:
                    st.error(
                        "❌ Llama Guard connection failed. Make sure Ollama is running and the llama-guard model is installed."
                    )
                    st.info("To install llama-guard, run: `ollama pull llama-guard`")

# --- Run Scanner ---
if st.button("🚀 Start Scan"):
    if provider.lower() != "ollama" and not target_api_key:
        st.error("Please enter an API key.")
    elif not default_openai_api_key:
        st.error("OpenAI API key not found in .env")
    else:
        if use_llama_guard:
            with st.spinner("Testing Llama Guard connection..."):
                llama_guard_available = test_llama_guard_connection(
                    llama_guard_endpoint
                )
                if llama_guard_available:
                    st.success(
                        "✅ Llama Guard is available and will be used for safety analysis"
                    )
                else:
                    st.warning("⚠️ Llama Guard connection failed, continuing without it")
                    use_llama_guard = False

        with st.spinner("Scanning for vulnerabilities..."):
            logs = run_adaptive_vulnerability_scan(
                generator_api_key=default_openai_api_key,
                target_api_key=target_api_key,
                target_provider=provider,
                target_model=model,
                target_endpoint=endpoint,
                selected_categories=attack_types,
                num_iterations=iterations,
                risk_threshold=risk_threshold,
                use_llama_guard=use_llama_guard,
                llama_guard_endpoint=llama_guard_endpoint if use_llama_guard else None,
                multilingual=multilingual,
                selected_languages=selected_languages,
                custom_api_url=custom_api_url,
                custom_api_key=custom_api_key,
                custom_templates=custom_templates,
            )

            df = pd.DataFrame(logs)
            st.success("✅ Scan Completed")

            tabs = st.tabs(["📊 Overview", "📜 Logs", "🚨 High Risk", "⬇️ Download"])

            with tabs[0]:
                st.subheader("Risk Score Distribution")
                if "risk_score" in df.columns and not df.empty:
                    fig = px.histogram(
                        df[df["risk_score"].notna()],
                        x="risk_score",
                        nbins=11,
                        title="Distribution of Risk Scores",
                        color_discrete_sequence=["#e74c3c"],
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric("Total Prompts Tested", len(df))
                    st.metric(
                        "High-Risk Responses (≥ Threshold)",
                        len(df[df["risk_score"] >= risk_threshold]),
                    )
                else:
                    st.warning("⚠️ No risk score data available to visualize.")

            with tabs[1]:
                st.subheader("All Scan Logs")

                rename_map = {
                    "response_translated_lang": "translated_response",
                    "response_back_to_english": "back_translated_response",
                    "template_used": "template",
                    "error": "failure_reason",
                }
                display_cols = [
                    "english_prompt",
                    "translated_prompt",
                    "translated_response",
                    "back_translated_response",
                    "adaptation_step",
                    "risk_score",
                    "failure_reason",
                    "template",
                ]
                df_logs = df.rename(columns=rename_map)
                for col in display_cols:
                    if col not in df_logs.columns:
                        df_logs[col] = ""

                st.markdown("### 📋 Tabular View (CSV-style Table)")
                st.dataframe(df_logs[display_cols], use_container_width=True)

                st.markdown("### 🧾 Detailed View (Expandable)")
                for idx, row in df.iterrows():
                    with st.expander(
                        f"🧪 {row.get('attack_type', 'Unknown Type')} | Risk: {row.get('risk_score', 'N/A')}"
                    ):
                        st.markdown(f"**Prompt:**\n{row.get('english_prompt', '')}")
                        if row.get("translated_prompt"):
                            st.markdown(
                                f"**Translated Prompt:**\n{row.get('translated_prompt')}"
                            )
                        if row.get("translated_response"):
                            st.markdown(
                                f"**Translated Response:**\n{row.get('translated_response')}"
                            )
                        if row.get("back_translated_response"):
                            st.markdown(
                                f"**Back Translated Response:**\n{row.get('back_translated_response')}"
                            )
                        if row.get("risk_score") is not None:
                            st.markdown(f"**Risk Score:** {row['risk_score']}")
                        if row.get("evaluator_explanation"):
                            st.markdown(
                                f"**Evaluation Explanation:**\n{row.get('evaluator_explanation')}"
                            )
                        if row.get("failure_reason"):
                            st.error(f"Failure Reason: {row['failure_reason']}")
            with tabs[2]:
                st.subheader("⚠️ High Risk Responses")
                if "risk_score" in df.columns:
                    high_risk_df = df[df["risk_score"] >= risk_threshold]
                    if not high_risk_df.empty:
                        st.dataframe(high_risk_df, use_container_width=True)
                    else:
                        st.info("No high-risk responses detected.")
                else:
                    st.warning("⚠️ No 'risk_score' column found in the results.")

            with tabs[3]:
                st.subheader("Download Full Report")
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📅 Download CSV", csv, "llm_vulnerability_report.csv", "text/csv"
                )
