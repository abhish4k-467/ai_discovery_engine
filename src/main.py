import os
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

DEFAULT_API_BASE_URL = os.getenv("DISCOVERY_API_URL", "http://127.0.0.1:8000")
DEFAULT_REQUEST_TIMEOUT = (5, 30)


def api_request(
    method: str, base_url: str, endpoint: str, **kwargs: Any
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        timeout = kwargs.pop("timeout", DEFAULT_REQUEST_TIMEOUT)
        response = requests.request(
            method=method, url=url, timeout=timeout, **kwargs
        )
        response.raise_for_status()

        if not response.content:
            return {}, None
        try:
            return response.json(), None
        except ValueError:
            return {"message": response.text}, None

    except requests.exceptions.HTTPError as exc:
        detail = None
        if exc.response is not None:
            try:
                payload = exc.response.json()
                if isinstance(payload, dict):
                    detail = payload.get("detail")
                if not detail:
                    detail = str(payload)
            except ValueError:
                detail = exc.response.text
        return None, detail or str(exc)

    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def init_session_state() -> None:
    if "api_base_url" not in st.session_state:
        st.session_state.api_base_url = DEFAULT_API_BASE_URL
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar() -> Tuple[str, str]:
    base_url = st.session_state.api_base_url
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["💬 Consultant Chat", "📑 Strategy Deliverables"], label_visibility="collapsed")
        
        st.divider()

        st.header("📄 Ingest Content")
        pdf_tab, url_tab = st.tabs(["PDF Upload", "URL"])

        with pdf_tab:
            with st.form("ingest_pdf_form", clear_on_submit=True):
                uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"]) 
                submit_pdf = st.form_submit_button("Ingest PDF", type="primary", use_container_width=True)

            if submit_pdf:
                if uploaded_file is None:
                    st.warning("Please choose a PDF file first.")
                else:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
                    with st.spinner("Ingesting PDF..."):
                        data, error = api_request("POST", base_url, "/ingest/file", files=files)
                    if error:
                        st.error(f"PDF ingestion failed: {error}")
                    else:
                        st.success(data.get("message", "PDF ingested successfully."))

        with url_tab:
            with st.form("ingest_url_form", clear_on_submit=True):
                url = st.text_input("Web page URL", placeholder="https://example.com/article")
                submit_url = st.form_submit_button("Ingest URL", type="primary", use_container_width=True)

            if submit_url:
                cleaned_url = url.strip()
                if not cleaned_url:
                    st.warning("Please enter a URL.")
                else:
                    with st.spinner("Ingesting URL..."):
                        data, error = api_request(
                            "POST",
                            base_url,
                            "/ingest/url",
                            json={"url": cleaned_url},
                        )
                    if error:
                        st.error(f"URL ingestion failed: {error}")
                    else:
                        st.success(data.get("message", "URL ingested successfully."))
        
        st.divider()
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return st.session_state.api_base_url, page


def main() -> None:
    st.set_page_config(page_title="The Launch Engine Assistant", layout="wide", page_icon="🚀")
    init_session_state()

    base_url, current_page = render_sidebar()

    st.title("🚀 The Launch Engine - Consultant AI")
    st.caption("Internal AI assistant connected to your firm's knowledge base. Use the chat for quick answers, or use the 'Generate Research' tab for deliverables.")

    if current_page == "💬 Consultant Chat":
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask about tech stacks, startup strategies, AWS infrastructure..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.spinner("Analyzing strategy..."):
                    data, error = api_request(
                        "POST",
                        base_url,
                        "/query",
                        json={"question": prompt},
                    )

                    if error:
                        response = f"**Error executing query:** {error}"
                    else:
                        response = data.get("answer", "No answer returned by API.")
                    
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    elif current_page == "📑 Strategy Deliverables":
        st.header("Generate Deep-Dive Research Report")
        st.write("Trigger comprehensive, formatted reports ready for client presentation. The AI will pull from ingested PDFs, URLs, and real-time web searches.")

        with st.form("research_form"):
            topic = st.text_input("Research Topic / Target Industry:", placeholder="e.g. AI-driven Supply Chain Startups in Series A")
            generate_submit = st.form_submit_button("Generate Report", type="primary")

        if generate_submit:
            if not topic.strip():
                st.warning("Please provide a topic.")
            else:
                with st.spinner(f"Compiling comprehensive report for '{topic}'. This may take a few minutes..."):
                    data, error = api_request(
                        "POST",
                        base_url,
                        "/research",
                        json={"topic": topic.strip()},
                        timeout=300 # Give it extra time
                    )
                
                if error:
                    st.error(f"Research generation failed: {error}")
                else:
                    report_text = data.get("report", "No report returned.")
                    st.success("Report generated successfully!")
                    
                    st.download_button(
                        label="Download Report as Markdown",
                        data=report_text,
                        file_name=f"research_report_{topic.replace(' ', '_')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                    with st.expander("Preview Report", expanded=True):
                        st.markdown(report_text)


if __name__ == "__main__":
    main()
