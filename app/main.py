"""
EXAI-ResumeIntel: Streamlit application entry point
===================================================

Module: app.main
Author: Mithin Sagar S
GitHub: https://github.com/mithinsagar
Institution: Vellore Institute of Technology (VIT)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend.model_loader import load_sbert_model, load_job_data
from backend.preprocessing import extract_text_from_document, preprocess_resume_text
from backend.similarity import suggest_roles
from backend.evaluation import evaluate_resume_for_role
from backend.xai_module import (
    compute_skill_contribution,
    simulate_skill_addition,
    generate_detailed_explanation,
    identify_missing_skills,
)
from backend.history_store import (
    init_history_db,
    save_history_entry,
    list_history_entries,
    get_history_entry,
)


APP_TITLE = "SkillLens AI — Explainable Career Intelligence"
APP_SUBTITLE = (
    "Understand how your resume aligns with modern job roles through interpretable AI."
)

BASE_DIR = Path(__file__).resolve().parent


def inject_custom_css(theme: str) -> None:
    css_path = BASE_DIR / "assets" / "styles.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        light_overrides = """
        body, .stApp {
            background:
                radial-gradient(circle at top left, rgba(56,189,248,0.10) 0, transparent 42%),
                radial-gradient(circle at bottom right, rgba(129,140,248,0.10) 0, transparent 46%),
                linear-gradient(145deg, #f7fafc 0, #eef2ff 38%, #f8fafc 100%) !important;
            color: #0f172a !important;
        }
        .app-header {
            background: linear-gradient(to bottom, rgba(255,255,255,0.88), rgba(255,255,255,0.72), transparent) !important;
            border-bottom: 1px solid rgba(148,163,184,0.35) !important;
        }
        .hero-card, .result-card, .loading-screen {
            background: linear-gradient(135deg, rgba(255,255,255,0.72), rgba(248,250,252,0.58)) !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
            box-shadow: 0 18px 45px rgba(100,116,139,0.14), 0 0 0 1px rgba(255,255,255,0.6) inset !important;
        }
        .hero-title, .nav-title, .result-title, .result-label, .loading-text { color: #0f172a !important; }
        .hero-subtitle, .nav-subtitle, .match-caption, .landing-hint, .loading-subtext { color: #475569 !important; }
        .upload-dropzone {
            background: linear-gradient(135deg, rgba(255,255,255,0.8), rgba(248,250,252,0.72)) !important;
            border: 1px dashed rgba(100,116,139,0.55) !important;
        }
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(135deg, rgba(255,255,255,0.78), rgba(248,250,252,0.72)) !important;
            border: 1px solid rgba(148,163,184,0.45) !important;
            box-shadow: 0 10px 26px rgba(100,116,139,0.10), 0 0 0 1px rgba(255,255,255,0.65) inset !important;
        }
        .stFileUploader [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p,
        .stFileUploader [data-testid="stFileUploaderDropzone"] small {
            color: #334155 !important;
        }
        .stFileUploader [data-testid="stFileUploaderDropzone"] button {
            background: linear-gradient(135deg, rgba(248,250,252,0.92), rgba(241,245,249,0.88)) !important;
            border: 1px solid rgba(148,163,184,0.45) !important;
            color: #0f172a !important;
        }
        .nav-text-link { color: #2563eb !important; text-shadow: 0 0 6px rgba(37,99,235,0.22) !important; }
        .nav-text-link.active { color: #0f172a !important; text-shadow: 0 0 9px rgba(45,212,191,0.24) !important; }
        .stMetric {
            background: rgba(255,255,255,0.72) !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
        }
        .app-footer { color: #64748b !important; border-top: 1px solid rgba(148,163,184,0.35) !important; }
        .stApp, .stMarkdown, .stCaption, .stText, .stAlert, .stMetric, p, span, label {
            color: #0f172a !important;
        }
        .stTextInput input, .stTextArea textarea {
            background: rgba(255,255,255,0.88) !important;
            color: #0f172a !important;
            border: 1px solid rgba(148,163,184,0.45) !important;
        }
        .stSelectbox [data-baseweb="select"] > div {
            background: rgba(255,255,255,0.88) !important;
            color: #0f172a !important;
            border: 1px solid rgba(148,163,184,0.45) !important;
        }
        .st-key-theme_switch_btn button {
            color: #0f172a !important;
            border: 1px solid rgba(100,116,139,0.45) !important;
            background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(241,245,249,0.9)) !important;
            box-shadow: 0 8px 18px rgba(100,116,139,0.18) !important;
        }
        .st-key-theme_switch_btn button:hover {
            color: #020617 !important;
            filter: brightness(1.03);
        }
        .st-key-quick_role_btn_0 button,
        .st-key-quick_role_btn_1 button,
        .st-key-quick_role_btn_2 button,
        .st-key-quick_role_btn_3 button,
        .st-key-quick_role_btn_4 button {
            background: linear-gradient(135deg, #e2e8f0, #cbd5e1) !important;
            color: #1e293b !important;
            border: 1px solid rgba(100,116,139,0.35) !important;
            box-shadow: 0 6px 16px rgba(100,116,139,0.14) !important;
        }
        .st-key-quick_role_btn_0 button:hover,
        .st-key-quick_role_btn_1 button:hover,
        .st-key-quick_role_btn_2 button:hover,
        .st-key-quick_role_btn_3 button:hover,
        .st-key-quick_role_btn_4 button:hover {
            background: linear-gradient(135deg, #dbeafe, #c7d2fe) !important;
            color: #0f172a !important;
            border-color: rgba(99,102,241,0.45) !important;
        }
        .st-key-analyze_btn button {
            background: linear-gradient(135deg, #6366f1, #3b82f6) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(99,102,241,0.55) !important;
            box-shadow: 0 10px 22px rgba(79,70,229,0.24) !important;
        }
        .st-key-analyze_btn button:hover {
            background: linear-gradient(135deg, #4f46e5, #2563eb) !important;
            color: #ffffff !important;
        }
        """
        if theme == "light":
            st.markdown(f"<style>{css}\n{light_overrides}</style>", unsafe_allow_html=True)
        else:
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _safe_rerun() -> None:
    """
    Compatibility wrapper for Streamlit rerun across versions.
    """
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


@st.cache_resource(show_spinner=False)
def get_model():
    return load_sbert_model()


@st.cache_resource(show_spinner=False)
def get_job_data():
    return load_job_data()


@st.cache_data(show_spinner=False)
def get_unique_roles(job_df: pd.DataFrame) -> List[str]:
    roles = job_df["Role"].dropna().astype(str).unique().tolist()
    roles = sorted(set(r.strip() for r in roles if r.strip()))
    return roles


@st.cache_data(show_spinner=False)
def get_quick_roles(job_df: pd.DataFrame, n: int = 5) -> List[str]:
    role_counts = (
        job_df["Role"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .value_counts()
    )
    picks = role_counts.head(n).index.tolist()
    if len(picks) < n:
        # fallback from sorted uniques
        uniques = sorted(set(job_df["Role"].dropna().astype(str).str.strip().tolist()))
        for role in uniques:
            if role and role not in picks:
                picks.append(role)
            if len(picks) >= n:
                break
    return picks


def render_top_nav(current_page: str, theme: str) -> None:
    st.markdown('<div class="app-header">', unsafe_allow_html=True)
    nav_left, nav_center, nav_right = st.columns([2, 3, 1.5])

    with nav_left:
        st.markdown(
            """
            <div class="nav-bar-left">
                <div class="nav-logo">SL</div>
                <div class="nav-title-block">
                    <div class="nav-title">SkillLens AI</div>
                    <div class="nav-subtitle">Explainable Career Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_center:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(
                "Analyze",
                key="nav_tab_analyze",
                use_container_width=True,
                type="primary" if current_page == "Analyze" else "secondary",
            ):
                st.query_params["page"] = "Analyze"
                _safe_rerun()
        with c2:
            if st.button(
                "Insights",
                key="nav_tab_insights",
                use_container_width=True,
                type="primary" if current_page == "Insights" else "secondary",
            ):
                st.query_params["page"] = "Insights"
                _safe_rerun()
        with c3:
            if st.button(
                "About",
                key="nav_tab_about",
                use_container_width=True,
                type="primary" if current_page == "About" else "secondary",
            ):
                st.query_params["page"] = "About"
                _safe_rerun()

    with nav_right:
        right_a, right_b = st.columns([1, 1])
        with right_a:
            theme_label = "☾" if theme == "light" else "☀"
            if st.button(theme_label, key="theme_switch_btn", use_container_width=False):
                st.session_state["theme"] = "light" if theme == "dark" else "dark"
                _safe_rerun()
        with right_b:
            st.markdown(
                """
                <div class="nav-right-inline">
                    <div class="nav-avatar">
                        <span class="nav-avatar-initial">AI</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def render_interaction_card(roles: List[str], quick_roles: List[str]) -> Tuple[Any, str, bool]:
    st.markdown('<div class="hero-wrapper">', unsafe_allow_html=True)
    with st.container():
        st.markdown(
            """
            <div class="hero-card">
                <div class="hero-header">
                    <div class="hero-pill">AI-powered resume alignment</div>
                    <h2 class="hero-title">Drop your resume to see how it aligns with your next role.</h2>
                    <p class="hero-subtitle">
                        SkillLens AI analyzes your experience, skills, and projects using transformer-based semantic similarity and perturbation-based explainability.
                    </p>
                </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="role-suggestion-label">Upload your resume (PDF/DOCX)</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Drop your resume here to begin analysis",
            type=["pdf", "docx"],
            help="Supported formats: PDF, DOCX. Max size ~200MB (subject to environment limits).",
            label_visibility="collapsed",
        )
        st.caption("Tip: Choose a target role after upload for better role suggestions.")

        with st.container(key="role_selector_panel"):
            st.markdown(
                '<div class="role-suggestion-label">Quick role picks</div>',
                unsafe_allow_html=True,
            )
            if "role_input_text" not in st.session_state:
                st.session_state["role_input_text"] = ""

            quick_cols = st.columns(len(quick_roles))
            for idx, role in enumerate(quick_roles):
                if quick_cols[idx].button(
                    role,
                    key=f"quick_role_btn_{idx}",
                    use_container_width=True,
                ):
                    st.session_state["role_input_text"] = role

            role_input = st.text_input(
                "Select or search for a target role...",
                placeholder="Select or search for a target role...",
                key="role_input_text",
            )

            suggestions = []
            selected_suggestion = ""
            if role_input.strip():
                suggestions = suggest_roles(
                    query=role_input, roles=roles, max_suggestions=len(roles)
                )
                if suggestions:
                    st.markdown(
                        '<div class="role-suggestion-label">Suggestions</div>',
                        unsafe_allow_html=True,
                    )
                    selected_suggestion = st.selectbox(
                        "Role suggestions",
                        options=["(Use typed role)"] + suggestions,
                        label_visibility="collapsed",
                        key="role_suggestions",
                    )

        analyze_clicked = st.button(
            "Analyze Resume",
            type="primary",
            use_container_width=True,
            key="analyze_btn",
        )

        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    final_role = (selected_suggestion or role_input).strip()
    if selected_suggestion and selected_suggestion == "(Use typed role)":
        final_role = role_input.strip()

    return uploaded_file, final_role, analyze_clicked


def _plot_match_gauge(match_percentage: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(match_percentage),
            number={"suffix": "%", "font": {"size": 40, "color": "#E2E8F0"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4B5563"},
                "bar": {"color": "#38BDF8"},
                "bgcolor": "rgba(15,23,42,0.0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(248,113,113,0.3)"},
                    {"range": [50, 75], "color": "rgba(250,204,21,0.3)"},
                    {"range": [75, 100], "color": "rgba(74,222,128,0.3)"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Overall Match", "font": {"size": 16, "color": "#CBD5F5"}},
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=0),
        paper_bgcolor="rgba(15,23,42,0.0)",
        plot_bgcolor="rgba(15,23,42,0.0)",
    )
    return fig


def _plot_skill_contributions(skills: List[Dict[str, Any]]) -> go.Figure:
    if not skills:
        return go.Figure()

    df = pd.DataFrame(skills)
    df = df.sort_values("contribution", ascending=True)

    colors = ["#22C55E" if v >= 0 else "#EF4444" for v in df["contribution"]]

    fig = px.bar(
        df,
        x="contribution",
        y="skill",
        orientation="h",
        color=df["contribution"] >= 0,
        color_discrete_sequence=["#EF4444", "#22C55E"],
        labels={"contribution": "Impact (%)", "skill": "Skill"},
        hover_data={"contribution": ":.2f", "skill": True},
    )

    fig.update_traces(
        marker_color=colors,
        marker_line_color="#020617",
        marker_line_width=0.8,
        opacity=0.9,
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(15,23,42,0.0)",
        plot_bgcolor="rgba(15,23,42,0.0)",
        xaxis_title="Contribution to Match (%)",
        yaxis_title="",
    )
    return fig


def _plot_improvement_simulation(improvements: List[Dict[str, Any]]) -> go.Figure:
    if not improvements:
        return go.Figure()

    df = pd.DataFrame(improvements)

    fig = px.bar(
        df,
        x="potential_increase",
        y="skill",
        orientation="h",
        color="potential_increase",
        color_continuous_scale="Blues",
        labels={
            "potential_increase": "Potential Match Increase (%)",
            "skill": "Skill (If Added)",
        },
        hover_data={"potential_increase": ":.2f", "skill": True},
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(15,23,42,0.0)",
        plot_bgcolor="rgba(15,23,42,0.0)",
        xaxis_title="Potential Match Gain (%)",
        yaxis_title="",
    )
    return fig


def _compute_skill_coverage(
    top_contrib: List[Dict[str, Any]], missing_skills: List[str]
) -> float:
    present = len(top_contrib)
    missing = len(missing_skills)
    total = present + missing
    if total == 0:
        return 0.0
    return 100.0 * present / total


def _export_report_pdf(summary: Dict[str, Any]) -> bytes:
    """Generate a simple PDF report using reportlab."""
    try:
        from io import BytesIO

        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        x_margin = 40
        y = height - 50

        def write_line(text: str, font_size: int = 11, leading: int = 14):
            nonlocal y
            if y < 60:
                c.showPage()
                y = height - 50
            c.setFont("Helvetica", font_size)
            c.drawString(x_margin, y, text)
            y -= leading

        c.setTitle("Resume–Job Matching XAI Report")
        write_line("Resume–Job Matching XAI Report", 16, 20)
        write_line("")

        write_line(f"Target Role: {summary.get('role', '-')}", 12, 16)
        write_line(f"Match Percentage: {summary.get('match_percentage', 0):.2f}%", 12, 16)
        write_line(
            f"Semantic Similarity: {summary.get('semantic_similarity', 0):.3f}", 12, 16
        )
        write_line("")

        write_line("Why this score?", 13, 18)
        explanation = summary.get("explanation", "")
        for line in explanation.splitlines():
            write_line(line)
        write_line("")

        write_line("Top Positive Influences", 13, 18)
        for item in summary.get("top_contributing_skills", []):
            write_line(
                f"- {item['skill']}: {item['contribution']:.2f}% contribution"
            )
        write_line("")

        write_line("Critical Missing Skills", 13, 18)
        for skill in summary.get("missing_skills", []):
            write_line(f"- {skill}")
        write_line("")

        write_line("How to Improve (Simulation)", 13, 18)
        for item in summary.get("improvements", []):
            write_line(
                f"- {item['skill']}: +{item['potential_increase']:.2f}% potential"
            )

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return b""


def render_results(
    result: Dict[str, Any],
    resume_text: str,
    role: str,
    top_contrib: List[Dict[str, Any]],
    missing_skills: List[str],
    improvements: List[Dict[str, Any]],
) -> None:
    st.markdown('<div class="results-wrapper">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="result-card result-card-main">
            <div class="result-label">Resume ↔ Role Match</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    match_percentage = float(result.get("match_percentage", 0.0))
    semantic_similarity = float(result.get("semantic_similarity", 0.0))

    col_score, col_meta = st.columns([1.2, 1])

    with col_score:
        st.plotly_chart(_plot_match_gauge(match_percentage), use_container_width=True)

    with col_meta:
        st.metric(
            label="Semantic Similarity",
            value=f"{semantic_similarity:.3f}",
            help="Cosine similarity between resume and target role embedding.",
        )
        coverage = _compute_skill_coverage(top_contrib, missing_skills)
        st.metric(
            label="Skill Coverage",
            value=f"{coverage:.1f}%",
            help="Proportion of required skills covered by the resume.",
        )
        st.markdown(
            """
            <p class="match-caption">
                Your profile shows semantic alignment with the selected role based on experience, skills, and projects.
            </p>
            """,
            unsafe_allow_html=True,
        )

    # Skill alignment
    st.markdown(
        """
        <div class="result-card">
            <div class="result-title">Skill Alignment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    strong_col, missing_col = st.columns(2)
    with strong_col:
        st.markdown("#### Strong Skills")
        strong_skills = [s for s in top_contrib if s.get("contribution", 0) >= 0]
        if strong_skills:
            for item in strong_skills[:10]:
                st.markdown(f"✔ **{item['skill']}**")
        else:
            st.markdown("_No clearly dominant positive skills identified._")
    with missing_col:
        st.markdown("#### Missing Skills")
        if missing_skills:
            for s in missing_skills[:10]:
                st.markdown(f"⚠ **{s}**")
        else:
            st.markdown("_No critical missing skills identified._")

    # Explainability sections
    st.markdown(
        """
        <div class="result-card">
            <div class="result-title">Why the model predicted this score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    explanation = generate_detailed_explanation(
        resume_text=resume_text,
        target_role=role,
        match_percentage=match_percentage,
        semantic_similarity=semantic_similarity,
        top_contributing_skills=top_contrib,
        missing_skills=missing_skills,
        skill_improvements=improvements,
    )
    st.write(explanation)

    # Feature importance bars if provided by backend
    feature_importances = result.get("feature_importances") or []
    if feature_importances:
        fi_df = pd.DataFrame(feature_importances)
        fi_df = fi_df.sort_values("importance", ascending=True)
        fig_fi = px.bar(
            fi_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="PuBuGn",
            labels={"importance": "Relative importance", "feature": "Signal"},
        )
        fig_fi.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(15,23,42,0.0)",
            plot_bgcolor="rgba(15,23,42,0.0)",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Top Positive Influences")
        st.plotly_chart(
            _plot_skill_contributions(top_contrib), use_container_width=True
        )

    with col_right:
        st.markdown("#### How to Improve (Skill-level)")
        st.plotly_chart(
            _plot_improvement_simulation(improvements), use_container_width=True
        )

    # Details and raw text
    st.markdown(
        """
        <div class="result-card">
            <div class="result-title">Analysis Details</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    with cols[0]:
        token_count = result.get("token_count", None)
        if token_count is not None:
            st.markdown(f"- **Token count**: {int(token_count)}")
        st.markdown(f"- **Target role**: `{role}`")

    with cols[1]:
        coverage = _compute_skill_coverage(top_contrib, missing_skills)
        st.markdown(f"- **Skill coverage**: {coverage:.1f}%")

    # Raw text & exports
    st.markdown("---")
    with st.expander("Show raw extracted resume text"):
        st.text(resume_text)

    export_summary = {
        "role": role,
        "match_percentage": match_percentage,
        "semantic_similarity": semantic_similarity,
        "explanation": explanation,
        "top_contributing_skills": top_contrib,
        "missing_skills": missing_skills,
        "improvements": improvements,
        "token_count": result.get("token_count"),
    }

    col_json, col_pdf = st.columns(2)
    with col_json:
        json_bytes = json.dumps(export_summary, indent=2).encode("utf-8")
        st.download_button(
            "Download JSON Summary",
            data=json_bytes,
            file_name="resume_xai_result.json",
            mime="application/json",
            use_container_width=True,
        )

    with col_pdf:
        pdf_bytes = _export_report_pdf(export_summary)
        if pdf_bytes:
            st.download_button(
                "Download PDF Report",
                data=pdf_bytes,
                file_name="resume_xai_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info(
                "PDF export unavailable (missing `reportlab` or runtime issue). "
                "JSON export is still available."
            )

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    theme = st.session_state["theme"]

    inject_custom_css(theme)

    query_page = st.query_params.get("page", "Analyze")
    page = str(query_page) if query_page in {"Analyze", "Insights", "About"} else "Analyze"
    st.query_params["page"] = page
    render_top_nav(page, theme)

    init_history_db()

    if page == "Analyze":
        # Fast path for home screen: load only dataset/roles.
        job_df, job_embeddings = get_job_data()
        roles = get_unique_roles(job_df)
        quick_roles = get_quick_roles(job_df, n=5)

        if "analyze_view" not in st.session_state:
            st.session_state["analyze_view"] = "home"

        view = st.session_state["analyze_view"]

        if view == "home":
            uploaded_file, role_for_eval, analyze_clicked = render_interaction_card(roles, quick_roles)

            if not analyze_clicked:
                st.markdown(
                    """
                    <div class="landing-hint">
                        Drop a resume and choose a role to generate explainable AI insights.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return

            if uploaded_file is None:
                st.error("Please upload a PDF or DOCX resume before analyzing.")
                return

            if not role_for_eval:
                st.error("Please select or type a target role before analyzing.")
                return

            # Persist inputs and move to loading view
            st.session_state["uploaded_bytes"] = uploaded_file.read()
            st.session_state["uploaded_name"] = getattr(uploaded_file, "name", "")
            st.session_state["selected_role"] = role_for_eval
            st.session_state["analyze_view"] = "loading"
            _safe_rerun()

        elif view == "loading":
            # Loading screen
            st.markdown(
                f"""
                <div class="loading-screen">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Analyzing resume with SkillLens AI…</div>
                    <div class="loading-subtext">
                        File: <span>{st.session_state.get('uploaded_name', 'unknown')}</span> •
                        Target role: <span>{st.session_state.get('selected_role', '')}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            model = get_model()
            # Perform analysis, then transition to results
            try:
                raw_text = extract_text_from_document(
                    st.session_state["uploaded_bytes"],
                    filename=st.session_state.get("uploaded_name", ""),
                )
            except Exception as e:
                st.session_state["analyze_view"] = "home"
                st.error(f"Failed to extract text from document. Details: {e}")
                return

            if not raw_text.strip():
                st.session_state["analyze_view"] = "home"
                st.error("The uploaded document appears to be empty or unreadable.")
                return

            processed_text, token_count = preprocess_resume_text(raw_text)

            try:
                eval_result = evaluate_resume_for_role(
                    resume_text=processed_text,
                    target_role=st.session_state["selected_role"],
                    job_df=job_df,
                    job_embeddings=job_embeddings,
                    model=model,
                )
            except KeyError:
                st.session_state["analyze_view"] = "home"
                st.error(
                    "The selected role could not be found in the job dataset. "
                    "Please choose a different role or refine your input."
                )
                return
            except Exception as e:
                st.session_state["analyze_view"] = "home"
                st.error(f"Evaluation failed due to an internal error: {e}")
                return

            eval_result["token_count"] = token_count

            try:
                top_contrib = compute_skill_contribution(
                    resume_text=processed_text,
                    target_role=st.session_state["selected_role"],
                    model=model,
                    job_df=job_df,
                    job_embeddings=job_embeddings,
                )
            except Exception as e:
                st.session_state["analyze_view"] = "home"
                st.error(f"Failed to compute skill contributions: {e}")
                return

            try:
                missing_skills = [
                    s
                    for s in eval_result.get("missing_skills", [])
                    if isinstance(s, str)
                ]
            except Exception:
                missing_skills = []

            # Fallback missing-skill inference for cleaner XAI output.
            if not missing_skills:
                missing_skills = identify_missing_skills(
                    resume_text=processed_text,
                    target_role=st.session_state["selected_role"],
                    max_missing=8,
                )

            try:
                improvements = simulate_skill_addition(
                    resume_text=processed_text,
                    target_role=st.session_state["selected_role"],
                    model=model,
                    missing_skills=missing_skills,
                    job_df=job_df,
                    job_embeddings=job_embeddings,
                )
            except Exception as e:
                st.session_state["analyze_view"] = "home"
                st.error(f"Failed to simulate skill additions: {e}")
                return

            # Cache results in session and navigate to results view
            st.session_state["analysis_result"] = eval_result
            st.session_state["analysis_resume_text"] = processed_text
            st.session_state["analysis_top_contrib"] = top_contrib
            st.session_state["analysis_missing_skills"] = missing_skills
            st.session_state["analysis_improvements"] = improvements

            # Save searchable analysis history for Insights page.
            history_payload = {
                "result": eval_result,
                "resume_text": processed_text,
                "top_contrib": top_contrib,
                "missing_skills": missing_skills,
                "improvements": improvements,
            }
            save_history_entry(
                created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                file_name=st.session_state.get("uploaded_name", "unknown"),
                role=st.session_state.get("selected_role", ""),
                match_percentage=float(eval_result.get("match_percentage", 0.0)),
                semantic_similarity=float(eval_result.get("semantic_similarity", 0.0)),
                payload=history_payload,
            )

            st.session_state["analyze_view"] = "results"
            _safe_rerun()

        elif view == "results":
            eval_result = st.session_state.get("analysis_result")
            processed_text = st.session_state.get("analysis_resume_text", "")
            top_contrib = st.session_state.get("analysis_top_contrib", [])
            missing_skills = st.session_state.get("analysis_missing_skills", [])
            improvements = st.session_state.get("analysis_improvements", [])
            role_for_eval = st.session_state.get("selected_role", "")

            if not eval_result:
                st.session_state["analyze_view"] = "home"
                _safe_rerun()
                return

            render_results(
                result=eval_result,
                resume_text=processed_text,
                role=role_for_eval,
                top_contrib=top_contrib,
                missing_skills=missing_skills,
                improvements=improvements,
            )

    elif page == "Insights":
        history = list_history_entries(limit=50)
        st.markdown(
            """
            <div class="results-wrapper">
                <div class="result-card">
                    <div class="result-title">Insights • Query History</div>
                    <p class="match-caption">
                        Previously analyzed resume-role queries are listed below. Click any item to open its final summary view.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not history:
            st.markdown(
                """
                <div class="results-wrapper">
                    <div class="result-card">
                        <p class="match-caption">No analysis history yet. Run at least one resume analysis to populate this panel.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="results-wrapper">', unsafe_allow_html=True)
            for i, item in enumerate(history):
                st.markdown('<div class="insight-history-card">', unsafe_allow_html=True)
                c1, c2 = st.columns([4, 1.2])
                with c1:
                    st.markdown(
                        f"**{item['role']}**  \n"
                        f"`{item['file_name']}`  \n"
                        f"Match: **{item['match_percentage']:.1f}%** | Similarity: **{item['semantic_similarity']:.3f}**  \n"
                        f"Time: `{item['created_at']}`"
                    )
                with c2:
                    if st.button("Open Result", key=f"open_hist_{i}", use_container_width=True):
                        history_item = get_history_entry(int(item["id"]))
                        if not history_item:
                            st.warning("Could not load this history record.")
                        else:
                            payload = history_item.get("payload", {})
                            st.session_state["analysis_result"] = payload.get("result", {})
                            st.session_state["analysis_resume_text"] = payload.get("resume_text", "")
                            st.session_state["analysis_top_contrib"] = payload.get("top_contrib", [])
                            st.session_state["analysis_missing_skills"] = payload.get("missing_skills", [])
                            st.session_state["analysis_improvements"] = payload.get("improvements", [])
                            st.session_state["selected_role"] = history_item.get("role", "")
                            st.session_state["analyze_view"] = "results"
                            st.query_params["page"] = "Analyze"
                            _safe_rerun()
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    elif page == "About":
        st.markdown(
            """
            <div class="results-wrapper">
                <div class="result-card">
                    <div class="result-title">About SkillLens AI</div>
                    <p class="match-caption">
                        SkillLens AI is an explainable career intelligence system designed to help candidates understand
                        how their resume aligns with modern job roles.
                    </p>
                    <p class="match-caption">
                        It combines transformer-based semantic similarity, role-aware retrieval, and perturbation-based
                        explainability to provide transparent insights such as match strength, top contributing skills,
                        missing competencies, and improvement pathways.
                    </p>
                    <p class="match-caption">
                        The goal is to make AI-driven career guidance more interpretable, actionable, and trustworthy
                        for students, professionals, and recruiters.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        """
        <div class="app-footer">
            <span>SkillLens AI • Explainable Career Intelligence</span>
            <span>Transformer-based semantic similarity • Perturbation-based explainability</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

