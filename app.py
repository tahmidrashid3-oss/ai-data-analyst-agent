
import io
import json
import os
import hashlib

import pandas as pd
import streamlit as st

from agent import ask_agent


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GEMINI API KEY CONFIGURATION
# ============================================================

def configure_api_key():
    """
    Load the Gemini API key from Streamlit Cloud secrets.

    If no Streamlit secret is configured, use the existing
    environment variable. This keeps local development working.
    """

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except (FileNotFoundError, KeyError):
        api_key = None

    if api_key:
        os.environ["GEMINI_API_KEY"] = str(api_key)

    return bool(os.environ.get("GEMINI_API_KEY"))


API_KEY_AVAILABLE = configure_api_key()


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 16px;
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    .dashboard-subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(raw):
    """Read uploaded CSV data and cache the result."""

    return pd.read_csv(io.BytesIO(raw))


def get_dataset_id(raw):
    """Generate a unique identifier for the uploaded dataset."""

    return hashlib.sha256(raw).hexdigest()


def get_missing_percentage(df):
    """Calculate the percentage of missing dataset cells."""

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 0.0

    missing_cells = int(df.isna().sum().sum())

    return (missing_cells / total_cells) * 100


def get_chart_dataframe(values):
    """Convert agent tool results into chart-ready data."""

    if not isinstance(values, dict) or not values:
        return pd.DataFrame()

    chart_df = pd.DataFrame(
        {
            "Category": [str(k) for k in values.keys()],
            "Value": list(values.values()),
        }
    )

    chart_df["Value"] = pd.to_numeric(
        chart_df["Value"],
        errors="coerce",
    )

    return chart_df.dropna(subset=["Value"])


def build_report(answer, question, history):
    """Build a downloadable Markdown analysis report."""

    report = [
        "# AI Data Analyst Report",
        "",
        "## Question",
        question,
        "",
        "## AI Analysis",
        answer,
        "",
        "## Tools Executed",
        "",
    ]

    if not history:
        report.append("No analysis tools were executed.")

    for index, step in enumerate(history, start=1):

        report.extend(
            [
                f"### Step {index}: "
                f"{step.get('tool', 'Unknown')}",
                "",
                "**Arguments**",
                "",
                "```json",
                json.dumps(
                    step.get("arguments", {}),
                    indent=2,
                    default=str,
                ),
                "```",
                "",
                "**Result**",
                "",
                "```json",
                json.dumps(
                    step.get("result", {}),
                    indent=2,
                    default=str,
                ),
                "```",
                "",
            ]
        )

    return "\n".join(report)


# ============================================================
# CHART VISUALISATIONS
# ============================================================

def show_tool_charts(history):
    """Display charts from agent tool results."""

    chart_count = 0

    for index, step in enumerate(history):

        result = step.get("result", {})

        if not isinstance(result, dict):
            continue

        values = result.get("values")

        if not values:
            continue

        chart_df = get_chart_dataframe(values)

        if chart_df.empty:
            continue

        chart_count += 1

        tool_name = step.get("tool", "Analysis")
        metric = result.get("metric", "Value")
        aggregation = result.get("aggregation", "")

        if tool_name == "time_trend":
            st.markdown(f"#### {metric} over time")
        else:
            st.markdown(f"#### {metric} by category")

        if aggregation:
            st.caption(
                f"Tool: {tool_name} | "
                f"Aggregation: {aggregation}"
            )

        default_chart = (
            1 if tool_name == "time_trend" else 0
        )

        chart_type = st.selectbox(
            "Chart type",
            [
                "Bar chart",
                "Line chart",
                "Data table",
            ],
            index=default_chart,
            key=f"chart_type_{index}",
        )

        if chart_type == "Bar chart":

            st.bar_chart(
                chart_df,
                x="Category",
                y="Value",
                horizontal=True,
                sort=False,
                width="stretch",
            )

        elif chart_type == "Line chart":

            st.line_chart(
                chart_df,
                x="Category",
                y="Value",
                width="stretch",
            )

        else:

            st.dataframe(
                chart_df,
                width="stretch",
                hide_index=True,
            )

        with st.expander("View exact chart values"):

            st.dataframe(
                chart_df,
                width="stretch",
                hide_index=True,
            )

        st.download_button(
            label="Download chart data",
            data=chart_df.to_csv(index=False),
            file_name=f"analysis_chart_{index + 1}.csv",
            mime="text/csv",
            key=f"download_chart_{index}",
        )

        st.divider()

    if chart_count == 0:

        st.info(
            "No chart-ready results were returned. "
            "Try asking a question involving category "
            "comparisons or monthly trends."
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 AI Analyst")

    st.caption(
        "Explore datasets using natural-language questions."
    )

    st.divider()

    st.subheader("Dataset")

    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Maximum file size: 5 MB",
    )

    st.divider()

    st.subheader("Example questions")

    examples = [
        "Which product generated the most revenue?",
        "Which category has the lowest average value?",
        "Show the monthly trend in revenue.",
        "Are any numeric columns correlated?",
        "Summarise the dataset.",
    ]

    for index, example in enumerate(examples):

        if st.button(
            example,
            key=f"example_{index}",
            width="stretch",
        ):

            st.session_state["question_input"] = example
            st.session_state["analysis_result"] = None
            st.rerun()

    st.divider()

    if API_KEY_AVAILABLE:
        st.success("Gemini API configured")
    else:
        st.warning("Gemini API key not configured")

    st.caption(
        "Powered by Google Gemini, Python, "
        "Pandas and Streamlit."
    )


# ============================================================
# HEADER
# ============================================================

st.title("AI Data Analyst Agent")

st.markdown(
    '<div class="dashboard-subtitle">'
    "Ask questions about your data and explore "
    "the analysis performed by an AI agent."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# FILE UPLOAD VALIDATION
# ============================================================

if uploaded is None:

    st.info(
        "Upload a CSV dataset using the sidebar "
        "to get started."
    )

    st.stop()


raw = uploaded.getvalue()

if len(raw) > 5_000_000:

    st.error(
        "The uploaded file is too large. "
        "Please select a CSV smaller than 5 MB."
    )

    st.stop()


try:
    df = load_csv(raw)

except Exception as exc:

    st.error(
        f"Could not read the CSV file: {exc}"
    )

    st.stop()


if df.empty:

    st.warning(
        "The uploaded CSV contains no data rows."
    )

    st.stop()


# ============================================================
# DATASET SESSION MANAGEMENT
# ============================================================

dataset_id = get_dataset_id(raw)

if st.session_state.get("dataset_id") != dataset_id:

    st.session_state["dataset_id"] = dataset_id
    st.session_state["analysis_result"] = None


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("Dataset overview")

rows = len(df)
columns = len(df.columns)

numeric_columns = len(
    df.select_dtypes(include="number").columns
)

missing_percentage = get_missing_percentage(df)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total rows",
        value=f"{rows:,}",
    )

with col2:
    st.metric(
        label="Columns",
        value=columns,
    )

with col3:
    st.metric(
        label="Numeric columns",
        value=numeric_columns,
    )

with col4:
    st.metric(
        label="Missing values",
        value=f"{missing_percentage:.1f}%",
    )


# ============================================================
# DATASET TABS
# ============================================================

preview_tab, statistics_tab, quality_tab = st.tabs(
    [
        "Dataset preview",
        "Summary statistics",
        "Data quality",
    ]
)


with preview_tab:

    st.dataframe(
        df.head(100),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        f"Showing up to 100 of {rows:,} rows."
    )

    st.download_button(
        label="Download original dataset",
        data=raw,
        file_name=uploaded.name,
        mime="text/csv",
    )


with statistics_tab:

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] > 0:

        st.dataframe(
            numeric_df.describe().round(2),
            width="stretch",
        )

    else:

        st.info(
            "No numeric columns are available "
            "for summary statistics."
        )


with quality_tab:

    missing = df.isna().sum()

    quality_df = pd.DataFrame(
        {
            "Column": df.columns,
            "Data type": df.dtypes.astype(str).values,
            "Missing values": missing.values,
            "Missing (%)": (
                missing.values / len(df) * 100
            ).round(2),
        }
    )

    st.dataframe(
        quality_df,
        width="stretch",
        hide_index=True,
    )

    st.metric(
        "Duplicate rows",
        int(df.duplicated().sum()),
    )


# ============================================================
# AI ANALYSIS SECTION
# ============================================================

st.divider()

st.subheader("Ask the AI Analyst")

st.caption(
    "Gemini selects appropriate Pandas tools "
    "and explains the results."
)


if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""


question = st.text_input(
    "Your question",
    key="question_input",
    placeholder=(
        "e.g. Which product generated the most revenue?"
    ),
)


analyse_clicked = st.button(
    "Analyse dataset",
    type="primary",
    icon=":material/auto_awesome:",
    width="stretch",
)


# ============================================================
# RUN AI AGENT
# ============================================================

if analyse_clicked:

    st.session_state["analysis_result"] = None

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    elif not API_KEY_AVAILABLE:

        st.error(
            "Gemini API key is missing. "
            "Configure GEMINI_API_KEY in Streamlit Cloud "
            "Secrets or your local environment."
        )

    else:

        with st.spinner(
            "The AI agent is analysing your dataset..."
        ):

            try:

                answer, history = ask_agent(
                    df,
                    question,
                )

                st.session_state["analysis_result"] = {
                    "dataset_id": dataset_id,
                    "question": question,
                    "answer": answer,
                    "history": history,
                }

            except Exception as exc:

                st.error(
                    f"Analysis failed: {exc}"
                )


# ============================================================
# DISPLAY RESULTS
# ============================================================

analysis = st.session_state.get("analysis_result")

if analysis and analysis.get("dataset_id") == dataset_id:

    st.divider()

    st.subheader("AI analysis results")

    st.markdown(
        analysis["answer"]
    )

    st.caption(
        f"Question: {analysis['question']}"
    )

    st.divider()

    results_tab, trace_tab = st.tabs(
        [
            "Visualisations",
            "Agent execution trace",
        ]
    )

    with results_tab:

        show_tool_charts(
            analysis["history"]
        )

    with trace_tab:

        history = analysis["history"]

        if not history:

            st.info(
                "The agent did not execute any tools."
            )

        else:

            st.caption(
                f"{len(history)} tool call(s) executed."
            )

            for index, step in enumerate(
                history,
                start=1,
            ):

                with st.expander(
                    f"Step {index}: "
                    f"{step.get('tool', 'Unknown')}",
                    expanded=(index == 1),
                ):

                    st.markdown("**Arguments**")

                    st.json(
                        step.get("arguments", {})
                    )

                    st.markdown("**Tool result**")

                    st.json(
                        step.get("result", {})
                    )

    # --------------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------------

    st.divider()

    st.subheader("Export analysis")

    report = build_report(
        answer=analysis["answer"],
        question=analysis["question"],
        history=analysis["history"],
    )

    st.download_button(
        label="Download analysis report",
        data=report,
        file_name="ai_analysis_report.md",
        mime="text/markdown",
        icon=":material/download:",
        width="stretch",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Data Analyst Agent | "
    "Built with Python, Pandas, Streamlit and Google Gemini"
)
