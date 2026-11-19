
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


def format_dataframe(df):
    """Return a copy suitable for display."""

    return df.copy()


def get_missing_percentage(df):
    """Calculate overall missing-value percentage."""

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 0.0

    missing_cells = int(df.isna().sum().sum())

    return (missing_cells / total_cells) * 100


def get_chart_dataframe(values):
    """Convert an agent tool result into chart-ready data."""

    if not isinstance(values, dict) or not values:
        return pd.DataFrame()

    chart_df = pd.DataFrame(
        {
            "Category": list(values.keys()),
            "Value": list(values.values()),
        }
    )

    chart_df["Value"] = pd.to_numeric(
        chart_df["Value"],
        errors="coerce",
    )

    return chart_df.dropna(subset=["Value"])


def download_analysis(answer, question, history):
    """Build a downloadable Markdown report."""

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
                f"### Step {index}: {step.get('tool', 'Unknown')}",
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


def show_tool_charts(history):
    """Display charts generated from agent tool results."""

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

        st.markdown(
            f"#### {metric} by category"
            if tool_name == "group_summary"
            else f"#### {metric} over time"
        )

        if aggregation:
            st.caption(
                f"Tool: {tool_name} | "
                f"Aggregation: {aggregation}"
            )

        # Allow the user to choose a visualisation.
        chart_type = st.selectbox(
            "Chart type",
            ["Bar chart", "Line chart", "Data table"],
            index=(
                1 if tool_name == "time_trend" else 0
            ),
            key=f"chart_type_{index}",
        )

        if chart_type == "Bar chart":

            st.bar_chart(
                chart_df.set_index("Category"),
                y="Value",
                horizontal=True,
                sort=False,
                width="stretch",
            )

        elif chart_type == "Line chart":

            st.line_chart(
                chart_df.set_index("Category"),
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

        csv_data = chart_df.to_csv(index=False)

        st.download_button(
            label="Download chart data",
            data=csv_data,
            file_name=f"analysis_chart_{index + 1}.csv",
            mime="text/csv",
            key=f"download_chart_{index}",
        )

    if chart_count == 0:
        st.info(
            "No chart-ready results were returned. "
            "Try asking a question involving a category "
            "comparison or a time trend."
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

    for example in examples:
        if st.button(
            example,
            key=f"example_{example}",
            width="stretch",
        ):
            st.session_state["question_input"] = example

    st.divider()

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
# FILE VALIDATION
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
        format_dataframe(df.head(100)),
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
    "Gemini selects the appropriate Pandas tools "
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

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    elif not os.environ.get("GEMINI_API_KEY"):

        st.error(
            "GEMINI_API_KEY is missing. "
            "Set your API key before running the agent."
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

if analysis:

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

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    with results_tab:

        show_tool_charts(
            analysis["history"]
        )

    # --------------------------------------------------------
    # AGENT TRACE
    # --------------------------------------------------------

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
                    f"Step {index}: {step.get('tool', 'Unknown')}",
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

    report = download_analysis(
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
