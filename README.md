# AI Data Analyst Agent (Google Gemini)

A small, agentic Python portfolio project. Upload a CSV, ask a question, and Gemini chooses safe Pandas analysis tools. The Streamlit app displays the answer, tool execution trace, and charts.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GEMINI_API_KEY="YOUR_KEY_HERE"
streamlit run app.py
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey). Do not commit keys to GitHub. The free tier is subject to quotas and availability.

## Try it

Upload `sample_sales.csv` and ask:
- Which product has the lowest total revenue?
- How does revenue change over time?
- Which numerical columns are correlated?

## Architecture

Streamlit UI → Gemini tool selection → allowlisted Pandas tools → Gemini explanation → charts and execution trace.

The agent can call `dataset_overview`, `group_summary`, `correlations`, and `time_trend`. It does **not** execute model-generated Python code. Tool use is bounded to five rounds.

## CV description

Built a Gemini-powered AI data analyst agent using Python, Pandas, Streamlit and function calling, enabling natural-language dataset exploration and transparent tool execution.

## Notes

The code has been syntax-checked but requires a working Gemini API key for end-to-end verification. Use non-sensitive datasets, since questions and analysis results are sent to the Gemini API.
