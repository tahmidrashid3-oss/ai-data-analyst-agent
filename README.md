# 📊 AI Data Analyst Agent

**An AI-powered data analysis application built with Python, Google Gemini, Pandas and Streamlit.**

AI Data Analyst Agent enables users to upload CSV datasets and explore their data using natural-language questions. Instead of manually writing Python code for each analysis, users can ask questions and receive AI-generated explanations, statistical summaries and interactive visualisations.

The application uses **Google Gemini function calling** to select appropriate Python analysis tools, execute calculations using Pandas and interpret the results.

This project demonstrates the practical integration of **agentic AI, data analytics, API integration and interactive web application development**.

## 🔗 Project Links

- **Live Application:** [AI Data Analyst Agent](https://ai-data-analyst-agent-a5tzr4se99ayc3appy2rzsa.streamlit.app)
- **GitHub Repository:** [View Source Code](https://github.com/tahmidrashid3-oss/ai-data-analyst-agent)

---

## ✨ Key Features

### 🤖 AI-Powered Analysis

- Ask questions about CSV datasets using natural language.
- Use Google Gemini to interpret questions and select appropriate analysis tools.
- Execute calculations using Python and Pandas.
- Generate explanations based on calculated results.
- Support multi-step tool execution for more complex analysis requests.

### 📈 Data Exploration and Visualisation

- Upload CSV datasets directly through the web interface.
- View dataset dimensions, column types and descriptive statistics.
- Identify missing values and duplicate records.
- Calculate grouped summaries and aggregations.
- Explore correlations between numerical variables.
- Analyse trends over time.
- Generate interactive bar charts, line charts and data tables.

### 🔍 Transparent AI Execution

- Inspect which analysis tools the AI agent selected.
- View function arguments and execution results.
- Examine the sequence of tool calls.
- Understand how calculations contribute to the final response.

### 📥 Data Export

- Download the original uploaded dataset.
- Export chart data as CSV files.
- Download Markdown reports containing the question, AI analysis and tool execution history.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Google Gemini API | Natural-language understanding and function calling |
| Google Gen AI SDK | Communication with Gemini models |
| Pandas | Data processing, aggregation and statistical analysis |
| Streamlit | Interactive web interface and visualisations |
| Git & GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Web application deployment |

---

## 🧠 How the AI Agent Works

The application combines the reasoning capabilities of a large language model with deterministic Python calculations.

Gemini interprets the user's question and selects an appropriate analysis function. The selected function is executed using Pandas, and the results are returned to Gemini to generate a natural-language explanation.

### Agent Architecture

```text
             ┌─────────────────────────┐
             │       User Input        │
             │  CSV File + Question    │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │   Streamlit Interface   │
             │  Dataset Exploration    │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │      Google Gemini      │
             │  Question Interpretation│
             │     & Tool Selection    │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │   Python Analysis Tools │
             │      Pandas Engine      │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │    Analysis Results     │
             │   Returned to Gemini    │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │     Final Response      │
             │  Explanation + Charts   │
             │   + Execution History   │
             └─────────────────────────┘
```

### Analysis Workflow

1. **Dataset Upload:** The user uploads a CSV file, which is loaded into a Pandas DataFrame.
2. **Question Interpretation:** Gemini receives the user's question alongside information about the dataset.
3. **Tool Selection:** Gemini determines which predefined Python function is appropriate and generates the required arguments.
4. **Tool Execution:** The application executes the selected function using Pandas.
5. **Result Interpretation:** The calculated results are returned to Gemini, which generates an explanation.
6. **Visualisation:** Streamlit presents the analysis, charts and tool execution history.

The agent can perform multiple tool calls when required, allowing it to combine supported analysis operations.

### Why Function Calling?

Rather than relying on an LLM to calculate statistics directly, the application delegates numerical calculations to predefined Python functions.

This design provides several advantages:

- **Accuracy:** Calculations are performed using Pandas rather than generated estimates.
- **Transparency:** Tool arguments and results can be inspected.
- **Control:** The agent can execute only the analysis functions exposed to it.
- **Modularity:** Analysis tools are separated from the user interface and LLM interaction.

This separation of responsibilities is a central part of the application's agentic architecture.

---

## 🔧 Available Analysis Tools

The AI agent has access to four predefined analysis tools.

| Tool | Functionality |
|---|---|
| `dataset_overview` | Provides dataset dimensions, column information, missing values and summary statistics |
| `group_summary` | Groups data by a categorical variable and calculates aggregations |
| `correlations` | Calculates correlations between numerical columns |
| `time_trend` | Aggregates numerical values over time to identify trends |

These tools are implemented using Pandas and selected dynamically by Gemini based on the user's question.

---

## 💬 Example Analysis

The repository includes a sample sales dataset for testing the application.

### Example Question

> Which product generated the most revenue?

### Example Result

Using the included sample dataset, the application identifies the following revenue totals:

| Product | Total Revenue |
|---|---:|
| Laptop | 9,600 |
| Keyboard | 1,200 |
| Mouse | 860 |

**AI Interpretation:**

Laptop generated the highest total revenue at 9,600, substantially exceeding the revenue generated by Keyboard and Mouse.

The agent uses the `group_summary` tool to calculate revenue totals by product. Streamlit then presents the results and allows users to visualise the comparison.

### Other Example Questions

- Summarise the dataset.
- Which category has the lowest average value?
- Show the monthly trend in revenue.
- Are any numeric columns correlated?
- Which product generated the highest total sales?

The available analysis depends on the uploaded dataset's columns and data types.

---

## 📁 Project Structure

```text
ai-data-analyst-agent/
│
├── app.py              # Streamlit dashboard
├── agent.py            # Gemini agent and analysis tools
├── requirements.txt    # Python dependencies
├── sample_sales.csv    # Example dataset
├── README.md           # Project documentation
└── .gitignore          # Git exclusion rules
```

### Main Components

**`agent.py`**

Contains the Gemini API integration, function-calling logic and predefined Pandas analysis tools.

It handles tool selection, function execution, result processing and response generation.

**`app.py`**

Implements the Streamlit dashboard, including CSV uploads, dataset exploration, natural-language questions, interactive charts, agent execution history and report downloads.

---

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or later
- A Google Gemini API key
- An internet connection

### 1. Clone the Repository

```bash
git clone https://github.com/tahmidrashid3-oss/ai-data-analyst-agent.git

cd ai-data-analyst-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API Key

Obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/api-keys).

Set the API key as an environment variable.

**Windows PowerShell:**

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

**macOS / Linux:**

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

Replace `YOUR_API_KEY` with your actual Gemini API key.

**Security:** Never commit API keys or other credentials to a public GitHub repository.

### 5. Run the Application

```bash
python -m streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

Upload a CSV dataset and start exploring your data.

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud** and connected to its GitHub repository.

The deployment process involves:

1. Hosting the application source code on GitHub.
2. Connecting the repository to Streamlit Community Cloud.
3. Selecting `app.py` as the application entry point.
4. Configuring the Gemini API key using Streamlit's secrets management.
5. Deploying the application and accessing it through a public URL.

### Streamlit Secrets Configuration

For cloud deployment, configure the following secret in the Streamlit application settings:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

The application loads the key securely without including it in the source code.

---

## 🔐 Reliability and Security

The application incorporates several measures to improve reliability and maintain control over agent execution.

**Controlled Tool Execution**

Gemini selects from predefined analysis functions rather than executing arbitrary Python code generated by the model.

**API Error Handling**

The Gemini integration includes retry handling for temporary service errors and support for configured model fallbacks.

**Dataset Validation**

Uploaded files are checked for basic validity, including file size and empty datasets.

**Credential Management**

API credentials are stored using environment variables or Streamlit secrets rather than being embedded directly in application code.

**Execution Transparency**

Tool calls, arguments and results are recorded and made available through the application interface.

---

## ⚠️ Limitations

The application is designed around a defined set of data analysis tools.

- CSV uploads are limited to 5 MB.
- Analysis capabilities depend on the available tools and dataset structure.
- Some questions may require calculations that are not currently supported.
- Gemini API availability and usage limits can affect performance.
- AI-generated explanations should be checked against the calculated results.

---

## 🎯 Project Objectives and Technical Skills

This project demonstrates how generative AI can be integrated with traditional data analysis workflows to create an interactive analytical application.

Key technical achievements include:

- Developed an AI agent using Google Gemini function calling.
- Implemented Pandas-based tools for data aggregation, descriptive statistics, correlation analysis and time-series summaries.
- Built an interactive Streamlit dashboard for data exploration and visualisation.
- Designed a multi-step workflow connecting LLM reasoning with Python calculations.
- Implemented API error handling and retry mechanisms.
- Added tool execution tracing to improve transparency.
- Deployed a functional application using Streamlit Community Cloud.
- Used Git and GitHub for version control and application deployment.

The project highlights practical skills in **Python, data analysis, generative AI, agentic workflows, API integration and cloud deployment**.

---

**Built with Python, Google Gemini, Pandas and Streamlit.**