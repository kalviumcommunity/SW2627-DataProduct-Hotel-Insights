# Hotel Occupancy Volatility Analysis

## Project Description

This project aims to analyze hotel booking data to identify which customer segments contribute most to occupancy volatility. By analyzing booking trends, cancellation history, and seasonal pricing records, the project provides actionable business insights to help hotel revenue managers optimize pricing strategies, reduce cancellations, and improve occupancy rates.

The final deliverable is an interactive Streamlit dashboard that presents key performance indicators (KPIs), customer segment analysis, booking trends, cancellation patterns, and business recommendations.

---

# Setup

## Prerequisites

- Python 3.10 or later
- Git
- pip

---

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
```

### Windows

```cmd
python -m venv .venv
```

---

## 3. Activate the Virtual Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows (Command Prompt)

```cmd
.venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Run the Streamlit Application

```bash
streamlit run app.py
```

---

# Project Structure

```
hotel-occupancy-analysis/
│
├── data/
│   ├── raw/                  # Original datasets collected from the source.
│   └── processed/            # Cleaned datasets used for analysis.
│
├── notebooks/                # Jupyter notebooks for EDA and experimentation.
│
├── src/                      # Source code for data cleaning, analysis, and utilities.
│
├── dashboard/                # Streamlit application and dashboard components.
│
├── reports/                  # Business insights, findings, and documentation.
│
├── images/                   # Images used in documentation and dashboard.
│
├── requirements.txt          # Python project dependencies.
│
├── .env.example              # Example environment variable configuration.
│
├── README.md                 # Project documentation.
│
└── .gitignore                # Files and folders ignored by Git.
```

### Directory Purpose

| Directory | Purpose |
|-----------|---------|
| **data/raw** | Stores the original datasets without modification. |
| **data/processed** | Stores cleaned and transformed datasets for analysis. |
| **notebooks** | Contains Jupyter notebooks used for exploratory data analysis and experimentation. |
| **src** | Contains reusable Python scripts for data processing and analysis. |
| **dashboard** | Contains the Streamlit dashboard and UI components. |
| **reports** | Stores project reports, insights, and documentation. |
| **images** | Stores images used in documentation or the dashboard. |

---

# Notes

## Environment Variables

This project uses environment variables for configuration.

Before running the project:

1. Copy the `.env.example` file.

### macOS / Linux

```bash
cp .env.example .env
```

### Windows

```cmd
copy .env.example .env
```

2. Open the newly created `.env` file.

3. Replace the placeholder values with your own configuration.

Example:

```
DATABASE_URL=your_database_url
DATASET_PATH=your_dataset_path
STREAMLIT_SERVER_PORT=8501
```

> **Important**
>
> - Commit **`.env.example`** to Git.
> - **Do NOT commit `.env`**, as it contains your personal configuration.

---

# Team

- **Project Admin:** Vantakula Durga Sai Mukesh
- **Team Member:** P. Sravani

---

# License

This project is developed for educational purposes as part of the Data Analytics Sprint.