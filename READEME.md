# Daily Deposit Statistics Toolkit

## Overview

This project is a Python-based statistics toolkit built to analyse daily deposit data collected from M-Pesa agents.

The goal is to understand deposit patterns, identify unusual values, and demonstrate how statistical concepts can be implemented using Python without relying on the built-in `statistics` module.

The project also compares the results of the custom calculations with NumPy to check their accuracy.

## What the project does

The toolkit performs the following tasks:

- Calculates the mean, median and mode.
- Calculates sample and population variance.
- Calculates standard deviation.
- Determines the first, second and third quartiles.
- Calculates the interquartile range (IQR).
- Identifies potential outliers using the 1.5 × IQR rule.
- Loads and validates numerical data from a CSV file.
- Compares custom statistical calculations with NumPy.
- Generates a report showing the results before and after outlier removal.

## Dataset

The project uses daily deposit records from M-Pesa agents.

The dataset contains information such as:

- Date
- Agent ID
- Town
- Daily deposits in KES
- Number of transactions

The analysis focuses on the `daily_deposits_kes` column.

Missing, negative, placeholder and incorrectly formatted deposit values are rejected during data loading. Rejected records are documented in the generated report.

Potential outliers are identified separately rather than automatically treated as invalid data.

## Statistical approach

The project implements its statistical calculations using plain Python.

**Central tendency**
- Mean
- Median
- Mode

**Measures of spread**
- Sample variance
- Population variance
- Standard deviation

**Distribution and outlier detection**
- Quartiles using the median-of-halves method
- Interquartile range
- Lower and upper outlier fences using the 1.5 × IQR rule

Sample variance and sample standard deviation are used in the main analysis.

NumPy is used only for verification, not for the primary statistical calculations.

## Project structure

```text
day3-stats-toolkit/
│
├── stats_toolkit.py
├── test_stats_toolkit.py
├── requirements.txt
│
├── data/
│   └── agent_deposits.csv
│
└── output/
    └── stats_report.txt
```

## Getting started

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate into the project folder

```bash
cd stats-toolkit
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

For Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Generate the statistics report

```bash
python stats_toolkit.py
```

The report will be displayed in the terminal and saved to:

```text
output/stats_report.txt
```

## Running the tests

Run the unit tests using:

```bash
python -m unittest -v
```

The tests cover statistical calculations, empty inputs, outlier detection and CSV data validation.

## NumPy verification

The toolkit includes a comparison function that checks the custom calculations against NumPy.

The verification covers:
- Mean
- Median
- Sample variance
- Sample standard deviation
- First and third quartiles

The comparison uses matching calculation conventions, including `ddof=1` for sample variance and standard deviation.

## Key learning outcomes

Through this project, I practised:

- Implementing statistical formulas using Python.
- Working with CSV files and handling inconsistent data.
- Applying data validation and cleaning techniques.
- Understanding statistical distributions and outlier detection.
- Writing and running unit tests.
- Organising a Python project into reusable functions and classes.
- Using NumPy to validate custom implementations.

## Tools and technologies

- Python
- NumPy
- CSV
- Unit Testing (`unittest`)
- Git and GitHub

---

**Project focus:** Statistical computing, data quality, and exploratory data analysis.