# ARG Data and Modeling – No-Code Data Science Platform

## Overview
This platform empowers data scientists to perform data analysis, preprocessing, and model building without writing manual code. It integrates drag-and-drop modules for data cleaning, feature engineering, visualization, and model training, enabling faster experimentation and improved productivity.

## Key Features
- **No-Code Interface**: Perform data analysis, preprocessing, and model building without manual coding.
- **Drag-and-Drop Modules**: intuitive modules for data cleaning, feature engineering, visualization, and model training.
- **Machine Learning Workflows**: Supports supervised and unsupervised models (Linear Regression, Logistic Regression, etc.).
- **Backend Power**: Powered by robust libraries including TensorFlow, PyTorch, Pandas, and NumPy.
- **Visualization**: User-friendly interface with visualization capabilities using Matplotlib and Seaborn.
- **Cloud Integration**: Enables scalable data processing and model deployment.
- **Accessibility**: Combines automation with customization for both beginners and advanced users.

## Installation

1. Clone the repository or download the source code.
2. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Project Structure

- **`modelfunction.ipynb`**: Core notebook containing model definitions and diagnostic functions (Linear Regression, diagnostics, etc.).
- **`sqltocsv.py` / `updated_sqltocsvcodes.py`**: Scripts to connect to a MySQL database and export tables to CSV format.
- **`MySQL to csv.ipynb`**: Notebook demonstrating database connection and data extraction.
- **`Checking of data to model.ipynb`**: Notebook for initial data checking and model selection logic.

## Usage

1. **Data Extraction**: Use `sqltocsv.py` to extract data from your MySQL database.
2. **Analysis & Modeling**: Open `modelfunction.ipynb` to run model diagnostics and training.

## Requirements
See `requirements.txt` for the full list of dependencies. Major libraries include:
- pandas
- numpy
- pymysql
- scikit-learn
- statsmodels
- matplotlib
- seaborn
- tensorflow
- torch
