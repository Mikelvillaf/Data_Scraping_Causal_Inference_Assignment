# Data_Scraping_Causal_Inference_Assignment
Joint project for our Data Science Master's course on Augmented Data Management.


# Goal: Build a MercadoLibre Solar Cream Scraper and Analyzer Notebook

## 1. Overview

This project consists of a Python-based web scraper designed to collect product data for solar creams from MercadoLibre Mexico (`mercadolibre.com.mx`). It systematically navigates through search result pages, extracts detailed information for each product, and saves this data into a CSV file.

Additionally, the notebook contains functionalities for data cleaning and preliminary analysis, with a focus on Propensity Score Matching (PSM) to assess the impact of certain product features (e.g., "FULL" shipping) on outcomes like sales or ratings.

## 2. Features

### 2.1. Web Scraping
* **Automated Data Collection**: Navigates MercadoLibre search result pages for solar cream products.
* **Detailed Data Extraction**: For each product, it scrapes:
    * Title
    * Product URL
    * Current Price
    * Old Price (if available)
    * Discount Percentage (if applicable)
    * Offer Type (e.g., "MÁS VENDIDO", "OFERTA DEL DÍA")
    * Average Rating
    * Review Count
    * Seller Information (if available)
    * Shipping Status (e.g., "Envío gratis")
* **Pagination Handling**: Iterates through multiple pages of search results.
* **Cookie Banner Dismissal**: Attempts to automatically dismiss cookie consent banners.
* **Robustness**: Includes retry mechanisms and page loading checks to handle common web issues.
* **Data Output**: Saves all scraped data into a structured CSV file.

### 2.2. Data Analysis (Propensity Score Matching - PSM)
* **Data Cleaning**: Functions to prepare the scraped data for analysis (e.g., converting price strings to numeric).
* **Treatment Definition**: Allows defining a treatment variable (e.g., products with "FULL" shipping).
* **Covariate Selection**: Identifies relevant covariates for PSM.
* **Propensity Score Estimation**: Uses logistic regression to estimate propensity scores.
* **Matching**: Implements nearest neighbor matching based on propensity scores.
* **Balance Checking**:
    * Calculates Standardized Mean Differences (SMD) for covariates.
    * Performs t-tests (for continuous covariates) and Chi-squared tests (for binary covariates) to assess balance before and after matching.
* **Outcome Analysis**: (Intended for) analyzing the effect of the treatment on an outcome variable after matching.

## 3. Dependencies

The script relies on several Python libraries. You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn selenium webdriver-manager scikit-learn statsmodels scipy