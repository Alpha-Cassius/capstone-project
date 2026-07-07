# Capstone Project: End-to-End Data Cleaning and EDA

## Executive Summary
This repository contains a capstone-level data engineering and exploratory data analysis (EDA) pipeline. The project is designed with professional software engineering principles: modular codebase, logging, comprehensive dependency management (`requirements.txt`), and reproducible data workflows. We have utilized the Titanic dataset (modified to mimic real-world raw data from a client) to demonstrate automated data cleaning, anomaly detection, statistical imputation, and advanced feature correlation analysis.

## Repository Structure
```
├── data/
│   ├── raw/                 # Original, immutable client data (raw_client_data.csv)
│   └── processed/           # Cleaned data ready for modeling (cleaned_data.csv)
├── notebooks/
│   └── Capstone_EDA.ipynb   # Presentation-ready Jupyter Notebook with full narrative
├── reports/
│   └── figures/             # Auto-generated visualization PNGs
├── src/                     # Modular, testable Python scripts
│   ├── data_cleaning.py
│   ├── eda_analysis.py
│   ├── visualizations.py
│   └── main.py              # Orchestration script
├── prep_data.py             # Script to generate the synthetic raw dataset
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Execute Pipeline**
   ```bash
   python src/main.py
   ```
   This will run the entire orchestration pipeline from reading the raw dataset, cleaning it, outputting logs, generating figures, and saving the processed dataset.
3. **Explore Notebook**
   Open `notebooks/Capstone_EDA.ipynb` in Jupyter or VSCode to walk through the narrative and interactive analysis.

---

## Dataset Description and Justification
The dataset chosen for this analysis is a slightly modified version of the classic Titanic dataset (provided in `data/raw/`). It contains information about passengers, including their demographics, ticket class, and whether they survived. The raw dataset effectively mimics real-world client data that includes common data entry issues, such as duplicate rows, incorrectly formatted numeric strings (e.g., dollar signs on fares), and missing values across multiple columns. This makes it an ideal candidate for demonstrating end-to-end data cleaning, imputation, and exploratory data analysis (EDA).

## Skewness Interpretation
The column with the highest absolute skewness is **Fare**, which exhibits a strong positive skew (skewness ≈ 4.90). A positive skew means that the distribution has a long right tail; while most passengers paid a lower fare, a small number of passengers paid exceptionally high fares. 
Because of this right-skewed distribution, the mean is pulled upward by these extreme high values and is not representative of a "typical" fare. Therefore, imputing missing values with the **median** is a much more robust strategy, as the median represents the true central tendency of the majority of the data.

## IQR Outlier Analysis Interpretation
We performed outlier detection using the Interquartile Range (IQR) method on two numeric columns:
1. **Age**: The IQR bounds were approximately [2.50, 54.50]. We found 66 outliers. 
2. **Fare**: The IQR bounds were approximately [-25.95, 64.37]. We found 105 outliers.

**Handling Strategy for Part 2**: We will **retain** these outliers rather than dropping them. In this context, older passengers and high-priced first-class tickets are genuine data points, not errors. If we apply linear models in Part 2 that are sensitive to extreme values, we may apply a capping strategy (Winsorization) or a log transformation, but we will not drop the rows to avoid losing valuable signal.

## Visualizations Interpretation

### Scatter Plot (Age vs Fare)
The scatter plot visualizes the relationship between passenger Age and Fare. The plot shows very little apparent correlation between the two variables; fares are widely spread across all age groups. There is no clear linear direction, and the approximate strength of the relationship is extremely weak. 

### Box Plot (Age split by Survived)
The box plot splits the Age distribution by survival status (0 = No, 1 = Yes). The visible spread and interquartile ranges between the two groups are fairly similar, but the median age of those who survived is slightly lower. Additionally, there is a visible extension of the lower whisker in the "Survived" group, indicating that younger passengers (children) were prioritized and had a higher likelihood of survival.

### Histogram (Fare)
The histogram of the Fare column confirms our skewness calculation. The distribution is heavily right-skewed (positively skewed). The vast majority of the data falls in the first bin (low fares), with a long, thin tail extending towards the higher fare values.

### Correlation Heat Map
The heat map visualizes the Pearson correlation between all numeric variables. 
The pair of variables with the highest absolute correlation is **Pclass and Fare** (r ≈ -0.55). 
**Explanation**: This correlation is a negative relationship, meaning that a lower Pclass number (1st class) is associated with a higher Fare. This strongly indicates a causal relationship: purchasing a premium ticket class inherently costs more money. A plausible alternative explanation (third variable) could be the "Port of Embarkation" or "Distance Traveled" — if 1st class cabins were mostly booked for longer, more expensive legs of the journey, this could also drive both class proportions and total fare prices.

## Imputation Strategy Comparison
For the two highest-skewness columns (`Fare` and `SibSp`), we computed the mean and median before applying imputation:
- **Fare BEFORE imputation**: Mean = 32.09, Median = 14.45
- **SibSp BEFORE imputation**: Mean = 0.52, Median = 0.00

**Justification**: Both columns are positively skewed. As seen with Fare, the mean (32.09) is more than double the median (14.45) due to the upward pull of extreme high values (e.g., $512 tickets). Similarly, SibSp has a few instances of large families that pull the mean up. Because the mean is heavily distorted by extreme values in positively skewed distributions, the **median** is a much more representative measure of the central tendency. Therefore, the median was chosen and applied to fill the remaining nulls in these columns, which was confirmed successfully (`isnull().sum() == 0`).

## Spearman Rank vs Pearson Correlation
The three column pairs with the largest absolute difference between Spearman and Pearson correlations are:
1. **(Fare, SibSp)**
2. **(Fare, Parch)**
3. **(Fare, Pclass)**

**Interpretation**:
- For **(Fare, SibSp)** and **(Fare, Parch)**, the Spearman correlation is significantly higher than Pearson (|Spearman| > |Pearson|). This indicates that the relationship is **monotonic but non-linear**: as the number of siblings/spouses or parents/children increases, the total fare generally increases (likely due to buying group tickets), but it does not increase at a strict proportional/linear rate.
- For **(Fare, Pclass)**, |Spearman| is also higher than |Pearson|. The relationship is monotonic (better classes cost strictly more) but non-linear (the price jump from 2nd to 1st class is much larger than 3rd to 2nd).

**Feature Selection Guidance**: For Part 2, we will rely on the **Spearman correlation** for these variables. Because many of our features exhibit skewed distributions and monotonic, non-linear relationships, Spearman is a more robust indicator of true predictive signal than Pearson, which assumes strict linearity.

## Grouped Aggregation
We grouped by `Pclass` (categorical) and aggregated `Fare` (numeric):
- **Highest Mean Group**: Pclass 1 (Mean = 76.66)
- **Highest Standard Deviation Group**: Pclass 1 (Std = 78.89)

**Modeling Implication**: The high within-group standard deviation in Pclass 1 is a concern for a predictive model using this categorical feature alone. The high variance implies that knowing a passenger is in Pclass 1 is insufficient to reliably predict their exact fare, as fares in 1st class range from relatively low to over $500. The feature alone lacks the precision needed without other interacting variables.

**Mean Ratio**: The ratio of the highest group mean to the lowest group mean is 76.66 / 13.86 ≈ **5.53**. This ratio is very large, strongly suggesting that `Pclass` carries significant predictive signal for `Fare`, despite the high within-group variance.
