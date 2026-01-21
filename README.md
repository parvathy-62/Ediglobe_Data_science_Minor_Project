# Ediglobe_Data_science_Minor_Project
COVID-19 Early Case Trend Analysis &amp; Recovery Insights

# COVID-19 Early Case Trend Analysis & Recovery Insights

## Company
**HealthGuard Analytics Pvt. Ltd.**  
A healthcare data analytics firm providing insights to government health departments and hospitals.

## Business Context
HealthGuard Analytics Pvt. Ltd. partnered with a national public health authority to analyze early-stage
infectious disease case data. The goal is to understand patient demographics, infection sources,
recovery patterns, and regional trends during the initial phase of the COVID-19 outbreak.

The dataset contains patient-level records including confirmed cases, demographic details,
infection history, and outcomes such as released, isolated, or deceased.

## Problem Statement
The public health authority requires data-driven insights to answer the following questions:

1. **Who is getting infected?**  
   Analysis of age, gender, and regional demographics.

2. **How are infections spreading?**  
   Identification of infection reasons, infection order, and contact exposure levels.

3. **What are the recovery trends?**  
   Evaluation of time taken from confirmation to recovery.

4. **Which regions are most impacted?**  
   Comparison of confirmed and released cases across regions.

5. **What factors influence recovery time?**  
   Exploration of age, contact number, and infection order using statistical analysis.

## Analytical Objectives
- Perform Exploratory Data Analysis (EDA)
- Analyze missing values and data quality
- Apply descriptive statistics to summarize patient data
- Visualize trends related to:
  - Gender distribution
  - Age distribution
  - Regional case concentration
  - Infection sources
  - Recovery timelines
- Calculate recovery duration from confirmed and released dates
- Apply Linear Regression to:
  - Predict recovery time
  - Identify statistically significant influencing factors

## Tech Stack
**Programming Language:** Python  

**Libraries Used:**
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Statsmodels

**Statistical Methods:**
- Descriptive Statistics
- Correlation Analysis
- Linear Regression


## Dataset Description
The dataset contains the following attributes:

- **Demographics:** sex, birth_year, country, region  
- **Infection Details:** infection_reason, infection_order, infected_by  
- **Exposure Metrics:** contact_number  
- **Timeline Data:** confirmed_date, released_date, deceased_date  
- **Case Outcome:** state (released, isolated, deceased)

Dataset source:  
https://drive.google.com/file/d/1TXoqikmE0S3LGem8IgGktaJJyZPGMgN/view

## Key Insights
- Gender and age distributions highlight vulnerable population groups
- Certain regions show significantly higher case concentration
- Close-contact transmission is a major infection source
- Average recovery time is approximately 15 days
- Linear regression shows weak correlation between recovery time and selected predictors,
  indicating recovery is influenced by multiple unobserved factors

## Project Structure
ediglobe/
│── project.py
│── patient.csv
│── requirements.txt
│── README.md
│── Figure_1.png
│── Figure_2.png
│── Figure_3.png
│── Figure_4.png
│── Figure_5.png
│── Figure_6.png

## Expected Outcomes
Clear understanding of early outbreak infection patterns
Identification of high-risk demographic groups
Insights into recovery timelines
Visual evidence to support data-driven public health decisions

## Optional Extensions
Feature engineering for recovery duration
Advanced regression models
Model evaluation using R² score and residual analysis

## sample code outputs
**STEP 1: Data Loading and Understanding**

**Dataset loaded using Pandas**
Displayed first 5 rows
 id     sex  birth_year  ... released_date deceased_date     state
0   1  female      1984.0  ...    2020-02-06           NaN  released
1   2    male      1964.0  ...    2020-02-05           NaN  released
2   3    male      1966.0  ...    2020-02-12           NaN  released
3   4    male      1964.0  ...    2020-02-09           NaN  released
4   5    male      1987.0  ...           NaN           NaN  isolated

[5 rows x 14 columns]

**Used info() and isnull() to understand structure and missing values**
DatasetInfo:

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4212 entries, 0 to 4211
Data columns (total 14 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   id                4212 non-null   int64  
 1   sex               318 non-null    object 
 2   birth_year        292 non-null    float64
 3   country           4212 non-null   object 
 4   region            305 non-null    object 
 5   group             76 non-null     object 
 6   infection_reason  130 non-null    object 
 7   infection_order   35 non-null     float64
 8   infected_by       62 non-null     float64
 9   contact_number    32 non-null     float64
 10  confirmed_date    4212 non-null   object 
 11  released_date     28 non-null     object 
 12  deceased_date     13 non-null     object 
 13  state             4212 non-null   object 
dtypes: float64(4), int64(1), object(9)
memory usage: 460.8+ KB
None

Missing Values:

id                     0
sex                 3894
birth_year          3920
country                0
region              3907
group               4136
infection_reason    4082
infection_order     4177
infected_by         4150
contact_number      4180
confirmed_date         0
released_date       4184
deceased_date       4199
state                  0
dtype: int64


**STEP 2: Data Cleaning & Preprocessing**

Converted date columns:confirmed_date,released_date & deceased_date
Handled missing values by:Ignoring rows where recovery time cannot be calculated
Created a new feature:recovery_days = released_date - confirmed_date

 confirmed_date released_date deceased_date
0     2020-01-20    2020-02-06           NaT
1     2020-01-24    2020-02-05           NaT
2     2020-01-26    2020-02-12           NaT
3     2020-01-27    2020-02-09           NaT
4     2020-01-30           NaT           NaT
  confirmed_date released_date  recovery_days
0     2020-01-20    2020-02-06           17.0
1     2020-01-24    2020-02-05           12.0
2     2020-01-26    2020-02-12           17.0
3     2020-01-27    2020-02-09           13.0
4     2020-01-30           NaT            NaN

Recovery Days Sample:

       state confirmed_date released_date  recovery_days
0   released     2020-01-20    2020-02-06           17.0
1   released     2020-01-24    2020-02-05           12.0
2   released     2020-01-26    2020-02-12           17.0
3   released     2020-01-27    2020-02-09           13.0
4   isolated     2020-01-30           NaT            NaN
5   released     2020-01-30    2020-02-19           20.0
6   released     2020-01-30    2020-02-15           16.0
7   released     2020-01-31    2020-02-12           12.0
8   released     2020-01-31    2020-02-24           24.0
9   released     2020-01-31    2020-02-19           19.0
10  released     2020-01-31    2020-02-10           10.0
11  released     2020-02-01    2020-02-18           17.0
12  released     2020-02-02    2020-02-24           22.0
13  released     2020-02-02    2020-02-18           16.0
14  released     2020-02-02    2020-02-24           22.0


**STEP 3: Descriptive Statistics**

Include:Mean, median, min, max of recovery days

Recovery Time Statistics (in days):

count    28.000000
mean     15.107143
std       5.626256
min       7.000000
25%       9.750000
50%      16.000000
75%      19.250000
max      24.000000
Name: recovery_days, dtype: float64


**STEP 4: Exploratory Data Analysis (EDA) with Visualizations**

1️.Gender Distribution
Bar chart #Figure_1.png
Insight: Female and male counts are comparable

2️.Age Distribution
Histogram #Figure_2.png
Insight: Most patients fall between young adults and elderly

3️. Region-wise Analysis
 #Figure_3.png
Insight: Certain regions like Daegu dominate case numbers

4️.Infection Sources
Horizontal bar chart #Figure_4.png
Insight: Contact with existing patients is the dominant source

5️.Recovery Time Distribution
Histogram #Figure_5.png
Insight: Recovery is clustered around 2–3 weeks


**STEP 5: Correlation Analysis**
You used a correlation matrix heatmap.
Analyzed correlation between:Age,Infection order,Contact number,Recovery days

Observations:
    - Weak correlation overall
    - No strong linear dependency


**STEP 6: Linear Regression**

Objective:Predict recovery time,Identify influencing factors,

Model Used: OLS Linear Regression (statsmodels)
Independent variables: Age,Contact number,Infection order
Dependent variable: Recovery days

   OLS Regression Results                            
==============================================================================
Dep. Variable:          recovery_days   R-squared:                       0.083
Model:                            OLS   Adj. R-squared:                 -0.055
Method:                 Least Squares   F-statistic:                    0.6018
Date:                Wed, 21 Jan 2026   Prob (F-statistic):              0.621
Time:                        22:30:38   Log-Likelihood:                -71.671
No. Observations:                  24   AIC:                             151.3
Df Residuals:                      20   BIC:                             156.1
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
const              14.7069      4.786      3.073      0.006       4.723      24.691
age                 0.0618      0.088      0.700      0.492      -0.122       0.246
contact_number     -0.0109      0.009     -1.243      0.228      -0.029       0.007
infection_order    -0.1304      1.431     -0.091      0.928      -3.115       2.854
==============================================================================
Omnibus:                        2.114   Durbin-Watson:                   2.365
Prob(Omnibus):                  0.348   Jarque-Bera (JB):                1.382
Skew:                          -0.333   Prob(JB):                        0.501
Kurtosis:                       2.031   Cond. No.                         665.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

**Interpretation of Your Regression Output**
R² value is 0.083, indicating weak predictive power
None of the independent variables are statistically significant (p > 0.05)
This suggests recovery time is influenced by other medical or external factors not present in the dataset


**STEP 7: Conclusion**

 -COVID-19 patient data was explored using Python
 -Key trends in age, gender, region, and infection sources were identified
 -Average recovery time was around 15 days
 -Linear regression showed limited predictive ability due to sparse data
 -Highlights the importance of richer clinical datasets
