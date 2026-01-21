import pandas as pd
df=pd.read_csv("patient.csv")
print(df.head())

print("\nDatasetInfo:\n")
print(df.info())
print("\nMissing Values:\n")
print(df.isnull().sum())

# Convert date columns to datetime
df['confirmed_date'] = pd.to_datetime(df['confirmed_date'], errors='coerce')
df['released_date'] = pd.to_datetime(df['released_date'], errors='coerce')
df['deceased_date'] = pd.to_datetime(df['deceased_date'], errors='coerce')

print(df[['confirmed_date', 'released_date', 'deceased_date']].head())

# Calculate recovery time in days
df['recovery_days'] = (df['released_date'] - df['confirmed_date']).dt.days
#Evaluate time taken for recovery from confirmation to release
print(df[['confirmed_date', 'released_date', 'recovery_days']].head())

print("\nRecovery Days Sample:\n")
print(df[['state','confirmed_date','released_date','recovery_days']].head(15))

#work done till now: 1.data loaded correctly 2.Missing values analyzed 3.Dates converted properly 4.Recovery time calculated correctly

#computing the basics of recovery time
# Recovery statistics (only for released patients)
recovered_df = df[df['state'] == 'released']

print("\nRecovery Time Statistics (in days):\n")
print(recovered_df['recovery_days'].describe())

#for visuals
import matplotlib.pyplot as plt

#gender distribution
genger_count=df['sex'].dropna().value_counts()
plt.figure()
genger_count.plot(kind='bar', color=['blue', 'pink'])
plt.title("Gender Distribution of Infected Patients")
plt.xlabel("Gender")
plt.ylabel("Number of Patients")
plt.show()
#“Gender analysis is based only on available records, 
# as a significant portion of data has missing gender information.”

#age distribution
# Calculate age (assuming year 2020)
df['age'] = 2020 - df['birth_year']
plt.figure()
plt.hist(df['age'].dropna(), bins=20)
plt.title("Age Distribution of Infected Patients")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()
#“The age distribution indicates that infections are spread 
# across a wide adult age range, with higher concentration among middle-aged individuals.”

#Regional case concentration
region_counts = df['region'].dropna().value_counts().head(10)
plt.figure()
region_counts.plot(kind='bar',color='purple')
plt.title("Top 10 Regions by Number of Cases")
plt.xlabel("Region")
plt.ylabel("Number of Patients")
plt.xticks(rotation=45)
plt.show()

#Infection sources
infection_counts = df['infection_reason'].dropna().value_counts().head(10)
plt.figure()
infection_counts.plot(kind='barh', color='green')
plt.title("Top Infection Sources")
plt.xlabel("Number of Patients")
plt.ylabel("Infection Reason")
plt.show()

#recovery Timeline
# Recovery timeline visualization (only released patients)
recovery_data = recovered_df['recovery_days'].dropna()

plt.figure()
plt.hist(recovery_data, bins=15)
plt.title("Recovery Time Distribution")
plt.xlabel("Days to Recovery")
plt.ylabel("Number of Patients")
plt.show()

#corerelation analysis
import seaborn as sns
import numpy as np

corr_df = df[['age', 'infection_order', 'contact_number', 'recovery_days']]
corr_matrix = corr_df.corr()
plt.figure()
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show() #a heatmap is seen showing numbersbetween -1 and 1 

import statsmodels.api as sm

# Prepare data for regression
reg_df = df[['age', 'contact_number', 'infection_order', 'recovery_days']].dropna()

X = reg_df[['age', 'contact_number', 'infection_order']]
y = reg_df['recovery_days']

# Add constant for intercept
X = sm.add_constant(X)

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Print detailed summary
print(model.summary())
