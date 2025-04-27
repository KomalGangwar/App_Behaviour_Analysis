import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from dateutil import parser

# Load dataset
dataset = pd.read_csv('appdata10.csv')

# Data Cleaning
dataset['hour'] = dataset.hour.str.slice(1, 3).astype(int)

# Copy dataset without unwanted columns
dataset2 = dataset.copy().drop(columns=['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])

# Plotting Histograms for Numerical Columns
plt.figure(figsize=(10, 6))
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.suptitle('Histograms of Numerical Columns', fontsize=14)

for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i - 1], fontsize=8)
    vals = np.size(dataset2.iloc[:, i - 1].unique())
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
plt.show()  # This will display the histograms
plt.clf()  # Clear the current figure

# Correlation with Response
dataset2.corrwith(dataset.enrolled).plot.bar(
    figsize=(20, 10),
    title='Correlation with Response Variable',
    fontsize=15,
    rot=45,
    grid=True,
    color=['#32a852', '#005c1f']
)
plt.tight_layout()
plt.show()  # Display correlation bar plot
plt.clf()

# Correlation Matrix Heatmap
sn.set(style="white", font_scale=2)
corr = dataset2.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize=40)
cmap = sn.diverging_palette(220, 10, as_cmap=True)

sn.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=.3,
    center=0,
    square=True,
    linewidths=.5,
    cbar_kws={"shrink": .5}
)

plt.show()  # Display heatmap
plt.clf()

# Feature Engineering: Difference Between Dates
dataset["first_open"] = [parser.parse(row_data) for row_data in dataset["first_open"]]
dataset["enrolled_date"] = [parser.parse(row_data) if isinstance(row_data, str) else row_data for row_data in dataset["enrolled_date"]]
dataset["difference"] = (dataset.enrolled_date - dataset.first_open).dt.total_seconds() / 3600

# Distribution of Time-Since Enrolled
plt.figure(figsize=(10, 6))
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

plt.hist(dataset["difference"].dropna(), color='#3F5D7D')
plt.title('Distribution of Time-Since Enrolled', fontsize=12)
plt.show()  # Display histogram
plt.clf()

# After cleaning 'enrolled' based on the 'difference' column
dataset.loc[dataset.difference > 48, 'enrolled'] = 0
dataset = dataset.drop(columns=['difference', 'enrolled_date', 'first_open'])

# Formatting screen_list Field and creating additional columns (using the top_screens.csv)
top_screens = pd.read_csv('top_screens.csv').top_screens.values
dataset['screen_list'] = dataset.screen_list.astype(str) + ','

for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset['screen_list'] = dataset.screen_list.str.replace(sc + ",", "")

dataset['Other'] = dataset.screen_list.str.count(",")
dataset = dataset.drop(columns=['screen_list'])

# Funnels (Eliminating co-related screens and combining them)
savings_screens = ["Saving1", "Saving2", "Saving2Amount", "Saving4", "Saving5", "Saving6", "Saving7", "Saving8", "Saving9", "Saving10"]
dataset["SavingsCount"] = dataset[savings_screens].sum(axis=1)
dataset = dataset.drop(columns=savings_screens)

cm_screens = ["Credit1", "Credit2", "Credit3", "Credit3Container", "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)

cc_screens = ["CC1", "CC1Category", "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ["Loan", "Loan2", "Loan3", "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)

# Saving the cleaned data
dataset.to_csv('new_appdata10.csv', index=False)

print("EDA and visualizations complete.")