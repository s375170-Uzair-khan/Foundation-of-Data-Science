import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load datasets
demographics = pd.read_csv('dataset1.csv')
screen_time = pd.read_csv('dataset2.csv')
wellbeing = pd.read_csv('dataset3.csv')

# Merge datasets on the ID
merged_data = demographics.merge(screen_time, on='ID').merge(wellbeing, on='ID')

# Group by gender to analyze screen time differences
gender_screen_time = merged_data.groupby('gender')[['C_wk', 'C_we', 'G_wk', 'G_we', 'S_wk', 'S_we', 'T_wk', 'T_we']].mean()

# Plotting with custom colors
colors = ['#1f77b4', '#ff7f0e']  # Example colors for different genders
gender_screen_time.plot(kind='bar', figsize=(10, 6), color=colors, title="Average Screen Time by Gender")
plt.ylabel("Average Hours")
plt.xticks(rotation=0)
plt.show()

# Plot well-being scores
wellbeing_columns = ['Optm', 'Usef', 'Relx', 'Conf', 'Engs', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
merged_data[wellbeing_columns].plot(kind='box', figsize=(10, 6), title="Distribution of Well-being Scores")
plt.show()

# Correlation analysis between screen time and well-being indicators
correlation_matrix = merged_data[['C_wk', 'S_wk', 'G_wk', 'T_wk', 'Optm', 'Relx', 'Conf']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')  # Changed color map to 'viridis'
plt.title("Correlation Between Screen Time and Well-being Indicators")
plt.show()

# Perform T-test to compare well-being scores by gender
male_scores = merged_data[merged_data['gender'] == 1][['Optm', 'Relx', 'Conf']]
female_scores = merged_data[merged_data['gender'] == 0][['Optm', 'Relx', 'Conf']]

t_test_result = stats.ttest_ind(male_scores, female_scores)
print(t_test_result)

