import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('iris')

print("🔹 First 5 rows of the dataset:")
print(df.head())


print("\n🔹 Data Types:")
print(df.dtypes)

print("\n🔹 Missing Values:")
print(df.isnull().sum())


print("\n🔹 Descriptive Statistics:")
print(df.describe())

print("\n🔹 Mean Sepal Length by Species:")
print(df.groupby('species')['sepal_length'].mean())


plt.figure(figsize=(8, 4))
plt.plot(df.index, df['sepal_length'], label='Sepal Length', color='green')
plt.title('Line Chart of Sepal Length')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
df.groupby('species')['petal_length'].mean().plot(kind='bar', color='skyblue')
plt.title('Average Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.show()

plt.figure(figsize=(6, 4))
plt.hist(df['sepal_width'], bins=10, color='orange', edgecolor='black')
plt.title('Histogram of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.show()

print("\n🔹 Observations:")
print("- Setosa has shorter petals compared to others.")
print("- Sepal length and petal length show a strong positive correlation.")
print("- There are no missing values in the dataset.")