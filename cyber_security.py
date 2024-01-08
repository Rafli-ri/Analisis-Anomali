import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv('cybersecurity_attacks.csv')

# Sidebar
# st.sidebar.title('Cybersecurity Analysts')
# show_data = st.sidebar.checkbox('Show Data', value=False)

# Main content
st.title('Cybersecurity Analysts')

if show_data:
    st.subheader('Raw Data')
    st.write(df)

# Data preprocessing
anomali = df[['Attack Type', 'Anomaly Scores']]
df = pd.DataFrame(anomali)

# Bar plot
st.subheader('Bar Plot Based on Attack Type')
plt.figure(figsize=(8, 6))
sns.barplot(x='Attack Type', y='Anomaly Scores', data=df)
plt.title('Bar Plot Based on Attack Type')
plt.xlabel('Attack Type')
plt.ylabel('Anomaly Scores')
st.pyplot()

# Distribution of Attack Types
st.subheader('Distribution of Attack Types')
attack_counts = df['Attack Type'].value_counts()
colors = sns.color_palette('inferno', len(attack_counts))

plt.figure(figsize=(12, 6))
sns.barplot(x=attack_counts.index, y=attack_counts, palette=colors)

for i, count in enumerate(attack_counts):
    plt.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel('Attack Type', fontsize=14, fontweight='bold')
plt.ylabel('Counts', fontsize=14, fontweight='bold')
plt.title('Distribution of Attack Types', fontsize=16)
plt.xticks(rotation=45, ha='right')

st.pyplot()

# Box plot for 'Anomaly Scores' by 'Attack Type'
st.subheader('Box Plot for Anomaly Scores by Attack Type')
plt.figure(figsize=(12, 6))
sns.boxplot(x='Attack Type', y='Anomaly Scores', data=df)
plt.title('Anomaly Scores by Attack Type')
plt.xticks(rotation=45, ha='right')
st.pyplot()

# Correlation Heatmap
st.subheader('Correlation Heatmap: Anomaly Scores vs Attack Type')
correlation_matrix = df[['Anomaly Scores', 'Attack Type']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap: Anomaly Scores vs Attack Type')
st.pyplot()

# Machine Learning
st.subheader('Machine Learning')

# Label encoding for 'Attack Type'
label_encoder = LabelEncoder()
df['Attack Type Encoded'] = label_encoder.fit_transform(df['Attack Type'])

X = df[['Anomaly Scores', 'Attack Type Encoded']]

# Initialize and train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

df['Anomaly Predictions'] = model.predict(X)

# Classification Report
st.subheader('Classification Report')
classification_report_text = classification_report(
    df['Anomaly Predictions'], [-1 if i == 1 else 0 for i in df['Anomaly Predictions']]
)
st.text(classification_report_text)

# Visualization of anomaly class separation
st.subheader('Anomaly Detection - Machine Learning')
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Anomaly Scores', y='Attack Type Encoded', hue='Anomaly Predictions', data=df)
plt.title('Anomaly Detection - Machine Learning')
plt.xlabel('Anomaly Scores')
plt.ylabel('Attack Type Encoded')
st.pyplot()