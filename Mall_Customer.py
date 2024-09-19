# Import Python Libraries :
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Data
data = pd.read_csv('E:\\Mall Customer\\Mall_Customers.csv')

# Explore The Data :
print(data.head(5)) # First 5
print("=====================================")
print(data.tail(5)) # Last 5
print("=====================================")
print(data.sample()) # One Random Data

# Data Description (Columns & Rows) :
print("=====================================")
print(data.shape)
print("=====================================")
print(data.info())
print("=====================================")
print(data.describe())

# Transform Object Data :
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender']) 
print("=====================================")
print(data) # Male = 1 & Female = 0

# Scale the data
standard_scaler = StandardScaler()
scaled_data = standard_scaler.fit_transform(data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Determine the optimal number of clusters using the elbow method
clusters_num = []
inertia_values = []

for i in range(1, 12):
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(scaled_data)
    clusters_num.append(i)
    inertia_values.append(model.inertia_)

# Plotting the elbow graph
plt.figure(figsize=(8, 4))
plt.plot(clusters_num, inertia_values, marker='o')
plt.title('Find The Best Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Train the model with the chosen number of clusters
model = KMeans(n_clusters=4, random_state=42)
model.fit(scaled_data)
data['Best_Cluster'] = model.predict(scaled_data)

# Print data with cluster assignments
print("=====================================")    
print(data)
print("=====================================")  

# Plotting the clusters using the scaled data
plt.figure(figsize=(10, 6))
for cluster in range(4):
    clustered_data = scaled_data[data['Best_Cluster'] == cluster]
    plt.scatter(clustered_data[:, 1], clustered_data[:, 2], label=f'Cluster {cluster + 1}')  # Using scaled columns for plotting
    
plt.title('Final Clusters (Scaled Data)')
plt.xlabel('Scaled Annual Income')
plt.ylabel('Scaled Spending Score')
plt.legend()
plt.show()

# Data Analysis: Visualizing the number of customers in each cluster
sns.countplot(x=data['Best_Cluster'])
plt.title('Number of Customers in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()

final_plot = data['Best_Cluster'].value_counts()
print(final_plot)
print("=====================================") 
plt.pie(final_plot,autopct='%0.2f%%')
plt.show()

sns.boxplot(x=data['Best_Cluster'], y=data['Spending Score (1-100)'])
plt.title('Boxplot of Spending Score by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Spending Score (1-100)')
plt.show()

sns.histplot(data.Age)
plt.show()

# Convert Gender back to a categorical type for plotting
data['Gender'] = data['Gender'].replace({0: 'Female', 1: 'Male'})

# Plotting with hue as categorical
sns.countplot(x=data['Best_Cluster'], hue=data['Gender'])
plt.title('Number of Customers in Each Cluster by Gender')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()

sns.countplot(data = data , x ='Gender')
plt.show()
