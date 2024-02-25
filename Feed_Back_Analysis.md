# Feed_Back_Analysis
## 1. Problem Statement
A study aiming to segment participants in the Intel Certification course based on their satisfaction levels endeavors to understand participants' perceptions of the course, covering dimensions like content quality, effectiveness, expertise, and relevance. The primary objective is to categorize participants using their feedback to improve course delivery and content.
## 2. Introduction

Analysis of feedback holds a crucial role in educational settings, influencing instruction quality. Through the scrutiny of participant feedback, educators can identify strengths and areas for improvement. Machine learning provides powerful tools for categorizing feedback into meaningful segments, allowing targeted interventions.
## 2. Dataset

The dataset comprises feedback from students who attended the Intel Unnati Machine Learning Foundation course. The feedback form likely collected information on various aspects of the course, allowing students to express their opinions and experiences.
## 3. Methodology

The study combines exploratory data analysis (EDA) and machine learning, using the K-means clustering algorithm for segmentation. EDA summarizes data characteristics, and K-means clustering groups data based on feature similarity. The Elbow method aids in determining the optimal number of clusters.

## Implementation
### Importing necessary libraries
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```
### Loading data
```python
df_class=pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")
```
```python
df_class.head()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/bbbe88f0-9450-483e-bd85-5d5ba55f377b)
```python
df_class.sample(5).style.set_properties(**{'background-color': 'darkgreen',
                           'color': 'white',
                           'border-color': 'darkblack'})
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/c368a358-163a-44fb-8466-f23a1b01f7b3)
### Data wrankling
```python

df_class.info()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/84546f54-9b5e-4855-ae8e-45ca8ad374ac)

```python
df_class = df_class.drop(['Timestamp','Email ID','Please provide any additional comments, suggestions, or feedback you have regarding the session. Your insights are valuable and will help us enhance the overall learning experience.'],axis=1)
```
```python
df_class.info()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/8158b743-223d-49de-8b6b-1dd11ea3733a)

```python
df_class.columns = ["Name","Branch","Semester","Resourse Person","Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
```
```python
df_class.sample(5)
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/b7b8de3e-4dfd-416d-987c-0a676b5f75c1)

```python
# checking for null
df_class.isnull().sum().sum()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/5b465b5c-a009-4eed-9770-ba4540163c93)

```python
# dimension

df_class.shape
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/127163ad-394d-4acb-8b47-0436c11e3e2d)

## Exploratory Data Analysis
Visualizations show a balanced distribution of data across resource persons, with Mrs. Akshara Sasidharan having the largest share. Content quality is generally high, as indicated by box plots, reflecting overall satisfaction. However, variations in effectiveness, expertise, and overall organization suggest the need for further examination.
```python
## creating a percentage analysis of RP-wise distribution of data
round(df_class["Resourse Person"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/95aa95d0-2786-410d-bd9d-fc63d017d686)

```python
## creating a percentage analysis of RP-wise distribution of data
round(df_class["Resource Person"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/6b5759ea-b0ac-4b09-9ffd-e0d83ee64816)
```python
## creating a percentage analysis of RP-wise distribution of data
round(df_class["Name"].value_counts(normalize=True)*100,2)
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/9f4f2c88-74cd-4922-b3c0-62edc90e6d35)
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/d4d6cbec-9f6a-44d1-8aac-63ce51339f0e)
## Visualization

```python
ax = plt.subplot(1,2,1)
ax = sns.countplot(x='Resourse Person', data=df_class)
#ax.bar_label(ax.containers[0])
plt.title("Faculty-wise distribution of data", fontsize=20,color = 'Brown',pad=20)
ax =plt.subplot(1,2,2)
ax=df_class['Resourse Person'].value_counts().plot.pie(explode=[0.1, 0.1,0.1,0.1],autopct='%1.2f%%',shadow=True);
ax.set_title(label = "Resourse Person", fontsize = 20,color='Brown',pad=20);
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/7718c493-994a-4d85-b5fc-04e4d4e489df)
### Summary of Responses

```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Content Quality'])
plt.show()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/cdd2f9e6-8b94-45f7-ae64-fa721cdf9268)

```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Effeciveness'])
plt.show()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/b040babc-eff1-432b-b610-d509caebfd97)

```python
df_class.info()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/3887f712-3c9b-4daa-8782-74ae4bdc39a6)

```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Expertise'])
plt.show()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/500a4bb1-cb9e-4d22-be55-b47181bba390)

```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Relevance'])
plt.show()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/815dec7c-5509-46c3-9e38-a2c9813720f8)

```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Overall Organization'])
plt.show()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/299819a2-d789-4331-bcc2-851a8e7c0159)

```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Branch'])
plt.show()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/c07cab58-a6da-4a89-9aa2-bf0e7ba269d3)

```python
sns.boxplot(y=df_class['Branch'],x=df_class['Content Quality'])
plt.show()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/e40eeaeb-a802-4bd2-9cc7-d159d8fb418e)
## 5. Machine Learning Model - K-means Clustering

The Elbow method suggests 4 clusters as optimal, but detailed feedback analysis leads to the selection of 3 clusters for in-depth analysis.

## Using K-means Clustering to identify segmentation over student's satisfaction
## Finding the best value of k using elbow method
```python
input_col=["Content Quality","Effeciveness","Expertise","Relevance","Overall Organization"]
X=df_class[input_col].values
```
```python
# Initialize an empty list to store the within-cluster sum of squares
from sklearn.cluster import KMeans
wcss = []

# Try different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)# here inertia calculate sum of square distance in each cluster

```
```python
# Plot the within-cluster sum of squares for different values of k
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/165bf291-b889-4471-8d74-74155f0b68f2)

## Using Gridsearch method
```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto',random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
```
```python
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/2dd49c37-abe3-48fd-b692-c43e4da9f8a0)
## Using Gridsearch method
```python
# Define the parameter grid
from sklearn.model_selection import GridSearchCV

param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto',random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
``
```python
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/a1f94913-7c64-4559-bb88-447dbe6284e0)
## Implementing K-means clustering
```python
# Perform k-means clustering
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/8be6a904-1d37-4b7c-820b-f261a7b2e00e)
## Extracting labels and cluster centers
```python
# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels

```
```python
df_class.head()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/d4f42644-6048-47f2-ad82-aa36b29aa877)

## Visualizing the clustering using first two features
```python
# Visualize the clusters
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:,1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/8b379d21-2cf2-4fb4-ac2b-0f8848def62c)
## Perception on content quality over Clustors
```python
pd.crosstab(columns = df_class['Cluster'], index = df_class['Content Quality'])
```
![image](https://github.com/Bhagya005/Feed_Back_Analysis/assets/106422457/91ba3e45-243e-40e2-bc1d-96c28666312e)
## 6. Results and Conclusion

K-means clustering (k=3) reveals distinct segments within participant feedback. Visualizing clusters using effectiveness and expertise as features highlights clear segmentation, possibly reflecting varying levels of participant satisfaction or expectations.

### Detailed Observations

- **Faculty-wise Distribution of Data:** Even distribution across faculties allows comprehensive feedback analysis.

- **Content Quality Summary:** High ratings indicate that the course material generally meets or exceeds expectations.

- **Effectiveness:** Variability suggests the need for adaptation to diverse learning styles.

- **Expertise:** Instructors' expertise is highly regarded, with minimal outliers.

- **Relevance:** High relevance scores are consistent.

- **Overall Organization:** Slightly more variability indicates room for improvement in time management and instructions' clarity.

- **Branch-wise Content Quality:** Consistent quality, with the ECE branch exhibiting slightly more variability.

### Elbow Method and K-means Clustering

The Elbow method suggests a preference for a smaller number of clusters. K-means clustering effectively visualizes these segments, guiding course providers in tailoring resources and interventions based on different groups' needs.

In summary, EDA and K-means clustering offer insights into Intel Certification course participant feedback. High satisfaction levels are noted, but opportunities exist to enhance effectiveness and overall organization. Future iterations could benefit from targeted improvements informed by segmentation analysis, allowing differentiated strategies for content delivery and organization.

