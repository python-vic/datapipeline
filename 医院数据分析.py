#!/usr/bin/env python
# coding: utf-8

# 我们将从以下几个角度对数据集进行分析，帮助提取有用的信息以支持决策：
# 
# 1. **患者数据分析**：
#    - 分析患者的性别、年龄、地区分布等特征。
#    - 识别主要患者群体，并发现潜在的服务需求。
# 
# 2. **医疗服务分析**：
#    - 调查医院提供的主要服务种类。
#    - 分析不同服务的使用情况及其趋势。
# 
# 3. **运营效率分析**：
#    - 检查医院的床位使用率、住院时长等关键指标。
#    - 识别高效或低效的服务环节。
# 
# 4. **财务数据分析**：
#    - 研究收入来源（自费、保险、政府补助等）。
#    - 识别可能提高收入的策略，或降低成本的机会。
# 
# 5. **患者满意度与结果分析**：
#    - 如果数据中包含患者反馈或医疗结果，分析患者满意度的驱动因素。
#    - 探索治疗效果与服务质量之间的关系。
# 
# 6. **疾病趋势与健康管理**：
#    - 识别常见疾病类型及其季节性趋势。
#    - 提供预防性健康管理的建议。
# 
# 7. **区域比较分析**：
#    - 比较不同地区或医院的服务效率和质量。
#    - 发现地区间的不平等或优化潜力。
# 
# 预览数据
# 
# 
# - **基础信息**：
#   - `case_id`：案例编号
#   - `Hospital_code`：医院代码
#   - `Hospital_type_code`：医院类型代码
#   - `City_Code_Hospital`：医院所在城市代码
#   - `Hospital_region_code`：医院所在地区代码
# 
# - **医院资源**：
#   - `Available Extra Rooms in Hospital`：医院的额外可用病房数量
#   - `Department`：医疗科室
#   - `Ward_Type` 和 `Ward_Facility_Code`：病房类型及设施代码
#   - `Bed Grade`：病床等级
# 
# - **患者信息**：
#   - `patientid`：患者编号
#   - `City_Code_Patient`：患者所在城市代码
#   - `Age`：年龄段
# 
# - **住院信息**：
#   - `Type of Admission`：入院类型（紧急、创伤等）
#   - `Severity of Illness`：病情严重程度
#   - `Visitors with Patient`：随行家属人数
#   - `Admission_Deposit`：入院押金
#   - `Stay`：住院时长（天数范围）
# 
# 接下来，我们将从上述分析方向入手，逐步生成洞察。
# 
# 

# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
healthcare_data=pd.read_csv(r"Public Healthcare Dataset_ Hospital - train_data.csv.csv")
healthcare_data


# In[6]:




# Preparing data for visualizations

# 1. Patient Demographics
age_distribution = healthcare_data['Age'].value_counts().sort_index()

# 2. Medical Services
department_distribution = healthcare_data['Department'].value_counts()

# 3. Operational Efficiency
stay_duration = healthcare_data['Stay'].value_counts().sort_index()

# 4. Financial Data
admission_deposit = healthcare_data['Admission_Deposit']

# 5. Patient Satisfaction or Outcomes
severity_of_illness = healthcare_data['Severity of Illness'].value_counts()

# 6. Disease Trends
admission_type = healthcare_data['Type of Admission'].value_counts()

# 7. Regional Comparisons
region_distribution = healthcare_data['Hospital_region_code'].value_counts()

# Creating charts

# 1. Age Distribution
plt.figure(figsize=(8, 6))
age_distribution.plot(kind='bar')
plt.title('Patient Age Distribution')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

# 2. Department Distribution
plt.figure(figsize=(8, 6))
department_distribution.plot(kind='bar', color='orange')
plt.title('Distribution of Medical Departments')
plt.xlabel('Department')
plt.ylabel('Count')
plt.show()

# 3. Stay Duration
plt.figure(figsize=(8, 6))
stay_duration.plot(kind='bar', color='green')
plt.title('Hospital Stay Duration')
plt.xlabel('Stay Duration (days)')
plt.ylabel('Count')
plt.show()

# 4. Admission Deposit Distribution
plt.figure(figsize=(8, 6))
plt.hist(admission_deposit, bins=20, color='purple', edgecolor='black')
plt.title('Admission Deposit Distribution')
plt.xlabel('Deposit Amount')
plt.ylabel('Frequency')
plt.show()

# 5. Severity of Illness
plt.figure(figsize=(8, 6))
severity_of_illness.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'salmon'])
plt.title('Severity of Illness Distribution')
plt.ylabel('')
plt.show()

# 6. Admission Type
plt.figure(figsize=(8, 6))
admission_type.plot(kind='bar', color='brown')
plt.title('Type of Admission')
plt.xlabel('Admission Type')
plt.ylabel('Count')
plt.show()

# 7. Regional Distribution
plt.figure(figsize=(8, 6))
region_distribution.plot(kind='bar', color='teal')
plt.title('Hospital Region Distribution')
plt.xlabel('Region Code')
plt.ylabel('Count')
plt.show()


# 深入的分析和这些分析如何支持决策的解释：
# 
# ---
# 
# ### 1. **患者数据分析**：深入年龄段、城市分布、和患者流动性
# 
# **分析**：
# - 年龄段细分：找出就诊率最高的年龄组，并与病情严重程度关联，分析特定年龄组的服务需求（如老年群体可能需要更多的慢病管理）。
# - 城市分布：通过 `City_Code_Patient` 的分布找出患者主要来源地。
# - 患者流动性：分析是否有患者跨区域就医的现象。
# 
# **意义**：
# - **优化资源分配**：针对需求量大的群体（如某个年龄段）提供更多资源，如专业科室或针对性的健康管理计划。
# - **区域医疗规划**：确定患者是否因为区域医疗资源不足而跨区就医，从而优化资源分配。
# 
# ---
# 
# ### 2. **医疗服务分析**：深挖各科室的就诊量和效率
# 
# **分析**：
# - 每个科室的患者量占比。
# - 不同科室的住院时长及严重病情患者比例。
# - 特定科室（如放射科）是否在高峰时段或特定季节面临过载。
# 
# **意义**：
# - **提高效率**：发现高需求科室（如放射科、麻醉科）的服务瓶颈，通过新增设备或优化排班缓解压力。
# - **战略投资**：为就诊量持续增长的科室增加投资，例如扩充设施或增加医护人员。
# 
# ---
# 
# ### 3. **运营效率分析**：住院时长与病床使用率
# 
# **分析**：
# - 住院时长的中位数和分布。
# - 病床等级与患者需求的匹配情况。
# - 病房额外房间的使用情况与入住率。
# 
# **意义**：
# - **资源优化**：缩短平均住院时长，提升病床周转率，有助于接收更多患者。
# - **提升服务质量**：分析病房使用效率，优化房间分配和等级匹配，提升患者体验。
# 
# ---
# 
# ### 4. **财务数据分析**：押金与成本回报分析
# 
# **分析**：
# - 押金金额与病情严重程度的关系。
# - 不同支付方式（如保险、自费）的收入分布。
# - 高押金患者是否有较高的住院时长或服务复杂性。
# 
# **意义**：
# - **财务规划**：确保押金政策与患者的实际需求和支付能力相匹配，避免影响患者就医意愿。
# - **提高回报率**：分析支付方式和服务成本，优化价格模型。
# 
# ---
# 
# ### 5. **患者满意度与结果分析**：关注高严重程度患者
# 
# **分析**：
# - 病情严重程度和住院时长的关系。
# - 患者随访数据（如果有），分析满意度。
# - 特殊群体（如极度病重患者）的处理效率和结果。
# 
# **意义**：
# - **提升患者体验**：帮助识别影响患者满意度的关键因素，从而改善服务流程。
# - **结果导向**：以病情严重度为切入点，衡量医院整体治疗效果。
# 
# ---
# 
# ### 6. **疾病趋势与健康管理**：入院类型与季节性分析
# 
# **分析**：
# - 各类型入院的高峰时段（如急诊高峰可能集中在某些节假日）。
# - 分析常见入院原因（如创伤性疾病）和趋势。
# - 患者年龄段与入院类型的关联。
# 
# **意义**：
# - **提前规划**：预测高峰时段的资源需求（如急诊资源）。
# - **预防措施**：根据常见病因设计预防性健康管理策略，减少可控入院情况。
# 
# ---
# 
# ### 7. **区域比较分析**：分析医院在不同地区的服务表现
# 
# **分析**：
# - 不同地区医院的住院时长和病床使用率对比。
# - 不同地区对某些病情的服务能力是否有差异。
# - 区域间患者满意度和医院收治能力的比较。
# 
# **意义**：
# - **平衡资源分配**：识别资源不足的区域，制定针对性的资源优化策略。
# - **区域合作**：通过区域对比，识别潜在的跨区合作机会（如专科资源共享）。
# 
# ---
# 
# ### 总结
# 
# 这些深入分析支持决策的方式包括：
# 1. **优化资源**：精准分配人力、病房和医疗设备，减少资源浪费。
# 2. **提升服务质量**：通过分析患者需求和满意度改进服务体验。
# 3. **财务效益**：通过更科学的收费和成本分析，提升财务表现。
# 4. **健康管理**：为政策制定者提供预防性健康管理的支持数据，减少整体医疗支出。
# 
# 

# In[7]:


# Deeper insights with additional charts for each analytical perspective

# 1. Patient Demographics - Age group and illness severity
plt.figure(figsize=(10, 6))
healthcare_data.groupby(['Age', 'Severity of Illness']).size().unstack().plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Age Group vs Severity of Illness')
plt.xlabel('Age Group')
plt.ylabel('Patient Count')
plt.legend(title='Severity of Illness')
plt.show()

# 2. Medical Services - Department and stay duration analysis
plt.figure(figsize=(10, 6))
healthcare_data.groupby(['Department', 'Stay']).size().unstack().plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Department vs Stay Duration')
plt.xlabel('Department')
plt.ylabel('Patient Count')
plt.legend(title='Stay Duration')
plt.show()

# 3. Operational Efficiency - Bed grade and admission type analysis
plt.figure(figsize=(10, 6))
healthcare_data.groupby(['Bed Grade', 'Type of Admission']).size().unstack().plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Bed Grade vs Type of Admission')
plt.xlabel('Bed Grade')
plt.ylabel('Patient Count')
plt.legend(title='Type of Admission')
plt.show()

# 4. Financial Analysis - Admission deposit by illness severity
plt.figure(figsize=(10, 6))
healthcare_data.boxplot(column='Admission_Deposit', by='Severity of Illness', grid=False)
plt.title('Admission Deposit vs Severity of Illness')
plt.suptitle('')  # Remove default suptitle
plt.xlabel('Severity of Illness')
plt.ylabel('Admission Deposit')
plt.show()

# 5. Patient Satisfaction - Stay duration and visitor count analysis
plt.figure(figsize=(10, 6))
healthcare_data.groupby(['Stay', 'Visitors with Patient']).size().unstack().plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stay Duration vs Visitors with Patient')
plt.xlabel('Stay Duration')
plt.ylabel('Patient Count')
plt.legend(title='Visitor Count')
plt.show()

# 6. Disease Trends - Admission type by severity
plt.figure(figsize=(10, 6))
healthcare_data.groupby(['Type of Admission', 'Severity of Illness']).size().unstack().plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Admission Type vs Severity of Illness')
plt.xlabel('Type of Admission')
plt.ylabel('Patient Count')
plt.legend(title='Severity of Illness')
plt.show()

# 7. Regional Comparisons - Region code and available rooms
plt.figure(figsize=(10, 6))
healthcare_data.groupby(['Hospital_region_code', 'Available Extra Rooms in Hospital']).size().unstack().plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Hospital Region vs Available Rooms')
plt.xlabel('Hospital Region')
plt.ylabel('Room Count')
plt.legend(title='Available Rooms')
plt.show()


# In[11]:


# Analyze factors influencing admission deposits
import seaborn as sns

# Check correlation between numerical variables and Admission_Deposit
correlation_matrix = healthcare_data[['Available Extra Rooms in Hospital', 'Admission_Deposit', 'Visitors with Patient']].corr()

# Plot the heatmap for correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation between Admission Deposit and Numerical Features')
plt.tight_layout()
plt.show()

# Analyze Admission_Deposit distribution across categorical variables
plt.figure(figsize=(10, 6))
sns.boxplot(x='Severity of Illness', y='Admission_Deposit', data=healthcare_data)
plt.title('Admission Deposit by Severity of Illness')
plt.xlabel('Severity of Illness')
plt.ylabel('Admission Deposit')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Age', y='Admission_Deposit', data=healthcare_data)
plt.title('Admission Deposit by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Admission Deposit')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[14]:


# Fixing and reattempting clustering analysis without external tool dependencies

# Importing necessary libraries again to ensure a clean environment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

data = healthcare_data

# Selecting numerical features for clustering
numerical_features = ['Available Extra Rooms in Hospital', 'Visitors with Patient', 'Admission_Deposit']

# Standardizing the features for better clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[numerical_features])

# Applying KMeans clustering with 3 clusters for demonstration
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Adding cluster labels to the original dataset
data['Cluster'] = clusters

# Visualizing the clusters with Admission_Deposit and Visitors with Patient
plt.figure(figsize=(10, 6))
plt.scatter(data['Admission_Deposit'], data['Visitors with Patient'], c=data['Cluster'], cmap='viridis', alpha=0.6)
plt.title('Clusters of Patients Based on Features')
plt.xlabel('Admission Deposit')
plt.ylabel('Visitors with Patient')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.show()

# Summarizing cluster characteristics
cluster_summary = data.groupby('Cluster')[numerical_features].mean()
cluster_summary


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

# Prepare the data for machine learning
# Selecting features and target for regression and classification tasks
regression_features = ['Stay_Encoded', 'Severity_Encoded', 'Visitors with Patient']
regression_target = 'Admission_Deposit'

classification_features = ['Admission_Deposit', 'Severity_Encoded', 'Visitors with Patient']
classification_target = 'Stay_Encoded'

# Clean the dataset for missing values
regression_data = healthcare_data.dropna(subset=regression_features + [regression_target])
classification_data = healthcare_data.dropna(subset=classification_features + [classification_target])

# Splitting data into train and test sets
X_reg = regression_data[regression_features]
y_reg = regression_data[regression_target]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

X_clf = classification_data[classification_features]
y_clf = classification_data[classification_target]
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# 1. Regression Model - Predicting Admission Deposit
reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)
reg_mse = mean_squared_error(y_test_reg, y_pred_reg)

# 2. Classification Model - Predicting Stay Duration
clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train_clf, y_train_clf)
y_pred_clf = clf_model.predict(X_test_clf)
clf_accuracy = accuracy_score(y_test_clf, y_pred_clf)
clf_report = classification_report(y_test_clf, y_pred_clf)

# Displaying results
model_results = {
    "Regression Model - Mean Squared Error": reg_mse,
    "Classification Model - Accuracy": clf_accuracy,
    "Classification Model - Report": clf_report
}


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns



# Setting up figure dimensions
plt.figure(figsize=(15, 8))

### 1. Department Utilization ###
department_counts = data['Department'].value_counts()

plt.figure(figsize=(10, 6))
department_counts.plot(kind='bar', title='Patient Distribution Across Departments')
plt.xlabel('Department')
plt.ylabel('Number of Patients')
plt.tight_layout()
plt.show()

# Note: Highlights which departments handle the most patients and may need resource scaling.

### 2. Regional Trends ###
region_counts = data['Hospital_region_code'].value_counts()

plt.figure(figsize=(10, 6))
region_counts.plot(kind='bar', title='Patient Distribution Across Hospital Regions')
plt.xlabel('Hospital Region Code')
plt.ylabel('Number of Patients')
plt.tight_layout()
plt.show()

# Note: Regional differences can guide targeted infrastructure improvements.

### 3. Severity and Resources ###
plt.figure(figsize=(10, 6))
sns.boxplot(x='Severity of Illness', y='Available Extra Rooms in Hospital', data=data)
plt.title('Severity of Illness vs. Available Extra Rooms in Hospital')
plt.xlabel('Severity of Illness')
plt.ylabel('Available Extra Rooms')
plt.tight_layout()
plt.show()

# Note: Shows how resource allocation varies with illness severity.

### 4. Bed Grades and Stay ###
plt.figure(figsize=(10, 6))
sns.boxplot(x='Bed Grade', y='Admission_Deposit', data=data)
plt.title('Bed Grade vs. Admission Deposit')
plt.xlabel('Bed Grade')
plt.ylabel('Admission Deposit')
plt.tight_layout()
plt.show()

# Note: Helps assess if higher bed grades correlate with higher admission deposits.

### 5. Hospital Types ###
hospital_type_counts = data['Hospital_type_code'].value_counts()

plt.figure(figsize=(10, 6))
hospital_type_counts.plot(kind='bar', title='Patient Distribution by Hospital Type Code')
plt.xlabel('Hospital Type Code')
plt.ylabel('Number of Patients')
plt.tight_layout()
plt.show()

# Note: Provides an overview of hospital type usage trends.

# Notes will be displayed below each chart automatically. Let me know if any specific aspect needs more attention!


# In[16]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from scipy.stats import chi2_contingency, spearmanr



# STEP 1: Data Cleaning and Preprocessing
# Handling missing values
data.fillna(data.median(numeric_only=True), inplace=True)

# Encoding categorical variables
label_encoders = {}
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Scaling numerical features
scaler = StandardScaler()
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
data_scaled = data.copy()
data_scaled[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# STEP 2: Advanced Statistical Analysis
# Correlation matrix
correlation_matrix = data_scaled.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Chi-square test for categorical variable independence
chi2_results = {}
for col in categorical_cols:
    if col != 'Stay':
        contingency_table = pd.crosstab(data['Stay'], data[col])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi2_results[col] = p

chi2_results = {k: v for k, v in sorted(chi2_results.items(), key=lambda item: item[1])}

# STEP 3: Machine Learning Modeling
# Predicting "Stay" category using RandomForestClassifier
X = data_scaled.drop(['Stay'], axis=1)
y = data_scaled['Stay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_class = rf_classifier.predict(X_test)

classification_rep = classification_report(y_test, y_pred_class)

# Regression Model for Admission Deposit
rf_regressor = RandomForestRegressor(random_state=42)
X_reg = data_scaled.drop(['Admission_Deposit'], axis=1)
y_reg = data_scaled['Admission_Deposit']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_regressor.predict(X_test_reg)

regression_metrics = {
    "MSE": mean_squared_error(y_test_reg, y_pred_reg),
    "R2": r2_score(y_test_reg, y_pred_reg)
}

# STEP 4: Clustering for Group Analysis
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
data['Cluster'] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Admission_Deposit'], y=data['Visitors with Patient'], hue=data['Cluster'], palette='viridis')
plt.title('Clustering Patients Based on Features')
plt.xlabel('Admission Deposit')
plt.ylabel('Visitors with Patient')
plt.tight_layout()
plt.show()

# Summary Outputs
import ace_tools as tools; tools.display_dataframe_to_user(name="Chi-Square Test P-Values for Independence", dataframe=pd.DataFrame(chi2_results.items(), columns=["Feature", "P-Value"]))
classification_rep, regression_metrics


# In[ ]:




