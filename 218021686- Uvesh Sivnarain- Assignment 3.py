#!/usr/bin/env python
# coding: utf-8

# <h1 style="background: linear-gradient(to right,  #b2dfdb, #e0f2f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;">
#     Assignment 3 <br>
#     Author: Uvesh Sivnarain <br>
#     Source: Excel File - Dataset.xlsx<br>
#     Data:   Coricraft Customer Profile<br>
#     Github Link: <br>
# </h1>
# 
# 

# In[89]:


get_ipython().system('pip install pandas')


# In[80]:


import pandas as pd


# In[200]:


df = pd.read_excel('Dataset.xlsx')



# In[196]:


df


# <h1 style="background: linear-gradient(to right,  #b2dfdb, #e0f2f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;">
#     Cleaning the Data
# </h1>

# <h2 style="text-align:center;background-color:#a8dda8">1. Removing Empty Values</h2>

# In[186]:


# Read Excel file
df = pd.read_excel('Dataset.xlsx')

# Removed rows with any missing values
cleaned_df = df.dropna()

# Displayed the first few rows
print(cleaned_df.head())


# <h2 style="text-align:center;background-color:#a8dda8">2. Resetting the Index</h2>

# In[192]:


# Reset the index column
df_no_index = df.reset_index(drop=True)

# Displayed the first few rows of the DataFrame 
print(df_no_index.head())


# <h1 style="background: linear-gradient(to right,  #b2dfdb, #e0f2f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;">
#     Measure of Location
# </h1>

# <h2 style="text-align:center;background-color:#a8dda8">1. Mean</h2>
# 

# In[224]:


# Calculated the mean of the Age and Work Experience columns
mean_age = round(df['Age'].mean(), 2)
mean_experience = round(df['Work Experience'].mean(), 2)
mean_income = round(df['Annual Income ($)'].mean(), 2)


print("Mean Age =", mean_age)
print("Mean Work Experience =", mean_experience)
print("Mean Annual Income ($):", mean_income)
print("The average age of the customers at Coricraft is 39.60 years old while there average work experience is 14.42 years old and       average income is $50122 which  indicates Coricraft customers are mainly middle aged and very experienced")


# <h2 style="text-align:center;background-color:#a8dda8">2. Mode</h2>

# In[205]:


import pandas as pd
from statistics import mode


age_mode = mode(df['Age'])
print("Mode of Age:", age_mode)


exp_mode = mode(df['Work Experience'])
print("Mode of Work Experience:", exp_mode)
print("The age group of 32 is the most frequent and the most common work experience among customers is 15 years which shows that Coricraft market is mainly middle aged adults who are working")


# <h2 style="text-align:center;background-color:#a8dda8">3. Median</h2>

# In[211]:


# Calculated median for Age and Work Experience columns
median_age = df['Age'].median()
median_experience = df['Work Experience'].median()


print("Median age:", median_age)
print("Median work experience:", median_experience)
print("The median(middle values) for the age and work experience of the customers is relateively high which indicate that the         customers is skewed towards slightly older and experienced customers. This suggests that the target audience for Coricraft        might be individuals who are in their middle-age and have a significant amount of work experience. ")


# <h2 style="text-align:center;background-color:#a8dda8">4. Weighted Mean</h2>

# In[215]:


# Calculated weighted average of Age column using Annual Income ($) 
weighted_avg_age = (df['Age'] * df['Annual Income ($)']).sum() / df['Annual Income ($)'].sum()

print("Weighted average age:", round(weighted_avg_age, 2))
print("The weighted average age of 40.76 years suggests that the average age of the customers is influenced more by customers with     higher annual incomes, as they have a greater weight in the calculation. This may indicate that the store's customer base      consists of a mix of age groups with varying incomes.")


# <h2 style="text-align:center;background-color:#a8dda8">5. Geometric Mean</h2>

# In[222]:


geo_mean = np.exp(np.mean(np.log(df['Annual Income ($)'])))
print("Geometric mean of Annual Income ($):", round(geo_mean, 2))
print("The geometric mean of annual income ($36,476.6) suggests that the distribution of incomes among the customers is not           symmetrical, and is skewed towards lower incomes as it is below the average 50122$ as seen above.")


#  <h2 style="text-align:center;background-color:#a8dda8">6. Percentiles (25%, 50%, 75%)</h2>

# In[227]:


# Calculated percentiles for Age and Work Experience columns
age_pct = np.percentile(df['Age'], [25, 50, 75])
exp_pct = np.percentile(df['Work Experience'], [25, 50, 75])


data = {'Age': age_pct, '      Work Experience': exp_pct}
df_pct = pd.DataFrame(data, index=['25%', '50%', '75%'])
df_pct = df_pct.applymap('{:.2f}'.format)


table = df_pct.style.set_caption("Percentiles for Age and Work Experience")
table
print("The percentiles give  an idea of the distribution of Age and Work Experience values among Coricraft customers with 25% being     above the age of 25 years showing that the majority of their customers are older adults and not teens or youngsters")


# <h2 style="text-align:center;background-color:#a8dda8">7. Quartiles(1st, 2nd, 3rd)</h2>

# In[228]:


# Calculated quartiles for Age and Work Experience
q1_age = df['Age'].quantile(0.25)
q2_age = df['Age'].quantile(0.5)
q3_age = df['Age'].quantile(0.75)

q1_exp = df['Work Experience'].quantile(0.25)
q2_exp = df['Work Experience'].quantile(0.5)
q3_exp = df['Work Experience'].quantile(0.75)


data = {'Age': [q1_age, q2_age, q3_age], 'Work Experience': [q1_exp, q2_exp, q3_exp]}
index = ['1st Quartile', '2nd Quartile (Median)', '3rd Quartile']
table = pd.DataFrame(data=data, index=index).applymap('{:.2f}'.format)
table.style.set_caption("Quartiles for Age and Work Experience")


# <h1 style="background: linear-gradient(to right,  #b2dfdb, #e0f2f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;">
#     Measure of Variability
# </h1>

# <h2 style="text-align:center;background-color:#a8dda8">1. Range</h2>

# In[230]:


import pandas as pd
import seaborn as sns


# Calculated the range for the Age column
age_range = df['Age'].max() - df['Age'].min()
print("Range for Age column:", age_range)

# Created a boxplot for to showcase how the 62 year range is spread out
sns.boxplot(data=df[['Age']], orient='h')
print("The range of 62 shows that Coricraft customers are diverse in terms of age")


# <h2 style="text-align:center;background-color:#a8dda8">2. Interquartile range</h2>

# In[232]:



import pandas as pd
import seaborn as sns

# Calculated the Interquartile Range for Age
q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)
iqr = q3 - q1

print("Interquartile range of Age:", iqr)
print("This shows that tThe age of customers is relatively spread out, with the middle half of the customers falling within a range of 20 years.")


# <h2 style="text-align:center;background-color:#a8dda8">3. Variance</h2>

# In[234]:


# Calculated the variance for age and work experience
var_age = df['Age'].var()
var_exp = df['Work Experience'].var()


table = [['Column', '       Age', '        Work Experience'],
         ['Variance', '{:12.2f}'.format(var_age), '{:20.2f}'.format(var_exp)]]


for row in table:
    print('{:<15} {:<15} {:<15}'.format(*row))
print("The variance value of age is 216.31 years and work experience is 133.75 years , which suggests that the data points for age are more spread out than those for work experience, indicating greater variability in the ages of the customers compared to their  work experience.")


# <h2 style="text-align:center;background-color:#a8dda8">4. Standard deviation</h2>

# In[242]:


# Calculated the standard deviation for Age and Work Experience columns
std_age = df['Age'].std()
std_exp = df['Work Experience'].std()


print("Standard deviation for Age: {:.2f}".format(std_age))
print("Standard deviation for Work Experience: {:.2f}".format(std_exp))
print("The Age column has a standard deviation of 14.71 which indicates that the ages are spread out from the mean age value of         approximately 38 years. Similarly, the standard deviation of the Work Experience column is 11.57, indicating that the work     experience values are dispersed from the mean work experience value of approximately 14 years.This information can help in understanding the age and work experience distribution of the customers and can help in making decisions related to marketing  or customer service.")


# <h2 style="text-align:center;background-color:#a8dda8">5. Coefficient variation</h2>

# In[246]:


cv_age = round((df['Age'].std() / df['Age'].mean()) * 100, 2)
print("Coefficient of variation for Age:", cv_age)

cv_exp = round((df['Work Experience'].std() / df['Work Experience'].mean()) * 100, 2)
print("Coefficient of variation for Work Experience:", cv_exp)
print("The coefficient of variation for Work Experience is higher than that for Age, which suggests that there is a greater            variability in the Work Experience values compared to the Age values for Coricraft customers ")


# <h1 style="background: linear-gradient(to right,  #b2dfdb, #e0f2f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;">
#     Measure of Distribution Shape
# </h1>

# <h2 style="text-align:center;background-color:#a8dda8">1. Distribution Shape</h2>

# In[250]:


import matplotlib.pyplot as plt

# create a histogram of the "Age" column
plt.hist(df['Age'], bins=10)

# add labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')

# display the plot
plt.show()


import matplotlib.pyplot as plt

# create a histogram for work experience
plt.hist(df['Work Experience'], bins=20)
plt.title('Histogram of Work Experience')
plt.xlabel('Work Experience')
plt.ylabel('Frequency')
plt.show()
print("The distribution of Age appears to be somewhat normal, with the majority of customers falling in the age range of 30-50. The   distribution of Work Experience is heavily skewed to the right, indicating that most customers have relatively low work         experience.")


# <h2 style="text-align:center;background-color:#a8dda8">2. z-scores</h2>

# In[251]:


import numpy as np

# calculate mean and standard deviation of work experience
we_mean = np.mean(df['Work Experience'])
we_std = np.std(df['Work Experience'])

# calculated z-score for work experience values of 25 and 40
z_score_25 = (25 - we_mean) / we_std
z_score_40 = (40 - we_mean) / we_std

print('Z-score for Work Experience = 25:', z_score_25)
print('Z-score for Work Experience = 40:', z_score_40)
print("The z-score for Work Experience = 25 is 0.92, which means that 25 years is about 0.92 standard deviations above the mean. The z-score for Work Experience = 40 is 2.22, which means that 40 is about 2.22 standard deviations above the mean.")


# <h2 style="text-align:center;background-color:#a8dda8">3. Chebyslev's Theorem</h2>

# In[254]:


import numpy as np

#calculated mean and standard deviation of work experience
we_mean = np.mean(df['Work Experience'])
we_std = np.std(df['Work Experience'])

#calculated Chebyshev's Theorem for k=3
k = 3
prop_within_k_std = 1 - 1/(k**2)


print('Proportion of data within', k, 'standard deviations of the mean for Work Experience column:', prop_within_k_std)
print("Chrbyshev's Theorm for work experiencve where the proportion of data within", k, "standard deviations of the mean:", prop_within_k_std)


# <h2 style="text-align:center;background-color:#a8dda8">4. Empirical Rule</h2>

# In[257]:


import numpy as np

# Calculated mean and standard deviation for Work Experience column
we_mean = np.mean(df['Work Experience'])
we_std = np.std(df['Work Experience'])

# Applied Empirical Rule for Work Experience column
we_range_1std = (round(we_mean - we_std, 2), round(we_mean + we_std, 2))
we_range_2std = (round(we_mean - 2 * we_std, 2), round(we_mean + 2 * we_std, 2))
we_range_3std = (round(we_mean - 3 * we_std, 2), round(we_mean + 3 * we_std, 2))

print("Empirical Rule for:")
print("Work Experience:")
print("Range within 1 standard deviation of the mean: ", we_range_1std)
print("Range within 2 standard deviations of the mean: ", we_range_2std)
print("Range within 3 standard deviations of the mean: ", we_range_3std)
print()

# Calculated mean and standard deviation for Age column
age_mean = np.mean(df['Age'])
age_std = np.std(df['Age'])

# Applied Empirical Rule for Age column
age_range_1std = (round(age_mean - age_std, 2), round(age_mean + age_std, 2))
age_range_2std = (round(age_mean - 2 * age_std, 2), round(age_mean + 2 * age_std, 2))
age_range_3std = (round(age_mean - 3 * age_std, 2), round(age_mean + 3 * age_std, 2))

print("Age:")
print("Range within 1 standard deviation of the mean: ", age_range_1std)
print("Range within 2 standard deviations of the mean: ", age_range_2std)
print("Range within 3 standard deviations of the mean: ", age_range_3std)

print("In the Age column, approximately 68% of the data falls within one standard deviation from the mean, which is between 24.92 and 54.27. Approximately 95% of the data falls within two standard deviations from the mean, which is between 10.25 and 68.94. Approximately 99.7% of the data falls within three standard deviations from the mean, which is between -4.43 and 83.62.")


# <h2 style="text-align:center;background-color:#a8dda8">5. Detecting Outliers</h2>

# In[261]:


import matplotlib.pyplot as plt

# created a scatter plot of work experience
plt.scatter(df.index, df['Work Experience'])


plt.title('Scatter plot of Work Experience')
plt.xlabel('Index')
plt.ylabel('Work Experience')

# calculated outliers
q1, q3 = df['Work Experience'].quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[(df['Work Experience'] < lower_bound) | (df['Work Experience'] > upper_bound)]['Work Experience']
print('Outliers:', list(outliers))

t
for i, outlier in outliers.iteritems():
    plt.annotate(str(outlier), xy=(i, outlier))


plt.show()

print("The outliers indicate some customers with extremely high or low work experience compared to the rest of the customers and the  same applies for age.")


# In[262]:


import matplotlib.pyplot as plt

# create a scatter plot of age
plt.scatter(df.index, df['Age'])

# calculate and print the outliers
q1, q3 = df['Age'].quantile([0.25, 0.75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]['Age']
print('Outliers:', list(outliers))

# highlight the outliers in the plot
plt.scatter(outliers.index, outliers, color='red', marker='x')

# add a title and labels to the plot
plt.title('Scatter plot of Age')
plt.xlabel('Index')
plt.ylabel('Age')

# show the plot
plt.show()


# <h1 style="background: linear-gradient(to right,  #b2dfdb, #e0f2f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;">
#     Exploratory Data Analysis
# </h1>

# <h2 style="text-align:center;background-color:#a8dda8">1. Five-Number Summary</h2>

# In[181]:


import pandas as pd
import seaborn as sns


# Calculate the five-number summary for Age
age_summary = df['Age'].describe()[['min', '25%', '50%', '75%', 'max']]
age_summary.index = ['Minimum', 'Q1', 'Median', 'Q3', 'Maximum']

# Calculate the five-number summary for Work Experience
exp_summary = df['Work Experience'].describe()[['min', '25%', '50%', '75%', 'max']]
exp_summary.index = ['Minimum', 'Q1', 'Median', 'Q3', 'Maximum']

# Create a boxplot for Age, Annual Income ($), and Work Experience
sns.boxplot(data=df[['Age', 'Work Experience']], orient='h')

# Add labels to the boxplot
labels = ['Age', 'Work Experience']
for i, label in enumerate(labels):
    plt.text(df[label].median(), i, "Median: {}".format(round(df[label].median(), 2)), 
             bbox=dict(facecolor='white', alpha=0.5), ha='center')
    plt.text(df[label].quantile(0.25), i-.2, "Q1: {}".format(round(df[label].quantile(0.25), 2)), 
             bbox=dict(facecolor='white', alpha=0.5), ha='center')
    plt.text(df[label].quantile(0.75), i+.2, "Q3: {}".format(round(df[label].quantile(0.75), 2)), 
             bbox=dict(facecolor='white', alpha=0.5), ha='center')
    plt.text(df[label].min(), i-.4, "Min: {}".format(round(df[label].min(), 2)), 
             bbox=dict(facecolor='white', alpha=0.5), ha='center')
    plt.text(df[label].max(), i+.4, "Max: {}".format(round(df[label].max(), 2)), 
             bbox=dict(facecolor='white', alpha=0.5), ha='center')

plt.yticks(range(len(labels)), labels)
plt.title('Boxplot with Five-Number Summary')
plt.show()


# <h2 style="text-align:center;background-color:#a8dda8">2. Box Plot</h2>

# In[263]:


import pandas as pd
import seaborn as sns


# Create a boxplot for Work Experience
sns.boxplot(x=df['Work Experience'], color='skyblue')

# Add labels and title
plt.xlabel("Work Experience (years)")
plt.title("Box Plot of Work Experience")

# Show the plot
plt.show()

# Create a boxplot for Age
sns.boxplot(x=df['Age'], color='pink')

# Add labels and title
plt.xlabel("Age")
plt.title("Box Plot of Age")

# Show the plot
plt.show()
print("The box plot for work experience shows that the median work experience is around 15 years, with the majority of values falling within the IQR of 5 to 20 years. There are a few outliers with work experience greater than 35 years.")


# <h1 style="background: linear-gradient(to right,  #b2dfdb, #e0f2f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;">
#     Measures of Association
# </h1>

# <h2 style="text-align:center;background-color:#a8dda8">1. Covariance</h2>

# In[267]:


import pandas as pd

# Calculated the covariance between Age and Work Experience
covariance = df['Age'].cov(df['Work Experience'])

print("Covariance between Age and Work Experience:", round(covariance, 2))
print("A covariance value of 160.77 between Age and Work Experience indicates that there is a positive relationship between the two    variables. It means that as one variable (age or work experience) increases, the other variable is also likely to increase. ")


# <h2 style="text-align:center;background-color:#a8dda8">2. Correlation Coefficient</h2>

# In[268]:


import pandas as pd

# Calculate the correlation coefficient
corr_coef = df['Age'].corr(df['Work Experience'])

# Print the result rounded off to two decimal places
print("Correlation coefficient between age and work experience:", round(corr_coef, 2))

print("A correlation coefficient of 0.95 means that there is a very strong positive correlation between age and work experience. This indicates that as age increases, so does work experience.")


# <h1 style="background: linear-gradient(to right,  #b2dfdb, #e0f2f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;">
#     Additional Statistics
# </h1>

# <h2 style="text-align:center;background-color:#a8dda8">1. Sample Variance</h2>

# In[279]:


sample_var_age = df['Age'].var(ddof=1)
print("Sample variance of Age:", round(sample_var_age, 2))
print("The sample variance of age is 216.31, which indicates that the age values in the dataset are relatively spread out from the    mean age value. This means that the customers' ages are diverse and vary significantly from each other.")


# <h2 style="text-align:center;background-color:#a8dda8">2. Kurtosis</h2>
# 

# In[276]:


from scipy.stats import kurtosis

# Calculated kurtosis for Age column
age_kurtosis = kurtosis(df['Age'])

print("Kurtosis for Age column:", round(age_kurtosis, 2))
print("A kurtosis value of -0.52 for the Age column indicates that the distribution of ages is flatter than a normal distribution,    with fewer outliers and less extreme values in the tails. This means that the ages of the customers are more tightly clustered around the mean, and there are fewer customers with very high or very low ages compared to what would be expected in a normal  distribution.")


# In[ ]:





# <h1 style="background: linear-gradient(to right,  #b2dfdb, #e0f2f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;">
#     Reflections on your learnings between this assignment and the previous one. What was different between Python and Excel? When will you use either of them?  If you could do this assignment over what would you do differently? What are the gaps in your programming or stats knowledge? (350 words)
# </h1>

# In reflecting upon my learnings from the two assignments, numerous key distinctions emerged between when I used Excel and when I used Jupyter Notebook with Python for performing the same statistical analysis on the same dataset of my choice. I found Excels biggest advantage in my view is its user-friendly interface which would make it accessible to perform data analysis without programming knowledge. On the other hand, Python, as a programming language, offers more flexibility and easier debugging  when working with my dataset, especially in combination with libraries such as pandas, numpy, and scipy and additionally I found  python combined with Jupyter Notebook allows for easy sharing and reproducing of code and easier debugging and reusability of code ensuring consistent results.
# 
# The most notable difference between Python and Excel is the ease of automation and replicability with Python. In Excel, hard coding the statistics may lead to errors and is less efficient when dealing with large datasets or complex analyses and the debugging process requires more clicks therefore it is to a certain extent less efficient and more time consuming. Additionally, Excel's limited formula language can hinder the implementation of advanced statistics. In contrast, Python enabled me to write concise and efficient code, with a wide range of available libraries to perform advanced statistical operations. Moreover, Python allows for seamless integration with other tools and technologies such as GitHub, making it a versatile option.
# 
# The choice between Excel and Python would depend on the nature and complexity of the task. For smaller datasets and simpler statistical analyses, I would use Excel. However, for more complex tasks, larger datasets, or projects requiring automation and reproducibility, I would use Python.
# 
# If given the opportunity to redo the assignment, I would focus on understanding the different libraries in Python which would enable me to take full advantage of Python's capabilities. I would also familiarise myself with understanding more of Jupiter notebooks capabilities to ensure I maximise my productivity and reduce the errors I make. This might involve ensuring that the code is well-documented and modular for better maintainability. Furthermore, I would like to explore visualization libraries such as matplotlib or seaborn in more detail to enhance the presentation of the statistics, making it easier to convey insights to a non-technical audience.
# 
# This experience has also highlighted some gaps in my programming and statistical knowledge. While I am familiar with basic programming concepts and common statistical methods, I recognise the need to deepen my understanding of both areas. For programming, this could involve learning more about best practices, optimisation techniques and quicker debugging techniques. Regarding statistics, I would benefit from exploring more advanced methods, such as hypothesis testing, time series analysis and machine learning algorithms. By addressing these gaps, I will be better equipped to handle a wider range of tasks and challenges in future regarding programming and statistics.
# 
