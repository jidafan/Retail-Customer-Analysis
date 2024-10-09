# Retail Customer Analysis

- [1. Introduction](#1-introduction)
- [2. Code](#2-code)
  * [2.1 Exploratory Data Analysis](#21-exploratory-data-analysis)
  * [2.2 Data Cleaning](#22-data-cleaning)
  * [2.3 Feature Engineering](#23-feature-engineering)
  * [2.4 KMeans Clustering](#24-kmeans-clustering)
    + [2.4.1 Cluster Analysis](#241-cluster-analysis)
    + [2.4.2 Outlier Analysis](#242-outlier-analysis)
  * [2.5 Visualization](#25-visualization)
- [3. Conclusion](#3-conclusion)

### 1. Introduction

The project shown is the customer analysis of shoppers from an online retail transaction data set of two years. The program uses KMeans Clustering to group customers and determine strategies for retaining and engaging customers. The data is from a UCI dataset that can be viewed [here](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

### 2. Code

The code for the project can be viewed [here](https://github.com/jidafan/Retail-Customer-Analysis/blob/main/Customers.ipynb)

#### 2.1 Exploratory Data Analysis

Before we begin working on the dataset, we must understand the data. In this step of the code, we create a data frame using Pandas to hold all the information from the Excel file. 

![image](https://github.com/user-attachments/assets/f49d6abc-1558-4eba-a718-69f31a56cbd1)

After taking a look at the data. Next is exploring the columns and searching for invalid entries and missing data.

From looking at the UCI website, we know that there are guidelines that the variables must follow to be a valid entry.

| Variable      | Guideline        | 
| ------------- |:---------------------| 
| `InvoiceNo`     | 6 digit number, can contain C at the end which indiciates a cancellation   |
| `StockCode`     | 5 digit number  |   
| `UnitPrice` | Price > 0                       |
| `CustomerID`     | 5 digit identifier for customers  |
| `Quantity` | Quantity > 0                       |

From analyzing the columns, we see that there a lot of entries that do meet the requirements and should not be included in the analysis.

However, in the StockCode column we find that there are many unique codes that do not follow the guidelines, so we must analyze each code and determine whether or not they should be included.

![image](https://github.com/user-attachments/assets/f3d5778f-bf44-4451-b9d7-9adaa2559501)

Following the investigation we find that,

* Stock codes follow a specific format of 5 digits from [0-9], however some do have a letter from [a-z] after the digits. The cases listed below are special cases

| Variable      | Description           |  Action |
| ------------- |-----------------| :----------------- |
| `DCGS`     | Looks valid, however some quantities are negative and missing customer ID | Exclude from clustering |
| `D`     | Looks valid and it represents discount values  | Exclude from clustering |
| `DOT`     | Looks valid and represents the charges from postage | Exclude from clustering |
| `M or m`     | Looks valid represents manual transactions | Exclude from clustering |
| `C2`     | Carriage transaction | Exclude from clustering |
| `C3`     | Only one value | Exclude |
| `Bank Charges or B`     | Bank charges | Exclude from clustering |
| `S`     | Samples sent to customers | Exclude from clustering |
| `TESTXXX`     | Testing data | Exclude from clustering |
| `gift_XXX`     | Purchases from Gift Card, does not hold customer data | Exclude |
| `PADS`     | Stock Code for padding, unique case | Include |
| `SP1002`     | 3 transactions for a special item, however, 1 has no pricing | Exclude |
| `AMAZONFEE`     | Amazon shipping fee | Exclude |
| `ADJUSTX`     | Manual adjustments by Admin | Exclude |

We find that only the unique case of 'PADS' is a valid code to include in our analysis.

#### 2.2 Data Cleaning

The first step of our data cleaning process is to only include valid invoices, which are 6 digit numbers. To do this we create a mask function which will only capture valid InvoiceNos.

![image](https://github.com/user-attachments/assets/5d931c8e-de6c-40b2-9f68-177509c4ef21)

Next, we look at valid stock codes, which we determined is any 5 digit number, 5 digit numbers that have a letter after, and PADS.

![image](https://github.com/user-attachments/assets/bc49dcde-6af0-448f-b50f-bdf4504098f3)

After, this we remove any customer IDS that are null and any prices that are <= 0.

Following this processes, we find that after cleaning the dataset. We have retained 77% of the data, losing 23% of the data in the process.

#### 2.3 Feature Engineering

The first step in this process is creating a new Dataframe with only information that we need for our cluster analysis.

```
cleaned_df["SalesLineTotal"] = cleaned_df["Quantity"] * cleaned_df["Price"]

aggregated_df = cleaned_df.groupby(by="Customer ID", as_index=False) \
    .agg(
        MonetaryValue=("SalesLineTotal", "sum"),
        Frequency=("Invoice", "nunique"),
        LastInvoiceDate=("InvoiceDate", "max")
    )

max_invoice_date = aggregated_df["LastInvoiceDate"].max()
max_invoice_date

aggregated_df["Recency"] = (max_invoice_date - aggregated_df["LastInvoiceDate"]).dt.days
```

![image](https://github.com/user-attachments/assets/8c5df388-1711-472b-9e6d-41d59ed16238)

This dataframe contains only the columns needed for analysis. 

| Variable      | Description   | 
| ------------- |:---------------------| 
| `Customer ID`     | A 5 digit identifier for customers   |
| `MonetaryValue`     | The amount of money that a customer has spent in total |   
| `Frequency` | How many purchases a customer has made                  |
| `LastInvoiceDate`     | The date of the customer's most recent purchase |
| `Recency` | How many days since the customer's last purchase                    |

Afterwards, we create visualizations to see the data and if there are outliers.

![image](https://github.com/user-attachments/assets/d20e77d8-c06d-4e54-8575-00be9ffddc98)

![image](https://github.com/user-attachments/assets/eb9cac67-290d-42d4-abb9-2f795c3c7815)

From the boxplots, we see that there are many outliers in our data. These outliers may possess important information, so we create a seperate data frame that has no outliers and two data frames to capture the Monetary Outliers and the Frequency Outliers

Monetary Outliers
```
M_Q1 = aggregated_df["MonetaryValue"].quantile(0.25)
M_Q3 = aggregated_df["MonetaryValue"].quantile(0.75)
M_IQR = M_Q3 - M_Q1

monetary_outliers_df = aggregated_df[(aggregated_df["MonetaryValue"] > (M_Q3 + 1.5* M_IQR)) | (aggregated_df["MonetaryValue"] < (M_Q1 - 1.5* M_IQR))].copy()
```

Frequency Outliers
```
F_Q1 = aggregated_df["Frequency"].quantile(0.25)
F_Q3 = aggregated_df["Frequency"].quantile(0.75)

F_IQR = F_Q3 - F_Q1

frequency_outliers_df = aggregated_df[(aggregated_df["Frequency"] > (F_Q3 + 1.5* F_IQR)) | (aggregated_df["Frequency"] < (F_Q1 - 1.5* F_IQR))].copy()
```

Data frame with no outliers
```
non_outliers_df = aggregated_df[(~aggregated_df.index.isin(monetary_outliers_df.index)) & (~aggregated_df.index.isin(frequency_outliers_df.index))]
```
Afterwards, we visualize the data without any outliers to see the changes.

![image](https://github.com/user-attachments/assets/d7dd225d-9498-4452-9427-48a3e6c258a6)

Next, we standardize the columns to ensure that they all have an equal impact during the analysis. We use the StandardScalar method from scikit-learn to accomplish this

```
scalar = StandardScaler()
scaled_data = scalar.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])

scaled_data_df = pd.DataFrame(scaled_data, index=non_outliers_df.index, columns=("MonetaryValue", "Frequency", "Recency"))
scaled_data_df
```

#### 2.4 KMeans Clustering

We select an upper limit of 12 for how many clusters we should have, and we use the graphs to determine which is an appropriate amount of clusters. The silohuette score shows how distinct a cluster is, a higher score indicating more distinct clusters. Silohuette can range from [-1, 1]. -1 means that the data is all over lapping, and 1 means that there is no overlap in the data.

![image](https://github.com/user-attachments/assets/8c8e6d54-2743-4081-8ee1-e69bb014bfc6)

From the charts, we can see that around 4 - 5 is an appropriate amount of clusters to have. As, there is not much improvement by adding more, and it keeps it simple by not having to maintain a large amount of data frames.

We then apply the clusters to our non outliers data frame.

```
kmeans = KMeans(n_clusters=4, random_state=42, max_iter=1000)
cluster_labels = kmeans.fit_predict(scaled_data_df)

non_outliers_df["Cluster"] = cluster_labels
non_outliers_df
```

![image](https://github.com/user-attachments/assets/f3923c67-8445-4480-b128-13a74bf81d77)

We then visualize the clusters using a scatterplot to see the spread of the data.

![image](https://github.com/user-attachments/assets/4dead60d-6ab7-40fc-bde0-6a11c2f89ebf)

After we look at violin plots of the different clusters to examine, the correct course of action for each group.

![image](https://github.com/user-attachments/assets/b71031c4-823e-40ec-ac58-17f3561c18a0)

![image](https://github.com/user-attachments/assets/d0f71ccf-fa7b-4dc5-b490-8471aaa35855)

![image](https://github.com/user-attachments/assets/47b18f6b-40c1-481c-8c7c-a3e931182135)

#### 2.4.1 Cluster Analysis

1. Cluster 0 (Blue): "Retain"

Rationale: This cluster represents high value customers who purchase products regularly, however, not always recently. The focus should be on retention efforts to maintain loyalty and spending habits

Action: Implement loyalty programs, personalized offers and regular engagement to ensure active spending

2. Cluster 1 (Orange): "Re-engage"

Rationale: This group represents lower-value and infrequent buyers who haven't purchased products recently. The focus should be on re-engagement to bring them into active purchasing behaviour

Action: Use targeted-marketing compaigns, special discounts, or reminders to encourage them to return and purchase products

3. Cluster 2 (Green): "Nurture"

Rationale: This group represents the least active and lowest-value customers, but they have made recent purchases. These customers may be new or need nurturing to increase engagement and spending

Action: Focus on building relationships, providing excellent customer service, and offering incentives to encourage frequent purchases

4. Cluster 3 (Red): "Reward"

Rationale: This cluster includes high-value and frequent customers, many of whom are still purchasing. They are the most loyal customers, and rewarding their loyalty is key to maintaining engagement

Action: Implement a robust loyalty program, provide exclusive offers and recognize their loyalty to keep them engaged and satisfied

#### 2.4.2 Outlier Analysis

For our next step, we will analyze the information from the outliers that we excluded from the analysis in a prior step. As this information can still be valuable to us and help with our customer analysis

To do so, we create data frames that hold only the monetary outliers, the frequency outliers, and a data frame that holds common ones shared between them. After, we combine everything into one outlier data frame

```
overlap_indices = monetary_outliers_df.index.intersection(frequency_outliers_df.index)

monetary_only_outliers = monetary_outliers_df.drop(overlap_indices)
frequency_only_outliers = frequency_outliers_df.drop(overlap_indices)

monetary_and_frequency_outliers = monetary_outliers_df.loc[overlap_indices]

monetary_only_outliers["Cluster"] = -1
frequency_only_outliers["Cluster"] = -2
monetary_and_frequency_outliers["Cluster"] = -3

outlier_clusters_df = pd.concat([monetary_only_outliers, frequency_only_outliers, monetary_and_frequency_outliers])
```

We denote a cluster to each grouping of outliers.

Next, we visualize the data using violin plots as we did previously to determine the actions we need to take

![image](https://github.com/user-attachments/assets/20704b26-26f9-435d-a142-6e2792a1703e)

![image](https://github.com/user-attachments/assets/5815d9c1-dc42-4c45-a299-9f8353eebac4)

![image](https://github.com/user-attachments/assets/28f4404f-4a5f-47f6-9b55-1bfb8e2b1073)

1. Cluster -1 (Monetary Outliers): "Pamper"

Rationale: This cluster represents high spenders but not frequent buyers, their purchases are large but infrequent. 

Action: Focus on maintaining their loyalty with personalized offers or luxury services that cater to their high spending capacity

2. Cluster -2 (Frequency Outliers): "Upsell"

Rationale: This cluster represents frequent buyers who spend less per purchase, these customers are engaged but may benefit from upselling opportunities

Action: Implement loyalty programs or bundle deals to encourage higher spending per visit, given their frequent engagement

3. Cluster -3 (Monetary & Frequency Outliers): "Delight"

Rationale: This cluster represents customers who spend heavily and make frequent purchases. They are likely top-tier customers who require special attention.

Action: Develop VIP programs or exclusive offers to maintain their loyalty and encourage continued engagement
