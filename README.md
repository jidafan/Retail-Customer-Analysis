# Retail Customer Analysis

## Table of Contents
* [Introduction](#introduction)
* [Code](#code)

### Introduction

The project shown is the customer analysis of shoppers from an online retail transaction data set of two years. The program uses KMeans Clustering to group customers and determine strategies for retaining and engaging customers. The data is from a UCI dataset that can be viewed [here](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

### Code

The code for the project can be viewed [here](https://github.com/jidafan/Retail-Customer-Analysis/blob/main/Customers.ipynb)

#### Exploratory Data Analysis

Before we begin working on the dataset, we must understand the data. In this step of the code, we create a data frame using Pandas to hold all the information from the Excel file. 

![image](https://github.com/user-attachments/assets/f49d6abc-1558-4eba-a718-69f31a56cbd1)

After taking a look at the data. Next is exploring the columns and searching for invalid entries and missing data.

From looking at the UCI website, we know that there are guidelines that the variables must follow to be a valid entry.

| Variable      | Guideline        | 
| ------------- |:---------------------| 
| `InvoiceNo`     | 6 digit number, can contain C at the end which indiciates a cancellation   |
| `StockCode`     | 5 digit number  |   
| `UnitPrice` | price > 0                       |
| `CustomerID`     | 5 digit identifier for customers  |

