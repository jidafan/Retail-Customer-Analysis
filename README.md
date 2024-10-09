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

#### Data Cleaning
