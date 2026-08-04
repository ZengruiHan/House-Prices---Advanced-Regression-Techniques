# House-Prices---Advanced-Regression-Techniques

## Project Description

This project focuses on predicting house sale prices based on various property characteristics. The training dataset contains 1460 rows and 81 columns, representing 1460 houses and 79 explanatory features, together with an identifier column (Id) and the target variable (SalePrice). The features include integer, floating-point, and categorical variables.

An initial exploration shows that several features have a high proportion of missing values. For example, approximately 99.5% of the values in PoolQC and 96.3% of the values in MiscFeature are missing. However, a missing value does not necessarily indicate an error in the dataset. In some cases, it may indicate that a property does not have the corresponding feature, such as a swimming pool or a miscellaneous additional structure. Therefore, the meanings of missing values should be examined before choosing an imputation strategy.

The next steps will be to study the distribution of SalePrice, separate numerical and categorical features, construct an appropriate preprocessing pipeline, and establish a simple regression baseline. Linear regression or Ridge regression can be used as initial models, followed by tree-based regression models for comparison. Model performance will be evaluated using cross-validation and root mean squared error on the logarithm of the sale price.