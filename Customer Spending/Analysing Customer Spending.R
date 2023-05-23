#3rd data set

#load data from your computer into R
c_data <- read.csv("Customers_Data_MA7007_CW.csv")
# Find the online link for this data set

# Randomly select 80% of the data
set.seed(123)
c_data <- c_data[sample(nrow(c_data), nrow(c_data)*0.8),]

# View the first few rows of the data
head(c_data)

# Size of data set
dim(c_data)

#The purpose of analyzing the customer data set can be to gain insights into customer behavior, preferences, and characteristics. This analysis can help businesses make informed decisions about their marketing strategies, product development, and customer service. For example, by analyzing the customer data set, a business can identify its most valuable customers and design targeted marketing campaigns to retain them. Similarly, they can identify customer segments with similar characteristics and create personalized product recommendations. The analysis of customer data set can also help in identifying areas where customer service can be improved and optimized. Therefore, analyzing the customer data set can provide valuable information to businesses for making data-driven decisions to enhance customer satisfaction, loyalty, and profitability.

# Check for missing data in every column
missing_values <- colSums(is.na(c_data))

# Print the number of missing values in each column
print(missing_values)
# so No missing value
library(worms)
c_data_df <- data.frame(Age = c_data$Age, Annual_Income = c_data$Annual_Income)
wp(c_data_df, xlab = "Age", ylab = "Annual Income", main = "Worm Plot of Age vs. Annual Income")

library(gamlss)

#rename the column heading
colnames(c_data) <- c("Customer_ID", "Gender", "Age", "Annual_Income", "Spending_Score", "Profession", "Work_Experience", "Family_Size")

 
library(ggplot2)

# Histogram of Age
ggplot(c_data, aes(x=Age)) +
  geom_histogram(binwidth=5, color="black", fill="blue") +
  ggtitle("Distribution of Age")

# Histogram of Annual Income
ggplot(c_data, aes(x=Annual_Income)) +
  geom_histogram(binwidth=5000, color="black", fill="green") +
  ggtitle("Distribution of Annual Income")

# Histogram of Spending Score
ggplot(c_data, aes(x=Spending_Score)) +
  geom_histogram(binwidth=5, color="black", fill="orange") +
  ggtitle("Distribution of Spending Score")

#distributions of Age, Annual Income, and Spending Score are roughly normal.

# Boxplot of Spending Score by Gender
ggplot(c_data, aes(x=Gender, y=Spending_Score)) +
  geom_boxplot(fill="blue", color="black") +
  ggtitle("Spending Score by Gender") +
  stat_summary(fun.data = "mean_cl_boot", geom = "errorbar", 
               width = 0.2, color = "red") +
  stat_summary(fun.y = "mean", geom = "point", color = "red", size = 3) +
  ylab("Spending Score")

# Calculate and print IQR
iqr_male <- IQR(c_data$Spending_Score[c_data$Gender == "Male"])
iqr_female <- IQR(c_data$Spending_Score[c_data$Gender == "Female"])
cat("IQR for Male:", iqr_male, "\n")
cat("IQR for Female:", iqr_female, "\n")


#The boxplot suggests that there is some difference in the Spending Score between males and females, with females having slightly higher scores on average.

# Scatterplot of Annual Income vs Spending Score
# Load libraries and data
library(ggplot2)
# Exclude rows with missing data
c_data <- c_data[complete.cases(c_data), ]

# Plot the data
ggplot(c_data, aes(x=Annual_Income, y=Spending_Score)) +
  geom_point(color="purple") +
  ggtitle("Annual Income vs Spending Score")

# Calculate the correlation
corr <- cor(c_data$Annual_Income, c_data$Spending_Score, use = "pairwise.complete.obs")

# Print the correlation and its type
if(!is.na(corr)) {
  if(corr > 0) {
    cat("The correlation is positive: ", round(corr, 2))
  } else if(corr < 0) {
    cat("The correlation is negative: ", round(corr, 2))
  } else {
    cat("There is no correlation")
  }
} else {
  cat("Correlation cannot be calculated due to missing values")
}


#The scatterplot shows a positive correlation between Annual Income and Spending Score, which is expected.

#scatterplot of age and spending score
# Load libraries and data
library(ggplot2)

# Exclude rows with missing data
c_data <- c_data[complete.cases(c_data), ]

# Plot the data
ggplot(c_data, aes(x = Age, y = Spending_Score)) +
  geom_point() +
  labs(x = "Age", y = "Spending Score")

# Calculate the correlation
corr <- cor(c_data$Age, c_data$Spending_Score, use = "pairwise.complete.obs")

# Print the correlation and its type
if(!is.na(corr)) {
  if(corr > 0) {
    cat("The correlation is positive: ", round(corr, 2))
  } else if(corr < 0) {
    cat("The correlation is negative: ", round(corr, 2))
  } else {
    cat("There is no correlation")
  }
} else {
  cat("Correlation cannot be calculated due to missing values")
}

#scatterplot of annual income and spending score
ggplot(c_data, aes(x = Annual_Income, y = Spending_Score)) + geom_point() + labs(x = "Annual Income", y = "Spending Score")

#box plot of profession and spending score
# Check the data for missing or non-finite values
summary(c_data$Spending_Score)
str(c_data$Spending_Score)

# Remove any missing or non-finite values
c_data <- na.omit(c_data)

# Plot the boxplot
ggplot(c_data, aes(x = Profession, y = Spending_Score)) + 
  geom_boxplot() + 
  labs(x = "Profession", y = "Spending Score")

# Calculate the IQR of Spending_Score for each profession
iqr_by_profession <- tapply(c_data$Spending_Score, c_data$Profession, IQR, na.rm = TRUE)

# Print the results
cat("IQR by Profession:\n")
print(format(round(iqr_by_profession, 2), nsmall = 2))


#scatterplot spending score and work experience
# Load libraries and data
library(ggplot2)

# Exclude rows with missing data
c_data <- c_data[complete.cases(c_data), ]

# Plot the data
ggplot(c_data, aes(x = Work_Experience, y = Spending_Score)) +
  geom_point() +
  labs(x = "Work Experience", y = "Spending Score")

# Calculate the correlation
corr <- cor(c_data$Work_Experience, c_data$Spending_Score, use = "pairwise.complete.obs")

# Print the correlation and its type
if(!is.na(corr)) {
  if(corr > 0) {
    cat("The correlation is positive: ", round(corr, 2))
  } else if(corr < 0) {
    cat("The correlation is negative: ", round(corr, 2))
  } else {
    cat("There is no correlation")
  }
} else {
  cat("Correlation cannot be calculated due to missing values")
}


#scatterplot family size and spending score
# Load libraries and data
library(ggplot2)

# Exclude rows with missing data
c_data <- c_data[complete.cases(c_data), ]

# Plot the data
ggplot(c_data, aes(x = Family_Size, y = Spending_Score)) +
  geom_point() +
  labs(x = "Family Size", y = "Spending Score")

# Calculate the correlation
corr <- cor(c_data$Family_Size, c_data$Spending_Score, use = "pairwise.complete.obs")

# Print the correlation and its type
if(!is.na(corr)) {
  if(corr > 0) {
    cat("The correlation is positive: ", round(corr, 2))
  } else if(corr < 0) {
    cat("The correlation is negative: ", round(corr, 2))
  } else {
    cat("There is no correlation")
  }
} else {
  cat("Correlation cannot be calculated due to missing values")
}


# Create histograms for numerical variables
#par(mfrow=c(2,3))
hist(c_data$Age, main="Age")
hist(c_data$Annual_Income, main="Annual Income")
hist(c_data$Spending_Score, main="Spending Score")
hist(c_data$Work_Experience, main="Work Experience")
hist(c_data$Family_Size, main="Family Size")

library(ggplot2)

# Age
library(ggplot2)

ggplot(c_data, aes(x = Age)) +
  geom_histogram(aes(y = ..density..), binwidth = 5, fill = "lightblue", color = "black") +
  geom_density(aes(y = ..density..), color = "red", size = 1) +
  labs(title = "Age Histogram with Normal Curve") +
  scale_y_continuous(name = "Density")


# Annual Income
ggplot(c_data, aes(x = Annual_Income)) +
  geom_histogram(aes(y = ..density..), binwidth = 5000, fill = "lightgreen", color = "black") +
  geom_density(color = "red", size = 1) +
  labs(title = "Annual Income Histogram with Normal Curve")

# Spending Score
ggplot(c_data, aes(x = Spending_Score)) +
  geom_histogram(aes(y = ..density..), binwidth = 5, fill = "lightpink", color = "black") +
  geom_density(color = "red", size = 1) +
  labs(title = "Spending Score Histogram with Normal Curve")

# Work Experience
ggplot(c_data, aes(x = Work_Experience)) +
  geom_histogram(aes(y = ..density..), binwidth = 1, fill = "lightyellow", color = "black") +
  geom_density(color = "red", size = 1) +
  labs(title = "Work Experience Histogram with Normal Curve")

# Family Size
ggplot(c_data, aes(x = Family_Size)) +
  geom_histogram(aes(y = ..density..), binwidth = 1, fill = "lightblue", color = "black") +
  geom_density(color = "red", size = 1) +
  labs(title = "Family Size Histogram with Normal Curve")





# Create scatterplots for numerical variables
#par(mfrow=c(2,3))
# Load data
# Load data

# Define function to print correlation coefficient and create scatter plot
plot_corr <- function(x, y, title) {
  corr <- round(cor(x, y), 2)
  if (corr > 0) {
    cat(title, ": Correlation: ", corr, " (Positive)\n")
  } else if (corr < 0) {
    cat(title, ": Correlation: ", corr, " (Negative)\n")
  } else {
    cat(title, ": Correlation: ", corr, " (No correlation)\n")
  }
  plot(x, y, main = title)
}

# Calculate correlation coefficients and create scatter plots
plot_corr(c_data$Age, c_data$Annual_Income, "Age vs. Annual Income")
plot_corr(c_data$Age, c_data$Spending_Score, "Age vs. Spending Score")
plot_corr(c_data$Annual_Income, c_data$Spending_Score, "Annual Income vs. Spending Score")
plot_corr(c_data$Work_Experience, c_data$Spending_Score, "Work Experience vs. Spending Score")
plot_corr(c_data$Family_Size, c_data$Spending_Score, "Family Size vs. Spending Score")







ggplot(c_data, aes(x = Age, y = Annual_Income)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Age vs. Annual Income with Fitted Line")

ggplot(c_data, aes(x = Age, y = Spending_Score)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Age vs. Spending Score with Fitted Line")

ggplot(c_data, aes(x = Annual_Income, y = Spending_Score)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Annual Income vs. Spending Score with Fitted Line")

ggplot(c_data, aes(x = Work_Experience, y = Spending_Score)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Work Experience vs. Spending Score with Fitted Line")

ggplot(c_data, aes(x = Family_Size, y = Spending_Score)) +
  geom_point(color = "blue") +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Family Size vs. Spending Score with Fitted Line")



# Fit the multiple linear regression model
model <- lm(Spending_Score ~ Age + Annual_Income + Work_Experience + Family_Size, data = c_data)

# Check the assumptions of the model
par(mfrow = c(2, 2)) # Set up a 2x2 grid of plots

# Residuals vs. Fitted plot
plot(model, which = 1)

# Normal Q-Q plot
plot(model, which = 2)

# Scale-Location plot
plot(model, which = 3)

# Residuals vs. Leverage plot
plot(model, which = 4)

# Reset the plot layout to default
# Reset the plot layout to default
par(mfrow = c(1, 1))

# Find the first non-missing row in the data set
# Find the first non-missing row in the data set
# Remove the first observation from the dataset and save it as test_obs
# Fit the multiple linear regression model
model <- lm(Spending_Score ~ Age + Annual_Income + Work_Experience + Family_Size, data = c_data)

# Remove the first observation from the dataset and save it as test_obs
test_obs <- c_data[1, ]
c_data <- c_data[-1, ]

# Predict the value of Spending_Score for the first observation and obtain the prediction interval
predicted_score <- predict(model, newdata = test_obs, interval = "prediction")

# Print the predicted value and prediction interval
cat("Predicted Score:", predicted_score[1], "\n")
cat("Prediction Interval:", predicted_score[2], "to", predicted_score[3], "\n")

# Plot the distribution of the predicted value
hist(predicted_score, main = "Distribution of Predicted Spending Score", xlab = "Spending Score")


# Display a summary of the model
summary(model)




