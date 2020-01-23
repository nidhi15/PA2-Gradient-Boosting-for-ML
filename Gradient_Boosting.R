#Gradient Boosting Learning for ML

# --------------------- Step 1 - Load the packages and generate the data ---------------------
# Install the required packages and load the libraries
require(rpart)
require(rpart.plot)
require(dplyr)
require(data.table)

library(rpart)          #Used for recursive partitioning for classification,regression trees
library(rpart.plot)     #Used for plotting the models generated using rpart package
library(dplyr)          #used to transform and summarize data with rows and columns
library(data.table)     

#Generating the dataset values of x and y using the given approximate function
x_data <- seq(0, 2, 0.005)  #interval [0,2] incrementing by 0.005
y_data <-  0.8*cos(3.2*3.14*x_data) + 0.64*cos(10.24*3.14*x_data) + 0.51*cos(32.77*3.14*x_data)
df <- data.frame(x_data,y_data)

#Glimpse of the generated dataset values
head(df)
dim(df)   #showing 401 samples 

#Plotting the data values for the given function(x)
plot(x_data, y_data, col="blue", xlab = "x", ylab = "y", main = 'Scatter plot of the Data(x,y)', font.main=1)   #Data Distribution
plot(x_data, y_data, type = 'line', col="blue", xlab = "x", ylab = "y = f(x)", main = 'Graph for the f(x)=0.8cos(3.2??x) + 0.64cos(10.24??x) + 0.51cos(32.77??x)', font.main=1) #Line graph

#--------------------- Step 2 - Modeling of the Data ---------------------
#Creating the empty list and dataframe to store the errors & predicted points 
error_list <- c()
pred_data <- data.frame()

#Learning Rate - alpha value
alpha = 0.05

# Create a decision tree model with rpart
#Creating Tree 1
fit_tree <- rpart(df$y_data ~ df$x_data, data = df)

#Visualize the decision tree with rpart.plot
rpart.plot(fit_tree, box.palette="RdBu", shadow.col="gray", nn=TRUE, main="Regression Tree Diagram for the given function(x)")

# -----------------------------Step 3 - Prediction --------------------------------
# Iteration 1 - Prediction of y value 
df$y_pred <- predict(fit_tree, data = df$x_data)
y1<-df$y_pred                   # Storing the 1st Y predicted value

# Calculating the error residual
df$er <- df$y_data - df$y_pred   
error <- df$er

#Using MAE for calculating the accuracy of the model
mae_acc <- sum(abs(df$y_data - df$y_pred)) # formula defining the accuracy 
message("Mean absolute Error of the Model is : ", mae_acc)
error_list <- c(error_list,mae_acc) #Adding error values in the error list

#Creating an error table to store error generated for each tree
err_tbl <- data.frame(matrix(nrow = 401, ncol = 1))
colnames(err_tbl) <- c("Data_Samples")
err_tbl$Data_Samples <- 1:401

#Defining error tree counter
counter = 1

# ----------------------------- Step 4 - Gradient Boosting --------------------------------
#Implementing Gradient boosting algorithm
#Running the algorithm with stopping criteria to minimize the error
while(mae_acc > 20.05){
  
  df$er <- df$y_data - df$y_pred   #Calculates the residuals
  err_tbl[,paste0("Error after ",counter," tree",sep="")] <- df$er #stores each tree error values in a table for the report
  counter <- counter+1 #incrementing for each iteration
  message("Mean absolute error:", mae_acc)
  
  fit_tree_1 <- rpart(df$er ~ df$x_data, data = df, control = rpart.control(cp = 0.0000000000000000001)) #new sub tree formation
  df$y_pred_1 <- predict(fit_tree_1, data = df$x_data) #new predicted value
  
  df$y_pred <- df$y_pred + alpha*df$y_pred_1  # Adjusting the predicted y value with the pervious one
  mae_acc <- sum(abs(df$y_data - df$y_pred)) #Mean absolute error - accuracy function
 
  error_list <- c(error_list, mae_acc)
  
  pred_data <- rbind(pred_data, df$y_pred)  #storing all the predicted values
  err_tbl <- rbind(err_tbl,df$err)          #storing all the error values 
}   #end of while loop

#List of errors and no. of trees
error_list
length(error_list) 

#Error Table for error progression report
head(err_tbl)
error_progression_report <- as.data.table(err_tbl)
View(error_progression_report)

#Writing the error report in a file
write.csv(error_progression_report, file = "error_progression_report_0.05.csv")

#Column name for the predicted values dataframe
colnames(pred_data) <- seq(1, nrow(df))
head(pred_data)  #Glimpse of the predicted values
dim(pred_data)   

#Plotting the final Model graph 
plot(df$x_data, df$y_data, type = 'line', col = 'blue', lwd = 1, xlab = "x", ylab = "y = f(x)")
lines(df$x_data, y1, col = 'darkolivegreen3', lwd = 2)
lines(df$x_data, pred_data[51,], col = 'dimgrey', lwd = 2)
lines(df$x_data, pred_data[359,], col = 'red', lwd = 1)
legend("bottomright", legend=c("Actual Function","1st Prediction Level ",
       "50th Prediction Level", "Different Prediction Level"), 
       lwd=c(1,2,2,1), col=c("blue","darkolivegreen3", "dimgrey", "red"), cex = 0.53)
title(main='Original Function and Approximation at Different Levels of Accuracy', 
      cex.main = 1, font.main = 1, col.main = "black", outer = FALSE)

# ----------------------------- End of the algorithm --------------------------------
