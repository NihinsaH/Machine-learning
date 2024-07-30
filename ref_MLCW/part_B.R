library(readxl)
library(dplyr)

# Importing the excel file and reading the 3rd attribute
df <- read_excel("C:/Users/Asus/Desktop/ref_MLCW/ExchangeUSD.xlsx")
dfn <- df[[3]]

set.seed(123)

# Function to create input/output matrix for MLP training/testing
io_matrix <- function(dfn, input_delay) {
  X <- matrix(nrow = length(dfn) - input_delay, ncol = input_delay)
  y <- vector()
  for (i in 1:(length(dfn) - input_delay)) {
    X[i, ] <- dfn[i:(i + input_delay - 1)]
    y[i] <- dfn[i + input_delay]
  }
  return(list(X = X, y = y))
}

# Function to normalize data
normalize_data <- function(data) {
  scaled <- scale(data)
  scaler <- list(scale = attr(scaled, "scaled:center"), scale = attr(scaled, "scaled:scale"))
  scaled_data <- as.vector(scaled)
  return(list(scaled = scaled_data, scaler = scaler))
}

# Split the data into training and testing sets
train <- dfn[1:400]
test <- dfn[401:500]

# Choose delay
input_delay <- c(1, 2, 3, 4)

for (delay in input_delay) {
  # Create input/output matrices for training data
  train_input <- io_matrix(train, delay)$X
  train_output <- io_matrix(train, delay)$y
  
  print(paste("Input delay:", delay))
  
  print("I/O matrix for training data")
  print("Training data input: ")
  print(train_input)
  print("Training data output: ")
  print(as.matrix(train_output))
  
  # Create input/output matrices for testing data
  test_input <- io_matrix(test, delay)$X
  test_output <- io_matrix(test, delay)$y
  
  print("I/O matrix for testing data")
  print("Testing data input: ")
  print(test_input)
  print("Testing data output: ")
  print(as.matrix(test_output))
  
  
  # Normalize the training input and output data
  train_input_normalized <- normalize_data(train_input)$scaled
  train_output_normalized <- normalize_data(train_output)$scaled
  
  print("I/O matrix for normalized training data")
  print("Normalized training data input: ")
  print(train_input_normalized)
  print("Normalized training data output: ")
  print(train_output_normalized)
  
  # Normalize the testing input data
  test_input_normalized <- normalize_data(test_input)$scaled
  test_output_normalized <- normalize_data(test_output)$scaled
  
  print("I/O matrix for normalized testing  data")
  print("Normalized testing data input: ")
  print(test_input_normalized)
  print("Normalized testing data output: ")
  print(test_output_normalized)
}


library(neuralnet)
library(ggplot2)
library(Metrics)

# Function to calculate MAPE
mean_absolute_percentage_error <- function(y_true, y_pred) {
  mean(abs((y_true - y_pred) / y_true)) * 100
}

# Function to calculate sMAPE
symmetric_mean_absolute_percentage_error <- function(y_true, y_pred) {
  100 * mean(2 * abs(y_pred - y_true) / (abs(y_pred) + abs(y_true)))
}

# Define MLP models with different configurations
models <- list(
  # Model 1
  list(
    description = "1 hidden layer (10 nodes), input delay = 1",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(10), 
      linear.output = FALSE
    )
  ),
  # Model 2
  list(
    description = "2 hidden layers (20, 5 nodes), input delay = 1",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(20,5), 
      linear.output = FALSE
    )
  ),
  # Model 3
  list(
    description = "3 hidden layers (10, 5, 5 nodes), input delay = 1",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(10, 5, 5), 
      linear.output = FALSE
    )
  ),
  # Model 4
  list(
    description = "1 hidden layer (10 nodes), input delay = 2",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(10), 
      linear.output = FALSE
    )
  ),
  # Model 5
  list(
    description = "2 hidden layers (20, 10 nodes), input delay = 2",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(20, 10), 
      linear.output = FALSE
    )
  ),
  # Model 6
  list(
    description = "3 hidden layers (10, 5, 5 nodes), input delay = 2",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(10, 5, 5), 
      linear.output = FALSE
    )
  ),
  # Model 7
  list(
    description = "1 hidden layer (10 nodes), input delay = 3",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(10), 
      linear.output = FALSE
    )
  ),
  # Model 8
  list(
    description = "2 hidden layers (10, 5 nodes), input delay = 3",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(10, 5), 
      linear.output = FALSE
    )
  ),
  # Model 9
  list(
    description = "3 hidden layers (10, 5, 10 nodes), input delay = 3",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(10, 5, 10), 
      linear.output = FALSE
    )
  ),
  # Model 10
  list(
    description = "1 hidden layer (7 nodes), input delay = 4",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(7), 
      linear.output = FALSE
    )
  ),
  # Model 11
  list(
    description = "2 hidden layers (20,20 nodes), input delay = 4",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(20,20), 
      linear.output = FALSE
    )
  ),
  # Model 12
  list(
    description = "3 hidden layers (10, 20, 10 nodes), input delay = 4",
    model = neuralnet(
      train_output_normalized ~ train_input_normalized,
      data = data.frame(train_input_normalized, train_output_normalized),
      hidden = c(10,20, 10), 
      linear.output = FALSE
    )
  )
)

# Evaluate each model
results <- data.frame(
  Description = character(length(models)),
  RMSE = numeric(length(models)),
  MAE = numeric(length(models)),
  MAPE = numeric(length(models)),
  sMAPE = numeric(length(models))
)

for (i in 1:length(models)) {
  model <- models[[i]]$model
  description <- models[[i]]$description
  predictions <- predict(model, data.frame(test_input_normalized))
  predictions_denormalized <- (predictions * sd(test_output)) + mean(test_output)
  
  results$Description[i] <- description
  results$RMSE[i] <- sqrt(mean((predictions_denormalized - test_output)^2))
  results$MAE[i] <- mae(predictions_denormalized, test_output)
  results$MAPE[i] <- mean_absolute_percentage_error(test_output, predictions_denormalized)
  results$sMAPE[i] <- symmetric_mean_absolute_percentage_error(test_output, predictions_denormalized)
}

# Print the comparison table
print(results)

# Find the index of the best model (with the lowest RMSE)
best_model_index <- which.min(results$RMSE)

# Get the predictions of the best model
best_model <- models[[best_model_index]]$model
best_predictions <- predict(best_model, data.frame(test_input_normalized))
best_predictions_denormalized <- (best_predictions * sd(test_output)) + mean(test_output)

# Create plot data for the best model
best_plot_data <- data.frame(
  Actual = test_output,
  Predicted = best_predictions_denormalized
)

# Create a scatter plot for the best model
best_plot <- ggplot(best_plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = "#4F99DF") +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(title = paste("Best Model:", results$Description[best_model_index]), x = "Actual", y = "Predicted") +
  theme_bw()

# Print the scatter plot for the best model
print(best_plot)

