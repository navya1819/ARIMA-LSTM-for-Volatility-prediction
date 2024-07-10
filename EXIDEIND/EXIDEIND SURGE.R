# install.packages("remotes")
remotes::install_github("rstudio/tensorflow")
reticulate::install_python()
library(tensorflow)
install_tensorflow(envname = "r-tensorflow")
install.packages("keras")
library(keras)
install_keras()
library(tensorflow)
tf$constant("Hello TensorFlow!")

install.packages("quantmod")
install.packages("tseries")
install.packages("forecast")
install.packages("xts")
install.packages("FinTS")
install.packages("rugarch")
install.packages("keras")
install.packages("tensorflow")
install.packages("randomForest")

library(forecast)
library(quantmod)
library(tseries)
library(xts)
library(FinTS)
library(rugarch)
library(keras)
library(tensorflow)
library(randomForest)

getSymbols("EXIDEIND.NS", from = "2021-01-01", to = "2024-05-01")
plot(EXIDEIND.NS$EXIDEIND.NS.Close)
exideind = (na.omit(EXIDEIND.NS))

#Stationary test for closing price
close = exideind$EXIDEIND.NS.Close
plot(close)
adf.test(close)

#Stationary test for daily log returns 
logprice = log(close)
logreturns = periodReturn(exideind$EXIDEIND.NS.Close,period = 'daily', type = 'log') 
plot(logreturns)
adf.test(logreturns)#series becomes stationary when calculating returns

#Splitting train and test data
test = as.numeric(tail(logprice,n = as.integer(length(logprice)/10)))
train = as.numeric(head(logprice, n = (as.integer(length(logprice)) - as.integer(length(logprice)/10))))

#ARMA Model testing
model2021 = auto.arima(train,d = 1, approximation = FALSE, stepwise = FALSE,trace = TRUE)
summary(model2021)
Box.test(residuals(model2021), type = c("Ljung-Box")) #p-value = 0.66

getSymbols("EXIDEIND.NS", from = "2020-01-01", to = "2024-05-01")
plot(EXIDEIND.NS$EXIDEIND.NS.Close)
exideind = (na.omit(EXIDEIND.NS))

#Stationary test for closing price
close = exideind$EXIDEIND.NS.Close
plot(close)
adf.test(close)

#Stationary test for daily log returns 
logprice = log(close)
logreturns = periodReturn(exideind$EXIDEIND.NS.Close,period = 'daily', type = 'log') 
plot(logreturns)
adf.test(logreturns)#series becomes stationary when calculating returns

#Splitting train and test data
test = as.numeric(tail(logprice,n = as.integer(length(logprice)/10)))
train = as.numeric(head(logprice, n = (as.integer(length(logprice)) - as.integer(length(logprice)/10))))

#ARMA Model testing
model2020 = auto.arima(train,d = 1, approximation = FALSE, stepwise = FALSE,trace = TRUE)
summary(model2020)
Box.test(residuals(model2020), type = c("Ljung-Box")) #p-value = 0.84

arima_residuals = residuals(model2020)
#As p-value for 2021 model is higher, we consider model from 2021 data

#Using 2021 model to predict future prices
forecast2020 = forecast(model2020, length(test))

#Calculating accuracy for the model
accuracy = accuracy(forecast2020, test)
print(accuracy)
metrics = data.frame(MODEL = "ARIMA",AIC = AIC(model2020), MAE = accuracy["Test set", "MAE"], RMSE = accuracy["Test set", "RMSE"], MASE = accuracy["Test set", "MASE"])

#In summary, the ARIMA model performs well in terms of RMSE, MAE, MAPE, and MASE, suggesting good accuracy in forecasting. However, there seems to be a positive bias in the forecasts for the test set, as indicated by the positive MPE. Additionally, the autocorrelation of the residuals (ACF1) is close to zero, indicating that the model captures the temporal patterns effectively.

qqnorm(residuals(model2020))
qqline(residuals(model2020), col = 2)
hist(residuals(model2020), breaks = 20, col = "skyblue", main = "Histogram of Residuals", xlab = "Residuals")

ArchTest(residuals(model2020)) #p-value less than 0.05 indicates ARCH effect

#Checking parameters for GARCH from 0 to 5 for lowest AIC and lowest AICc
# Loop through different GARCH orders
results = data.frame()
for (p_garch in 0:5) {
  for (q_garch in 0:5) {
    garch_fit <- tryCatch(garch(residuals(model2020), order = c(p_garch, q_garch), trace = FALSE), error = function(e) NULL)
    if (!is.null(garch_fit)) {
      log_likelihood <- logLik(garch_fit)
      num_params <- length(coef(garch_fit))
      num_obs <- length(residuals(model2020))
      aic <- -2 * log_likelihood + 2 * num_params
      aicc <- aic + (2 * num_params * (num_params + 1)) / (num_obs - num_params - 1)
      
      results <- rbind(results, data.frame(p_garch = p_garch, q_garch = q_garch, log_likelihood = log_likelihood, aic = aic, aicc = aicc))
    }
  }
}

p_garch = results[which.min(results$aic), ]$p_garch
q_garch = results[which.min(results$aic), ]$q_garch

#Found GARCH(3,2) to have the least AIC value 
garch_spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(p_garch, q_garch)),
  mean.model = list(armaOrder = c(0, 0))  # No mean model since residuals are used
)

garch_fit <- ugarchfit(spec = garch_spec, data = residuals(model2020))  #fitting the model

print(garch_fit)# Summary of the GARCH model

summary(garch_fit)
arima_garch_residuals = residuals(garch_fit)
# Calculate residuals
mae_residuals <- mean(abs(residuals(garch_fit)))
lagged_actual <- c(NA, logreturns[-length(logreturns)])# Calculate differenced series (lagged actual values)
differenced_series <- abs(logreturns - lagged_actual)
mae_differenced <- mean(differenced_series, na.rm = TRUE)
mase <- mae_residuals / mae_differenced
print(paste("MASE for Garch Model:", mase))
AIC_arima = AIC(model2020)
AIC_garch = infocriteria(garch_fit)[1]*length(train)
metrics_row = data.frame(MODEL = "ARIMA-GARCH",AIC = AIC_garch, MAE = mae_residuals, RMSE = sqrt(mean(arima_garch_residuals^2)), MASE = mase)
metrics = rbind(metrics, metrics_row)
print(AIC_garch<AIC_arima) 
#As ARIMA-GARCH model had a lower AIC than ARIMA model, we conclude that ARIMA-GARCH model are superior in capturing volatility


# Function to normalize data using Min-Max scaling
normalize <- function(data) {
  min_val <- min(data)
  max_val <- max(data)
  normalized_data <- (data - min_val) / (max_val - min_val)
  return(list(data = normalized_data, min_val = min_val, max_val = max_val))
}

create_sequences <- function(data, length) {
  X = list()
  y = list()
  for (i in seq(length, length(data) - 1)) {
    X[[length(X) + 1]] = data[(i - length + 1):i]
    y[[length(y) + 1]] = data[i + 1]
  }
  return(list(X = array(do.call(rbind, X), dim = c(length(X), length, 1)), y = unlist(y)))
}

# Define and compile the LSTM model
create_model <- function(input_shape) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = 50, return_sequences = TRUE, input_shape = input_shape) %>%
    layer_lstm(units = 50) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = 'adam'
  )
  
  return(model)
}

# Normalize train and test data
normalized_train <- normalize(train)
normalized_test <- normalize(test)

# Use normalized data to create sequences
sequence_length = 10  # Window size
train_sequences <- create_sequences(normalized_train$data, sequence_length)
test_sequences <- create_sequences(normalized_test$data, sequence_length)

# Define and compile the LSTM model
input_shape = c(sequence_length, 1)
model <- create_model(input_shape)
summary(model)

# Train the LSTM model
history <- model %>% fit(
  x = train_sequences$X,
  y = train_sequences$y,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(callback_early_stopping(monitor = 'val_loss', patience = 10)),
  verbose = 1
)

# Evaluate the model
evaluation = model %>% evaluate(
  x = test_sequences$X,
  y = test_sequences$y
)
print(paste("Test Loss:", evaluation))

# Make predictions
predictions = model %>% predict(test_sequences$X)

# Denormalize predictions and actual values
denormalize = function(data, min_val, max_val) {
  denormalized_data <- data * (max_val - min_val) + min_val
  return(denormalized_data)
}

predicted_denormalized = denormalize(predictions, normalized_test$min_val, normalized_test$max_val)
actual_denormalized = denormalize(test_sequences$y, normalized_test$min_val, normalized_test$max_val)
lstm_residuals = tail(test, n = length(predicted_denormalized)) - as.numeric(predicted_denormalized)

# Combine actual and predicted values into a data frame for plotting
results = data.frame(
  time = 1:length(test_sequences$y),
  actual = actual_denormalized,
  predicted = predicted_denormalized
)

# Plot predictions vs actual
plot(results$time, results$actual, type = "l", col = "blue", lwd = 2, main = "LSTM Predictions vs Actual", ylab = "Log Price", xlab = "Time")
lines(results$time, results$predicted, col = "red", lwd = 2)
legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)

# Calculate MAE, MSE, RMSE using denormalized values
mae = mean(abs(predicted_denormalized - actual_denormalized))
mse = mean((predicted_denormalized - actual_denormalized)^2)
rmse = sqrt(mse)
#MASE for LSTM
mean_abs_residual = mean(abs(predicted_denormalized - actual_denormalized))
naive_forecast = actual_denormalized[length(actual_denormalized)]
naive_residuals = actual_denormalized[2:length(actual_denormalized)] - naive_forecast
scale = mean(abs(naive_residuals))
mase = mean_abs_residual / scale

metrics_row = data.frame(MODEL = "LSTM",AIC = "", MAE = mae, RMSE = rmse, MASE = mase)
metrics = rbind(metrics, metrics_row)

print(paste("MAE for LSTM:", mae))
print(paste("MSE for LSTM:", mse))
print(paste("RMSE for LSTM:", rmse))
print(paste("MASE for LSTM:", mase))

#ARIMA LSTM Model using ACF of squared residuals
sq_residuals = (residuals(model2020))^2
acf(sq_residuals, lag.max = 15)
#Max correlation observed for lag of 15 days

# Use normalized data to create sequences
sequence_length = 9  # Window size
train_sequences = create_sequences(normalized_train$data, sequence_length)
test_sequences = create_sequences(normalized_test$data, sequence_length)

# Define and compile the LSTM model
input_shape = c(sequence_length, 1)
model = create_model(input_shape)
summary(model)

# Train the LSTM model
history = model %>% fit(
  x = train_sequences$X,
  y = train_sequences$y,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(callback_early_stopping(monitor = 'val_loss', patience = 10)),
  verbose = 1
)

# Evaluate the model
evaluation = model %>% evaluate(
  x = test_sequences$X,
  y = test_sequences$y
)
print(paste("Test Loss:", evaluation))

# Make predictions
predictions = model %>% predict(test_sequences$X)

# Denormalize predictions and actual values
denormalize = function(data, min_val, max_val) {
  denormalized_data <- data * (max_val - min_val) + min_val
  return(denormalized_data)
}

predicted_denormalized = denormalize(predictions, normalized_test$min_val, normalized_test$max_val)
actual_denormalized = denormalize(test_sequences$y, normalized_test$min_val, normalized_test$max_val)
lstm_acf_residuals = tail(test, n = length(predicted_denormalized)) - as.numeric(predicted_denormalized)

# Combine actual and predicted values into a data frame for plotting
results = data.frame(
  time = 1:length(test_sequences$y),
  actual = actual_denormalized,
  predicted = predicted_denormalized
)

# Plot predictions vs actual
plot(results$time, results$actual, type = "l", col = "blue", lwd = 2, main = "LSTM(ACF) Predictions vs Actual", ylab = "Log Price", xlab = "Time")
lines(results$time, results$predicted, col = "red", lwd = 2)
legend("topright", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)

# Calculate MAE, MSE, RMSE using denormalized values
mae <- mean(abs(predicted_denormalized - actual_denormalized))
mse <- mean((predicted_denormalized - actual_denormalized)^2)
rmse <- sqrt(mse)
#MASE for LSTM
mean_abs_residual <- mean(abs(predicted_denormalized - actual_denormalized))
naive_forecast <- actual_denormalized[length(actual_denormalized)]
naive_residuals <- actual_denormalized[2:length(actual_denormalized)] - naive_forecast
scale <- mean(abs(naive_residuals))
mase <- mean_abs_residual / scale

metrics_row = data.frame(MODEL = "LSTM using ACF",AIC = "", MAE = mae, RMSE = rmse, MASE = mase)
metrics = rbind(metrics, metrics_row)

print(paste("MAE for LSTM using ACF:", mae))
print(paste("MSE for LSTM using ACF:", mse))
print(paste("RMSE for LSTM using ACF:", rmse))
print(paste("MASE for LSTM using ACF:", mase))

#ARIMA LSTM Model using Random Forest Technique
# Function to create lagged data frame
create_lagged_df <- function(series, max_lag) {
  lagged_df <- data.frame(matrix(ncol = max_lag + 1, nrow = length(series)))
  colnames(lagged_df) <- c(paste0("lag_", 1:max_lag), "t")
  
  for (i in 1:max_lag) {
    lagged_df[, i] <- c(rep(NA, i), series[1:(length(series) - i)])
  }
  lagged_df$t <- series
  lagged_df <- na.omit(lagged_df)
  
  return(lagged_df)
}

# Create lagged data frame with maximum lag of 12
max_lag <- 15
lagged_df <- create_lagged_df(sq_residuals, max_lag)

# Separate features (X) and target (y)
X <- lagged_df[, 1:max_lag]
y <- lagged_df$t

# Train the Random Forest model
rf_model <- randomForest(X, y, ntree = 100, random_state = 1)

# Get the feature importances
importance_scores <- importance(rf_model)
feature_names <- colnames(X)

# Plot feature importance using base R plot function
plot(importance_scores, type = "h", lwd = 2, col = "skyblue", xlab = "Feature", ylab = "Importance")
axis(side = 1, at = 1:length(importance_scores), labels = feature_names, las = 2)
title(main = "Feature Importance for Lagged Residuals")

#Maximum Lag correlation is observed at lag 8
# Normalize train and test data
normalized_train <- normalize(train)
normalized_test <- normalize(test)

# Use normalized data to create sequences
sequence_length = 9  # Window size
train_sequences <- create_sequences(normalized_train$data, sequence_length)
test_sequences <- create_sequences(normalized_test$data, sequence_length)

# Define and compile the LSTM model
input_shape = c(sequence_length, 1)
model <- create_model(input_shape)
summary(model)

# Train the LSTM model
history <- model %>% fit(
  x = train_sequences$X,
  y = train_sequences$y,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(callback_early_stopping(monitor = 'val_loss', patience = 10)),
  verbose = 1
)

# Evaluate the model
evaluation <- model %>% evaluate(
  x = test_sequences$X,
  y = test_sequences$y
)
print(paste("Test Loss:", evaluation))

# Make predictions
predictions <- model %>% predict(test_sequences$X)

# Denormalize predictions and actual values
denormalize <- function(data, min_val, max_val) {
  denormalized_data <- data * (max_val - min_val) + min_val
  return(denormalized_data)
}

predicted_denormalized <- denormalize(predictions, normalized_test$min_val, normalized_test$max_val)
actual_denormalized <- denormalize(test_sequences$y, normalized_test$min_val, normalized_test$max_val)
lstm_rf_residuals = tail(test, n = length(predicted_denormalized)) - as.numeric(predicted_denormalized)

# Combine actual and predicted values into a data frame for plotting
results <- data.frame(
  time = 1:length(test_sequences$y),
  actual = actual_denormalized,
  predicted = predicted_denormalized
)

# Plot predictions vs actual
plot(results$time, results$actual, type = "l", col = "blue", lwd = 2, main = "LSTM(Random Forest) Predictions vs Actual", ylab = "Log Price", xlab = "Time")
lines(results$time, results$predicted, col = "red", lwd = 2)
legend("topleft", legend = c("Actual", "Predicted"), col = c("blue", "red"), lty = 1, lwd = 2)

# Calculate MAE, MSE, RMSE using denormalized values
mae <- mean(abs(predicted_denormalized - actual_denormalized))
mse <- mean((predicted_denormalized - actual_denormalized)^2)
rmse <- sqrt(mse)
#MASE for LSTM
mean_abs_residual <- mean(abs(predicted_denormalized - actual_denormalized))
naive_forecast <- actual_denormalized[length(actual_denormalized)]
naive_residuals <- actual_denormalized[2:length(actual_denormalized)] - naive_forecast
scale <- mean(abs(naive_residuals))
mase <- mean_abs_residual / scale

metrics_row = data.frame(MODEL = "LSTM using Random Forest",AIC = "", MAE = mae, RMSE = rmse, MASE = mase)
metrics = rbind(metrics, metrics_row)

print(paste("MAE for LSTM using Random Forest:", mae))
print(paste("MSE for LSTM using Random Forest:", mse))
print(paste("RMSE for LSTM using Random Forest:", rmse))
print(paste("MASE for LSTM using Random Forest:", mase))

dm_test_result <- dm.test(arima_residuals, lstm_residuals, alternative = "two.sided", h = 1)
dm_test_log <- data.frame(
  Model_1 = "ARIMA",
  Model_2 = "LSTM",
  DM_Statistic = dm_test_result$statistic,
  p_value = dm_test_result$p.value,
  stringsAsFactors = FALSE
)

dm_test_result <- dm.test(arima_garch_residuals, lstm_residuals, alternative = "two.sided", h = 1)
dm_row <- data.frame(
  Model_1 = "ARIMA-GARCH",
  Model_2 = "LSTM",
  DM_Statistic = dm_test_result$statistic,
  p_value = dm_test_result$p.value,
  stringsAsFactors = FALSE
)
dm_test_log = rbind(dm_test_log, dm_row)

dm_test_result <- dm.test(arima_garch_residuals, arima_residuals, alternative = "two.sided", h = 1)
dm_row <- data.frame(
  Model_1 = "ARIMA-GARCH",
  Model_2 = "ARIMA",
  DM_Statistic = dm_test_result$statistic,
  p_value = dm_test_result$p.value,
  stringsAsFactors = FALSE
)
dm_test_log = rbind(dm_test_log, dm_row)

dm_test_result <- dm.test(lstm_residuals, lstm_rf_residuals, alternative = "two.sided", h = 1)
dm_row <- data.frame(
  Model_1 = "LSTM",
  Model_2 = "ARIMA-LSTM(RANDOM FOREST)",
  DM_Statistic = dm_test_result$statistic,
  p_value = dm_test_result$p.value,
  stringsAsFactors = FALSE
)
dm_test_log = rbind(dm_test_log, dm_row)

dm_test_result <- dm.test(arima_garch_residuals, lstm_rf_residuals, alternative = "two.sided", h = 1)
dm_row <- data.frame(
  Model_1 = "ARIMA-GARCH",
  Model_2 = "ARIMA-LSTM(RANDOM FOREST)",
  DM_Statistic = dm_test_result$statistic,
  p_value = dm_test_result$p.value,
  stringsAsFactors = FALSE
)
dm_test_log = rbind(dm_test_log, dm_row)

print(dm_test_log)

