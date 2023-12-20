library(tidyverse)
library(readr)
library(tidymodels)
library(glmnet)
library(ggplot2)
set.seed(327)
# vowel_df <- read_csv("Data/VowelPivotedDataset.csv")
# original_df <- read_csv("Data/FullDataset.csv")
clean_df <- read_csv("Data/CleanDataset.csv")

clean_df <- clean_df %>%
  mutate_at(vars(vowel, syl, end, closed, rhymes), factor)

bw=30
clean_df %>% ggplot(aes(x=length, color=end)) +
  geom_histogram(color = col, fill=col, size=2) +
  geom_histogram(fill='white')

split <- initial_split(clean_df, prop = 0.9, strata = end)
train <- split %>%
        training() %>%
        mutate_at(vars(vowel, end), factor)
test <- split %>%
        testing() %>%
        mutate_at(vars(vowel, end), factor)




# Define the logistic regression model with penalty and mixture hyperparameters
log_reg <- logistic_reg(mixture = tune(), penalty = tune(), engine = "glmnet")

# Define the grid search for the hyperparameters
grid <- grid_regular(mixture(), penalty(), levels = c(mixture = 4, penalty = 3))

# Define the workflow for the model
log_reg_wf <- workflow() %>%
  add_model(log_reg) %>%
  add_formula(end ~ .)

# Define the resampling method for the grid search
folds <- vfold_cv(train, v = 5)

# Tune the hyperparameters using the grid search
log_reg_tuned <- tune_grid(
  log_reg_wf,
  resamples = folds,
  grid = grid,
  control = control_grid(save_pred = TRUE)
)

# select_best(log_reg_tuned, metric = "roc_auc")

# Train a logistic regression model
model <- logistic_reg(mixture = double(1), penalty = 0.0000000001) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(end ~ ., data = train)

# Model summary
# tidy(model)

base_model <- glm(end ~.,family=binomial(link='logit'),data=train)
summary(base_model)

base_model$coefficients
# Class Predictions
# pred_class <- predict(model,
#                       new_data = test,
#                       type = "class")
#
# # Class Probabilities
# pred_proba <- predict(model,
#                       new_data = test,
#                       type = "prob")
#
# results <- test %>%
#   select(end) %>%
#   bind_cols(pred_class, pred_proba)
#
# accuracy(results, truth = end, estimate = .pred_class)