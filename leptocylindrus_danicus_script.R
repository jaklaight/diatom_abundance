library(tidyverse)
library(tidymodels)
library(vip)
library(pdp)
library(patchwork)

############################# Data preparation ##############################

data <- read.csv("data.csv")

leptocylindrus_danicus <- data |> 
  select("Temp","NO3","dFe","PAR","P_s","Si_s","Leptocylindrus.danicus") |> 
  rename(abundance = Leptocylindrus.danicus) |> 
  drop_na()

#############################################################################-
########################## Classification problem ###########################
#############################################################################-

######################### Spending our data budget ##########################

leptocylindrus_danicus$abundance_factor <- 
  as.factor(
    ifelse(
      leptocylindrus_danicus$abundance > 0, 1, 0
    )
  )

set.seed(123)
leptocylindrus_danicus_classification_split <- 
  initial_split(
    leptocylindrus_danicus, 
    strata = abundance_factor, 
    prop = 0.80
  )

leptocylindrus_danicus_train_classification <- 
  training(leptocylindrus_danicus_classification_split)

leptocylindrus_danicus_test_classification <- 
  testing(leptocylindrus_danicus_classification_split)

# validation folds

set.seed(234)
leptocylindrus_danicus_classification_folds <- 
  vfold_cv(
    leptocylindrus_danicus_train_classification,
    v = 5, 
    repeats = 3, 
    strata = abundance_factor
  )

########################## Pre-processing recipes ###########################

base_classification_rec <-
  recipe(
    abundance_factor ~ Temp + NO3 + dFe + PAR + P_s + Si_s,
    data = leptocylindrus_danicus_train_classification
  )
# for models which do not require centering or scaling

norm_classification_rec <-
  recipe(
    abundance_factor ~ Temp + NO3 + dFe + PAR + P_s + Si_s,
    data = leptocylindrus_danicus_train_classification
  ) |> 
  step_normalize(
    all_numeric_predictors()
  ) 
# for models which DO require centering and scaling

########################### Model specifications ############################

# Random Forest

rf_classification_spec <- 
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = 2000
  ) |> 
  set_engine("ranger",
             importance = "impurity") |> 
  set_mode("classification")

# Boosted Trees

xgb_classification_spec <-
  boost_tree(
    trees = 2000,
    tree_depth = tune(),
    mtry = tune(),
    learn_rate = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    stop_iter = tune()
  ) |> 
  set_engine("xgboost") |>
  set_mode("classification")

# Artificial Neural Network

ann_classification_hidden_layers_spec <-
  mlp(
    hidden_units = tune(), 
    activation = "relu"
  ) |>
  set_engine("brulee") |>
  set_mode("classification")

ann_classification_spec <-
  mlp(
    hidden_units = 7, 
    activation = "relu",
    dropout = tune(),
    epochs = tune(),
    learn_rate = tune(),
  ) |>
  set_engine("brulee") |>
  set_mode("classification")

# K-Nearest Neighbour

knn_classification_spec <-
  nearest_neighbor(
    neighbors = tune(), 
    weight_func = "gaussian", 
    dist_power = tune()
  ) |>
  set_engine("kknn") |>
  set_mode("classification")

################################ Workflows ##################################

# Random Forest

rf_classification_workflow  <- workflow(base_classification_rec, rf_classification_spec)

# Boosted Trees

xgb_classification_workflow <- workflow(base_classification_rec, xgb_classification_spec)

# Artificial Neural Network

ann_classification_hidden_layers_workflow <- workflow(norm_classification_rec, ann_classification_hidden_layers_spec)

ann_classification_workflow <- workflow(norm_classification_rec, ann_classification_spec)

# K-Nearest Neighbour

knn_classification_workflow <- workflow(norm_classification_rec, knn_classification_spec)

################################## Tuning ###################################

doParallel::registerDoParallel()

# Random Forest

rf_classification_grid <- grid_latin_hypercube(   # using a space-filling design
  min_n(),
  finalize(mtry(), leptocylindrus_danicus_train_classification[,1:6]), # we have to treat mtry differently since it relies on the number of features in the data
  size = 30
)

rf_classification_grid

set.seed(345)
rf_classification_tuning <- tune_grid(
  rf_classification_workflow,
  resamples = leptocylindrus_danicus_classification_folds,
  grid = rf_classification_grid,
  control = control_grid(save_pred = TRUE)
)

rf_classification_tuning

autoplot(rf_classification_tuning) +
  theme_bw()

show_best(rf_classification_tuning, "roc_auc")
show_best(rf_classification_tuning, "accuracy")

rf_classification_tuned_spec <- 
  rand_forest(
    mtry = 1,
    min_n = 32,
    trees = 2000,
  ) |> 
  set_engine("ranger",
             importance = "permutation") |> 
  set_mode("classification")

# Boosted Trees

xgb_classification_grid <- grid_latin_hypercube(   
  tree_depth(),
  finalize(mtry(), leptocylindrus_danicus_train_classification[,1:6]),
  learn_rate(), 
  min_n(), 
  loss_reduction(), 
  sample_size = sample_prop(), 
  stop_iter(), 
  size = 30
)

xgb_classification_grid

set.seed(345)
xgb_classification_tuning <- tune_grid(
  xgb_classification_workflow,
  resamples = leptocylindrus_danicus_classification_folds,
  grid = xgb_classification_grid,
  control = control_grid(save_pred = TRUE)
)

xgb_classification_tuning

autoplot(xgb_classification_tuning) +
  theme_bw()

show_best(xgb_classification_tuning, "roc_auc")
show_best(xgb_classification_tuning, "accuracy")

xgb_classification_tuned_spec <- 
  boost_tree(
    trees = 2000,
    tree_depth = 12, 
    mtry = 3, 
    learn_rate = 0.003, 
    min_n = 20, 
    loss_reduction = 0.00002, 
    sample_size = 0.4, 
    stop_iter = 12
  ) |> 
  set_engine("xgboost") |>
  set_mode("classification")

# Artificial Neural Network

ann_classification_hidden_layers_grid <- grid_latin_hypercube(   
  hidden_units(), 
  size = 30
)

ann_classification_hidden_layers_grid

set.seed(345)
ann_classification_hidden_layers_tuning <- tune_grid(
  ann_classification_hidden_layers_workflow,
  resamples = leptocylindrus_danicus_classification_folds,
  grid = ann_classification_hidden_layers_grid,
  control = control_grid(save_pred = TRUE)
)

ann_classification_hidden_layers_tuning

autoplot(ann_classification_hidden_layers_tuning) +
  theme_bw()

show_best(ann_classification_hidden_layers_tuning, "roc_auc")
show_best(ann_classification_hidden_layers_tuning, "accuracy")
# choose best value of hidden_units and go back to model specs


ann_classification_grid <- grid_latin_hypercube(   
  dropout(),
  epochs(),
  learn_rate(),
  size = 30
)

ann_classification_grid

set.seed(345)
ann_classification_tuning <- tune_grid(
  ann_classification_workflow,
  resamples = leptocylindrus_danicus_classification_folds,
  grid = ann_classification_grid,
  control = control_grid(save_pred = TRUE)
)

ann_classification_tuning

autoplot(ann_classification_tuning) +
  theme_bw()

show_best(ann_classification_tuning, "roc_auc")
show_best(ann_classification_tuning, "accuracy")

ann_classification_tuned_spec <- 
  mlp(
    hidden_units = 10, 
    dropout = 0.4,
    epochs = 1000,
    learn_rate = 0.01,
    activation = "relu"
  ) |>
  set_engine("brulee") |>
  set_mode("classification")

# K-Nearest Neighbour

knn_classification_grid <- grid_latin_hypercube(   
  neighbors(), 
  dist_power(),
  size = 30
)

knn_classification_grid

set.seed(345)
knn_classification_tuning <- tune_grid(
  knn_classification_workflow,
  resamples = leptocylindrus_danicus_classification_folds,
  grid = knn_classification_grid,
  control = control_grid(save_pred = TRUE)
)

knn_classification_tuning

autoplot(knn_classification_tuning) +
  theme_bw()

show_best(knn_classification_tuning, "roc_auc")
show_best(knn_classification_tuning, "accuracy")

knn_classification_tuned_spec <- 
  nearest_neighbor(
    neighbors = 10, 
    weight_func = "gaussian", 
    dist_power = 1.5
  ) |>
  set_engine("kknn") |>
  set_mode("classification")

############################ Tuned workflow set #############################

leptocylindrus_danicus_classification_set <-
  workflow_set(
    list(
      base_classification_rec, 
      norm_classification_rec
    ),
    list(
      rf_classification_tuned_spec, 
      xgb_classification_tuned_spec,
      ann_classification_tuned_spec, 
      knn_classification_tuned_spec
    )
  )

leptocylindrus_danicus_classification_set <-
  leptocylindrus_danicus_classification_set |> 
  filter(
    wflow_id %in% c(
      "recipe_1_rand_forest",
      "recipe_1_boost_tree",
      "recipe_2_mlp",
      "recipe_2_nearest_neighbor"
    )
  )

leptocylindrus_danicus_classification_set

set.seed(345)
leptocylindrus_danicus_classification_rs <-
  workflow_map(
    leptocylindrus_danicus_classification_set,
    "fit_resamples",
    resamples = leptocylindrus_danicus_classification_folds
  )

leptocylindrus_danicus_classification_results_table_auc <-
  leptocylindrus_danicus_classification_rs |> 
  rank_results() |> 
  filter(.metric == "roc_auc") |> 
  select(model, AUC = mean, std_err)

leptocylindrus_danicus_classification_results_table_acc <-
  leptocylindrus_danicus_classification_rs |> 
  rank_results() |> 
  filter(.metric == "accuracy") |> 
  select(model, accuracy = mean, std_err)

leptocylindrus_danicus_classification_rs |> 
  autoplot() +
  theme_bw()

############################## Fit the model ################################

leptocylindrus_danicus_classification_final_wf <-
  workflow(
    base_classification_rec, 
    rf_classification_tuned_spec
  )

leptocylindrus_danicus_classification_fit <- 
  fit(
    leptocylindrus_danicus_classification_final_wf,
    leptocylindrus_danicus_train_classification
  )

leptocylindrus_danicus_classification_last_fit <- 
  last_fit(
    leptocylindrus_danicus_classification_fit,
    leptocylindrus_danicus_classification_split
  )

collect_metrics(leptocylindrus_danicus_classification_last_fit)

#############################################################################-
############################ Regression problem #############################
#############################################################################-

######################### Spending our data budget ##########################

# filter out zeroes

leptocylindrus_danicus_regression <- 
  leptocylindrus_danicus |> filter(abundance > 0)

# log-transform the (non-zero) abundances

leptocylindrus_danicus_regression$log_abundance <-
  log(leptocylindrus_danicus_regression$abundance)

# drop factor

leptocylindrus_danicus_regression <- 
  leptocylindrus_danicus_regression |> 
  mutate(
    abundance_factor = NULL
  )

set.seed(123)
leptocylindrus_danicus_regression_split <- 
  initial_split(
    leptocylindrus_danicus_regression, 
    strata = log_abundance, 
    prop = 0.80
  )

leptocylindrus_danicus_train_regression <- 
  training(leptocylindrus_danicus_regression_split)

leptocylindrus_danicus_test_regression <- 
  testing(leptocylindrus_danicus_regression_split)

# validation folds

set.seed(234)
leptocylindrus_danicus_regression_folds <- 
  vfold_cv(
    leptocylindrus_danicus_train_regression,
    v = 5, 
    repeats = 3, 
    strata = log_abundance
  )

########################## Pre-processing recipes ###########################

base_regression_rec <-
  recipe(
    log_abundance ~ Temp + NO3 + dFe + PAR + P_s + Si_s,
    data = leptocylindrus_danicus_train_regression
  )
# for models which do not require centering or scaling

norm_regression_rec <-
  recipe(
    log_abundance ~ Temp + NO3 + dFe + PAR + P_s + Si_s,
    data = leptocylindrus_danicus_train_regression
  ) |> 
  step_normalize(
    all_numeric_predictors()
  ) 
# for models which DO require centering and scaling

########################### Model specifications ############################

# Random Forest

rf_regression_spec <- 
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = 2000
  ) |> 
  set_engine("ranger",  
             importance = "impurity") |> 
  set_mode("regression")

# Boosted Trees

xgb_regression_spec <-
  boost_tree(
    trees = 2000,
    tree_depth = tune(),
    mtry = tune(),
    learn_rate = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    stop_iter = tune()
  ) |> 
  set_engine("xgboost") |>
  set_mode("regression")

# Artificial Neural Network

ann_regression_hidden_layers_spec <-
  mlp(
    hidden_units = tune(), 
    activation = "relu"
  ) |>
  set_engine("brulee") |>
  set_mode("regression")

ann_regression_spec <-
  mlp(
    hidden_units = 10, 
    activation = "relu",
    dropout = tune(),
    epochs = tune(),
    learn_rate = tune(),
  ) |>
  set_engine("brulee") |>
  set_mode("regression")

# K-Nearest Neighbour

knn_regression_spec <-
  nearest_neighbor(
    neighbors = tune(), 
    weight_func = "gaussian", 
    dist_power = tune()
  ) |>
  set_engine("kknn") |>
  set_mode("regression")

################################ Workflows ##################################

# Random Forest

rf_regression_workflow  <- workflow(base_regression_rec, rf_regression_spec)

# Boosted Trees

xgb_regression_workflow <- workflow(base_regression_rec, xgb_regression_spec)

# Artificial Neural Network

ann_regression_hidden_layers_workflow <- workflow(norm_regression_rec, ann_regression_hidden_layers_spec)

ann_regression_workflow <- workflow(norm_regression_rec, ann_regression_spec)

# K-Nearest Neighbour

knn_regression_workflow <- workflow(norm_regression_rec, knn_regression_spec)

################################## Tuning ###################################

doParallel::registerDoParallel()

# Random Forest

rf_regression_grid <- grid_latin_hypercube(  
  min_n(),
  finalize(mtry(), leptocylindrus_danicus_train_regression[,1:6]),
  size = 30
)

rf_regression_grid

set.seed(345)
rf_regression_tuning <- tune_grid(
  rf_regression_workflow,
  resamples = leptocylindrus_danicus_regression_folds,
  grid = rf_regression_grid,
  control = control_grid(save_pred = TRUE)
)

rf_regression_tuning

autoplot(rf_regression_tuning) +
  theme_bw()

show_best(rf_regression_tuning, "rsq")
show_best(rf_regression_tuning, "rmse")

rf_regression_tuned_spec <- 
  rand_forest(
    mtry = 1,
    min_n = 15,
    trees = 2000
  ) |> 
  set_engine("ranger",
             importance = "permutation") |> 
  set_mode("regression")

# Boosted Trees

xgb_regression_grid <- grid_latin_hypercube(   
  tree_depth(), 
  finalize(mtry(), leptocylindrus_danicus_train_regression[,1:6]), 
  learn_rate(), 
  min_n(), 
  loss_reduction(), 
  sample_size = sample_prop(), 
  stop_iter(), 
  size = 30
)

xgb_regression_grid

set.seed(345)
xgb_regression_tuning <- tune_grid(
  xgb_regression_workflow,
  resamples = leptocylindrus_danicus_regression_folds,
  grid = xgb_regression_grid,
  control = control_grid(save_pred = TRUE)
)

xgb_regression_tuning

autoplot(xgb_regression_tuning) +
  theme_bw()

show_best(xgb_regression_tuning, "rsq") 
show_best(xgb_regression_tuning, "rmse")

xgb_regression_tuned_spec <- 
  boost_tree(
    tree_depth = 14, 
    trees = 2000, 
    mtry = 5,
    learn_rate = 0.003, 
    min_n = 15, 
    loss_reduction = 0.0002, 
    sample_size = 0.4, 
    stop_iter = 18
  ) |> 
  set_engine("xgboost") |>
  set_mode("regression")

# Artificial Neural Network

ann_regression_hidden_layers_grid <- grid_latin_hypercube(   
  hidden_units(), 
  size = 30
)

ann_regression_hidden_layers_grid

set.seed(345)
ann_regression_hidden_layers_tuning <- tune_grid(
  ann_regression_hidden_layers_workflow,
  resamples = leptocylindrus_danicus_regression_folds,
  grid = ann_regression_hidden_layers_grid,
  control = control_grid(save_pred = TRUE)
)

ann_regression_hidden_layers_tuning

autoplot(ann_regression_hidden_layers_tuning) +
  theme_bw()

show_best(ann_regression_hidden_layers_tuning, "rsq")
show_best(ann_regression_hidden_layers_tuning, "rmse")
# choose best value of hidden_units and go back to model specs

ann_regression_grid <- grid_latin_hypercube(   
  dropout(),
  epochs(),
  learn_rate(),
  size = 30
)

ann_regression_grid

set.seed(345)
ann_regression_tuning <- tune_grid(
  ann_regression_workflow,
  resamples = leptocylindrus_danicus_regression_folds,
  grid = ann_regression_grid,
  control = control_grid(save_pred = TRUE)
)

ann_regression_tuning

autoplot(ann_regression_tuning) +
  theme_bw()

show_best(ann_regression_tuning, "rsq")
show_best(ann_regression_tuning, "rmse")

ann_regression_tuned_spec <- 
  mlp(
    hidden_units = 10, 
    dropout = 0.4,
    epochs = 1000, 
    learn_rate = 0.01, 
    activation = "relu"
  ) |>
  set_engine("brulee") |>
  set_mode("regression")

# K-Nearest Neighbour

knn_regression_grid <- grid_latin_hypercube(   
  neighbors(), 
  dist_power(),
  size = 30
)

knn_regression_grid

set.seed(345)
knn_regression_tuning <- tune_grid(
  knn_regression_workflow,
  resamples = leptocylindrus_danicus_regression_folds,
  grid = knn_regression_grid,
  control = control_grid(save_pred = TRUE)
)

knn_regression_tuning

autoplot(knn_regression_tuning) +
  theme_bw()

show_best(knn_regression_tuning, "rsq")
show_best(knn_regression_tuning, "rmse")

knn_regression_tuned_spec <- 
  nearest_neighbor(
    neighbors = 10, 
    weight_func = "gaussian", 
    dist_power = 1.5
  ) |>
  set_engine("kknn") |>
  set_mode("regression")

############################ Tuned workflow set #############################

leptocylindrus_danicus_regression_set <-
  workflow_set(
    list(
      base_regression_rec, 
      norm_regression_rec
    ),
    list(
      rf_regression_tuned_spec, 
      xgb_regression_tuned_spec,
      ann_regression_tuned_spec, 
      knn_regression_tuned_spec
    )
  )

leptocylindrus_danicus_regression_set <-
  leptocylindrus_danicus_regression_set |> 
  filter(
    wflow_id %in% c(
      "recipe_1_rand_forest",
      "recipe_1_boost_tree",
      "recipe_2_mlp",
      "recipe_2_nearest_neighbor"
    )
  )

leptocylindrus_danicus_regression_set

set.seed(345)
leptocylindrus_danicus_regression_rs <-
  workflow_map(
    leptocylindrus_danicus_regression_set,
    "fit_resamples",
    resamples = leptocylindrus_danicus_regression_folds
  )

leptocylindrus_danicus_regression_results_table_rsq <-
  leptocylindrus_danicus_regression_rs |> 
  rank_results() |> 
  filter(.metric == "rsq") |> 
  select(model, "$R^2$" = mean, std_err)

leptocylindrus_danicus_regression_results_table_rmse <-
  leptocylindrus_danicus_regression_rs |> 
  rank_results() |> 
  filter(.metric == "rmse") |> 
  select(model, RMSE = mean, std_err)

leptocylindrus_danicus_regression_rs |> 
  autoplot() +
  theme_bw()

############################## Fit the model ################################

leptocylindrus_danicus_regression_final_wf <-
  workflow(
    base_regression_rec, 
    rf_regression_tuned_spec
  )

leptocylindrus_danicus_regression_fit <- 
  fit(
    leptocylindrus_danicus_regression_final_wf,
    leptocylindrus_danicus_train_regression
  )

leptocylindrus_danicus_regression_last_fit <- 
  last_fit(
    leptocylindrus_danicus_regression_fit,
    leptocylindrus_danicus_regression_split
  )

collect_metrics(leptocylindrus_danicus_regression_last_fit)

#############################################################################-
######################### Fitting the hurdle model ##########################
#############################################################################-

leptocylindrus_danicus_classification_predictions <- 
  predict(
    leptocylindrus_danicus_classification_fit, 
    leptocylindrus_danicus_test_classification
  )

leptocylindrus_danicus_regression_predictions <- 
  predict(
    leptocylindrus_danicus_regression_fit,
    leptocylindrus_danicus_test_classification
  )

# convert predicted classification values to numeric
leptocylindrus_danicus_classification_predictions$.pred_class <- 
  as.numeric(
    as.character(
      leptocylindrus_danicus_classification_predictions$.pred_class
    )
  )

# combine the predictions based on the hurdle model
leptocylindrus_danicus_hurdle_predictions <- 
  ifelse(
    leptocylindrus_danicus_classification_predictions$.pred_class == 0, 0, 
    exp(leptocylindrus_danicus_regression_predictions$.pred)
  ) 

#############################################################################-
################### Collect metrics for the hurdle model ####################
#############################################################################-

leptocylindrus_danicus_metrics_bootstrap <- 
  bootstraps(
    leptocylindrus_danicus_test_classification, 
    times = 500, 
    strata = abundance
  )

leptocylindrus_danicus_classification_predictions_bootstrap <- c()
leptocylindrus_danicus_regression_predictions_bootstrap <- c()

# Iterate over each bootstrap sample
for (i in 1:length(leptocylindrus_danicus_metrics_bootstrap$id)) {
  
  # Get the current bootstrap sample
  bootstrap_sample <- 
    leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$data[
      leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$in_id, 
    ]
  
  # Make predictions on the bootstrap sample
  classification_predictions <- 
    predict(
      leptocylindrus_danicus_classification_fit, 
      bootstrap_sample
    )
  
  regression_predictions <- 
    predict(
      leptocylindrus_danicus_regression_fit, 
      bootstrap_sample
    )
  
  # Convert classification values to numeric
  classification_predictions$.pred_class <- 
    as.numeric(
      as.character(
        classification_predictions$.pred_class
      )
    )
  
  # Store the bootstrap predictions
  leptocylindrus_danicus_classification_predictions_bootstrap[[i]] <- 
    classification_predictions
  
  leptocylindrus_danicus_regression_predictions_bootstrap[[i]] <- 
    regression_predictions
  
}

leptocylindrus_danicus_hurdle_predictions_bootstrap <- c()

# Iterate over each bootstrap sample
for (i in 1:length(leptocylindrus_danicus_metrics_bootstrap$id)) {
  
  # Get the classification and regression predictions for the current bootstrap sample
  classification_predictions <- 
    leptocylindrus_danicus_classification_predictions_bootstrap[[i]]
  
  regression_predictions <- 
    leptocylindrus_danicus_regression_predictions_bootstrap[[i]]
  
  # Combine the predictions based on the hurdle model
  hurdle_predictions <- ifelse(
    classification_predictions$.pred_class == 0, 0, 
    regression_predictions$.pred
  )
  
  # Store the hurdle predictions
  leptocylindrus_danicus_hurdle_predictions_bootstrap[[i]] <- 
    hurdle_predictions
}

# Log-transform non-zero abundance

for (i in 1:length(leptocylindrus_danicus_metrics_bootstrap$id)) {
  leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$data[
    leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$in_id, 7
  ] <-
    ifelse(
      leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$data[
        leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$in_id, 7
      ] == 0, 0,
      log(
        leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$data[
          leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$in_id, 7
        ]
      )
    )
  
}

# calculate R-squared (R2)

leptocylindrus_danicus_rsq_boot <- c()

for (i in 1:length(leptocylindrus_danicus_metrics_bootstrap$id)) {
  leptocylindrus_danicus_rsq_boot[i] <- 
    cor(
      leptocylindrus_danicus_hurdle_predictions_bootstrap[[i]], 
      leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$data[
        leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$in_id, 7
      ]
    )^2 
}

# calculate Mean Absolute Error (MAE)

leptocylindrus_danicus_mae_boot <- c()

for (i in 1:length(leptocylindrus_danicus_metrics_bootstrap$id)) {
  leptocylindrus_danicus_mae_boot[i] <- 
    mean(
      abs(
        leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$data[
          leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$in_id, 7
        ] - leptocylindrus_danicus_hurdle_predictions_bootstrap[[i]]
      )
    )
}

# calculate RMSE

leptocylindrus_danicus_rmse_boot <- c()

for (i in 1:length(leptocylindrus_danicus_metrics_bootstrap$id)) {
  leptocylindrus_danicus_rmse_boot[i] <- 
    sqrt(
      mean(
        (leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$data[
          leptocylindrus_danicus_metrics_bootstrap$splits[[i]]$in_id, 7
        ] - leptocylindrus_danicus_hurdle_predictions_bootstrap[[i]])^2
      )
    )
}

leptocylindrus_danicus_hurdle_metrics <- 
  tibble(
    "$R^2$ (SE)" = paste0(
      round(mean(leptocylindrus_danicus_rsq_boot),3),
      " (",
      round(sd(leptocylindrus_danicus_rsq_boot),3),
      ")"
    ),
    "RMSE (SE)" = paste0(
      round(mean(leptocylindrus_danicus_rmse_boot),3),
      " (",
      round(sd(leptocylindrus_danicus_rmse_boot),3),
      ")"
    ),
    "MAE (SE)" = paste0(
      round(mean(leptocylindrus_danicus_mae_boot),3),
      " (",
      round(sd(leptocylindrus_danicus_mae_boot),3),
      ")"
    )
  )

#############################################################################-
############################ Variable importance ############################
#############################################################################-

# classification vip:

leptocylindrus_danicus_classification_vip <-
  leptocylindrus_danicus_classification_fit |>  
  extract_fit_parsnip() |> 
  vip(geom = "point", metric = "roc_auc") + 
  labs(title = "Variable importance") +
  theme_bw()

# classification pdp:

leptocylindrus_danicus_classification_pdp_temp <-
  leptocylindrus_danicus_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "Temp",
    train    = leptocylindrus_danicus_train_classification
  ) 

leptocylindrus_danicus_classification_pdp_no3 <-
  leptocylindrus_danicus_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "NO3",
    train    = leptocylindrus_danicus_train_classification
  ) 

leptocylindrus_danicus_classification_pdp_dfe <-
  leptocylindrus_danicus_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "dFe",
    train    = leptocylindrus_danicus_train_classification
  ) 

leptocylindrus_danicus_classification_pdp_par <-
  leptocylindrus_danicus_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "PAR",
    train    = leptocylindrus_danicus_train_classification
  ) 

leptocylindrus_danicus_classification_pdp_p_s <-
  leptocylindrus_danicus_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "P_s",
    train    = leptocylindrus_danicus_train_classification
  ) 

leptocylindrus_danicus_classification_pdp_si_s <-
  leptocylindrus_danicus_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "Si_s",
    train    = leptocylindrus_danicus_train_classification
  ) 

# regression vip:

leptocylindrus_danicus_regression_vip <-
  leptocylindrus_danicus_regression_fit |>  
  extract_fit_parsnip() |> 
  vip(geom = "point", metric = "rmse") +
  labs(title = "Variable importance") +
  theme_bw()

# regression pdp:

leptocylindrus_danicus_regression_pdp_temp <-
  leptocylindrus_danicus_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "Temp",
    train    = leptocylindrus_danicus_train_regression
  )

leptocylindrus_danicus_regression_pdp_no3 <-
  leptocylindrus_danicus_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "NO3",
    train    = leptocylindrus_danicus_train_regression
  ) 

leptocylindrus_danicus_regression_pdp_dfe <-
  leptocylindrus_danicus_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "dFe",
    train    = leptocylindrus_danicus_train_regression
  ) 

leptocylindrus_danicus_regression_pdp_par <-
  leptocylindrus_danicus_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "PAR",
    train    = leptocylindrus_danicus_train_regression
  ) 

leptocylindrus_danicus_regression_pdp_p_s <-
  leptocylindrus_danicus_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "P_s",
    train    = leptocylindrus_danicus_train_regression
  ) 

leptocylindrus_danicus_regression_pdp_si_s <-
  leptocylindrus_danicus_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "Si_s",
    train    = leptocylindrus_danicus_train_regression
  )

#############################################################################-
####################### Predicting on the global ocean ######################
#############################################################################-

#Add global grid to run extrapolation
load('Global_grid_SFChl.Rdata')

global_grid <- 
  Global_grid |> 
  select(Longitude, Latitude, DOY, Temp, PAR, NO3, dFe, P_s, Si_s)

DOYs <- unique(global_grid$DOY)

#Run prediction by season
for (i in 1:length(DOYs)){
  
  #Extract non-NA data in each month
  wx <- which(
      !is.na(global_grid$Temp) &
      !is.na(global_grid$NO3)  &
      !is.na(global_grid$P_s)  &
      !is.na(global_grid$Si_s) &
      !is.na(global_grid$dFe)  &
      !is.na(global_grid$PAR)  &
      global_grid$DOY == DOYs[i]
  )
  
  environmental_data <- global_grid[wx,]
  
  environmental_data$presence <- 
    predict(
      leptocylindrus_danicus_classification_fit, 
      environmental_data
    )$.pred_class
  
  environmental_data$regression <- 
    exp(
      predict(
        leptocylindrus_danicus_regression_fit, 
        environmental_data
      )
    )$.pred
  
  environmental_data$abundance <-
    ifelse(
      environmental_data$presence == 0, 0,
      environmental_data$regression
    )
  
}

environmental_data <-
  environmental_data |> 
  mutate(
    regression = NULL
  )

world_coordinates <-
  map_data(
    "world"
  )

leptocylindrus_danicus_map <-
  ggplot() +
  geom_map(
    data = world_coordinates,
    map =  world_coordinates,
    aes(
      long,
      lat,
      map_id = region
    ),
    fill   = "grey",
    alpha  = 0.25
  ) +
  geom_point(
    data = environmental_data[environmental_data$abundance > 0, ],
    aes(
      Longitude,
      Latitude,
      colour   = log(abundance)
    ),
    size  = 2,
    alpha = 0.5
  ) +
  theme_bw() +
  scale_fill_viridis_c(option="C", aesthetics = "colour") +
  labs(colour = "Log(abundance)",
       x      = "",
       y      = "",
       title  = "Predicted leptocylindrus danicus presence and abundance in the global ocean.") 
  
### End