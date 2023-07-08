library(tidyverse)
library(tidymodels)
library(vip)
library(pdp)
library(patchwork)

############################# Data preparation ##############################

data <- read.csv("data.csv")

nitzschia_longissima <- data |> 
  select("Temp","NO3","dFe","PAR","P_s","Si_s","Nitzschia.longissima") |> 
  rename(abundance = Nitzschia.longissima) |> 
  drop_na()

#############################################################################
########################## Classification problem ###########################
#############################################################################

######################### Spending our data budget ##########################

nitzschia_longissima$abundance_factor <- 
  as.factor(
    ifelse(
      nitzschia_longissima$abundance > 0, 1, 0
    )
  )

set.seed(123)
nitzschia_longissima_classification_split <- 
  initial_split(
    nitzschia_longissima, 
    strata = abundance_factor, 
    prop = 0.80
    )

nitzschia_longissima_train_classification <- 
  training(nitzschia_longissima_classification_split)

nitzschia_longissima_test_classification <- 
  testing(nitzschia_longissima_classification_split)

# validation folds

set.seed(234)
nitzschia_longissima_classification_folds <- 
  vfold_cv(
    nitzschia_longissima_train_classification,
    v = 10, 
    repeats = 5, 
    strata = abundance_factor
    )

########################## Pre-processing recipes ###########################

base_classification_rec <-
  recipe(
    abundance_factor ~ Temp + NO3 + dFe + PAR + P_s + Si_s,
    data = nitzschia_longissima_train_classification
    )
# for models which do not require centering or scaling

norm_classification_rec <-
  recipe(
    abundance_factor ~ Temp + NO3 + dFe + PAR + P_s + Si_s,
    data = nitzschia_longissima_train_classification
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
    tree_depth = tune(),
    trees = tune(),
    learn_rate = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    stop_iter = tune()
  ) |> 
  set_engine("xgboost") |>
  set_mode("classification")

# Artifical Neural Network

ann_classification_spec <-
  mlp(
    hidden_units = tune(), 
    penalty = tune(), 
    epochs = tune(), 
    learn_rate = tune(), 
    activation = "relu"
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

ann_classification_workflow <- workflow(norm_classification_rec, ann_classification_spec)

# K-Nearest Neighbour

knn_classification_workflow <- workflow(norm_classification_rec, knn_classification_spec)

################################## Tuning ###################################

doParallel::registerDoParallel()

# Random Forest

rf_classification_grid <- grid_latin_hypercube(   # using a space-filling design
  min_n(),
  finalize(mtry(), nitzschia_longissima_train_classification), # we have to treat mtry differently since it relies on the number of features in the data
  size = 30
  )

rf_classification_grid

set.seed(345)
rf_classification_tuning <- tune_grid(
  rf_classification_workflow,
  resamples = nitzschia_longissima_classification_folds,
  grid = rf_classification_grid,
  control = control_grid(save_pred = TRUE)
  )

rf_classification_tuning

autoplot(rf_classification_tuning) +
  theme_bw()

show_best(rf_classification_tuning, "roc_auc", 10)
show_best(rf_classification_tuning, "accuracy", 10)

rf_classification_tuned_spec <- 
  rand_forest(
    mtry = 2,
    min_n = 23,
    trees = 2000,
  ) |> 
  set_engine("ranger",
             importance = "permutation") |> 
  set_mode("classification")

# Boosted Trees

xgb_classification_grid <- grid_latin_hypercube(   
  tree_depth(), 
  trees(), 
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
  resamples = nitzschia_longissima_classification_folds,
  grid = xgb_classification_grid,
  control = control_grid(save_pred = TRUE)
  )

xgb_classification_tuning

autoplot(xgb_classification_tuning) +
  theme_bw()

show_best(xgb_classification_tuning, "roc_auc") |> View() 
show_best(xgb_classification_tuning, "accuracy") |> View()

xgb_classification_tuned_spec <- 
  boost_tree(
    tree_depth = 12, 
    trees = 180, 
    learn_rate = 0.0784, 
    min_n = 11, 
    loss_reduction = 0.00000000259, 
    sample_size = 0.845, 
    stop_iter = 12
  ) |> 
  set_engine("xgboost") |>
  set_mode("classification") # same as for thaln, interestingly

# Artificial Neural Network

ann_classification_grid <- grid_latin_hypercube(   
  hidden_units(), 
  penalty(), 
  epochs(), 
  learn_rate(),  
  size = 30
  )

ann_classification_grid

set.seed(345)
ann_classification_tuning <- tune_grid(
  ann_classification_workflow,
  resamples = nitzschia_longissima_classification_folds,
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
    hidden_units = 9, 
    penalty = 7.45e-1, 
    epochs = 700, 
    learn_rate = 0.005, 
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
  resamples = nitzschia_longissima_classification_folds,
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
    neighbors = 9, 
    weight_func = "gaussian", 
    dist_power = 1.1
  ) |>
  set_engine("kknn") |>
  set_mode("classification")

############################ Tuned workflow set #############################

nitzschia_longissima_classification_set <-
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

nitzschia_longissima_classification_set <-
  nitzschia_longissima_classification_set |> 
  filter(
    wflow_id %in% c(
      "recipe_1_rand_forest",
      "recipe_1_boost_tree",
      "recipe_2_mlp",
      "recipe_2_nearest_neighbor"
    )
  )

nitzschia_longissima_classification_set

set.seed(345)
nitzschia_longissima_classification_rs <-
  workflow_map(
    nitzschia_longissima_classification_set,
    "fit_resamples",
    resamples = nitzschia_longissima_classification_folds
    )

nitzschia_longissima_classification_rs |> 
  autoplot() +
  theme_bw()

collect_metrics(nitzschia_longissima_classification_rs)

############################## Fit the model ################################

nitzschia_longissima_classification_final_wf <-
  workflow(
    base_classification_rec, 
    rf_classification_tuned_spec
    )

nitzschia_longissima_classification_last_fit <- 
  last_fit(
    nitzschia_longissima_classification_final_wf,
    nitzschia_longissima_classification_split
    )

collect_metrics(nitzschia_longissima_classification_last_fit)

nitzschia_longissima_classification_fit <- 
  fit(
    nitzschia_longissima_classification_final_wf,
    nitzschia_longissima
    )

#############################################################################
############################ Regression problem #############################
#############################################################################

######################### Spending our data budget ##########################

# drop factor

nitzschia_longissima_regression <- 
  nitzschia_longissima |> 
  mutate(
    abundance_factor = NULL
  )

# filter out zeroes

nitzschia_longissima_regression <- 
  nitzschia_longissima |> filter(abundance > 0)

# log-transform the (non-zero) abundances

nitzschia_longissima_regression$log_abundance <-
  log(nitzschia_longissima_regression$abundance)

set.seed(123)
nitzschia_longissima_regression_split <- 
  initial_split(
    nitzschia_longissima_regression, 
    strata = log_abundance, 
    prop = 0.80
  )

nitzschia_longissima_train_regression <- 
  training(nitzschia_longissima_regression_split)

nitzschia_longissima_test_regression <- 
  testing(nitzschia_longissima_regression_split)

# validation folds

set.seed(234)
nitzschia_longissima_regression_folds <- 
  vfold_cv(
    nitzschia_longissima_train_regression,
    v = 10, 
    repeats = 5, 
    strata = log_abundance
  )

########################## Pre-processing recipes ###########################

base_regression_rec <-
  recipe(
    log_abundance ~ Temp + NO3 + dFe + PAR + P_s + Si_s,
    data = nitzschia_longissima_train_regression
    )
# for models which do not require centering or scaling

norm_regression_rec <-
  recipe(
    log_abundance ~ Temp + NO3 + dFe + PAR + P_s + Si_s,
    data = nitzschia_longissima_train_regression
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
    tree_depth = tune(),
    trees = tune(),
    learn_rate = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune(),
    stop_iter = tune()
  ) |> 
  set_engine("xgboost") |>
  set_mode("regression")

# Artifical Neural Network

ann_regression_spec <-
  mlp(
    hidden_units = tune(), 
    penalty = tune(), 
    epochs = tune(),
    learn_rate = tune(),
    activation = "relu"
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

ann_regression_workflow <- workflow(norm_regression_rec, ann_regression_spec)

# K-Nearest Neighbour

knn_regression_workflow <- workflow(norm_regression_rec, knn_regression_spec)

################################## Tuning ###################################

doParallel::registerDoParallel()

# Random Forest

rf_regression_grid <- grid_latin_hypercube(  
  min_n(),
  finalize(mtry(), nitzschia_longissima_train_regression),
  size = 30
  )

rf_regression_grid

set.seed(345)
rf_regression_tuning <- tune_grid(
  rf_regression_workflow,
  resamples = nitzschia_longissima_regression_folds,
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
    mtry = 4,
    min_n = 14,
    trees = 2000
  ) |> 
  set_engine("ranger",
             importance = "permutation") |> 
  set_mode("regression")

# Boosted Trees

xgb_regression_grid <- grid_latin_hypercube(   
  tree_depth(), 
  trees(), 
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
  resamples = nitzschia_longissima_regression_folds,
  grid = xgb_regression_grid,
  control = control_grid(save_pred = TRUE)
  )

xgb_regression_tuning

autoplot(xgb_regression_tuning) +
  theme_bw()

show_best(xgb_regression_tuning, "rsq") |> View()
show_best(xgb_regression_tuning, "rmse") |> View()

xgb_regression_tuned_spec <- 
  boost_tree(
    tree_depth = 14, 
    trees = 1420, 
    learn_rate = 0.00426, 
    min_n = 14, 
    loss_reduction = 0.0000000117, 
    sample_size = 0.436, 
    stop_iter = 17
  ) |> 
  set_engine("xgboost") |>
  set_mode("regression")

# Artificial Neural Network

ann_regression_grid <- grid_latin_hypercube(   
  hidden_units(), 
  penalty(), 
  epochs(), 
  learn_rate(),
  size = 30
  )

ann_regression_grid

set.seed(345)
ann_regression_tuning <- tune_grid(
  ann_regression_workflow,
  resamples = nitzschia_longissima_regression_folds,
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
    penalty = 5.69e-7, 
    epochs = 452, 
    learn_rate = 0.0160, 
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
  resamples = nitzschia_longissima_regression_folds,
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
    neighbors = 8, 
    weight_func = "gaussian", 
    dist_power = 1.1
  ) |>
  set_engine("kknn") |>
  set_mode("regression")

############################ Tuned workflow set #############################

nitzschia_longissima_regression_set <-
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

nitzschia_longissima_regression_set <-
  nitzschia_longissima_regression_set |> 
  filter(
    wflow_id %in% c(
      "recipe_1_rand_forest",
      "recipe_1_boost_tree",
      "recipe_2_mlp",
      "recipe_2_nearest_neighbor"
    )
  )

nitzschia_longissima_regression_set

set.seed(345)
nitzschia_longissima_regression_rs <-
  workflow_map(
    nitzschia_longissima_regression_set,
    "fit_resamples",
    resamples = nitzschia_longissima_regression_folds
    )

nitzschia_longissima_regression_rs |> 
  autoplot() +
  theme_bw()

############################## Fit the model ################################

nitzschia_longissima_regression_final_wf <-
  workflow(
    base_regression_rec, 
    rf_regression_tuned_spec
    )

nitzschia_longissima_regression_last_fit <- 
  last_fit(
    nitzschia_longissima_regression_final_wf,
    nitzschia_longissima_regression_split
    )

collect_metrics(nitzschia_longissima_regression_last_fit)

nitzschia_longissima_regression_fit <- 
  fit(
    nitzschia_longissima_regression_final_wf,
    nitzschia_longissima_regression
    )

#############################################################################
######################### Fitting the hurdle model ##########################
#############################################################################

nitzschia_longissima_classification_predictions <- 
  predict(
    nitzschia_longissima_classification_fit, 
    nitzschia_longissima_test_classification
    )

nitzschia_longissima_regression_predictions <- 
  predict(
    nitzschia_longissima_regression_fit,
    nitzschia_longissima_test_classification
    )

# convert predicted classification values to numeric
nitzschia_longissima_classification_predictions$.pred_class <- 
  as.numeric(
    as.character(
      nitzschia_longissima_classification_predictions$.pred_class
    )
  )

# combine the predictions based on the hurdle model
nitzschia_longissima_hurdle_predictions <- 
  ifelse(
    nitzschia_longissima_classification_predictions$.pred_class == 0, 0, 
    exp(nitzschia_longissima_regression_predictions$.pred)
    ) 

#############################################################################
################### Collect metrics for the hurdle model ####################
#############################################################################

nitzschia_longissima_metrics_bootstrap <- 
  bootstraps(nitzschia_longissima_test_classification, times = 500, strata = abundance)

nitzschia_longissima_classification_predictions_bootstrap <- c()
nitzschia_longissima_regression_predictions_bootstrap <- c()

# Iterate over each bootstrap sample
for (i in 1:length(nitzschia_longissima_metrics_bootstrap$id)) {
  # Get the current bootstrap sample
  bootstrap_sample <- 
    nitzschia_longissima_metrics_bootstrap$splits[[i]]$data[nitzschia_longissima_metrics_bootstrap$splits[[i]]$in_id, ]
  
  # Make predictions on the bootstrap sample
  classification_predictions <- predict(nitzschia_longissima_classification_fit, bootstrap_sample)
  regression_predictions <- predict(nitzschia_longissima_regression_fit, bootstrap_sample)
  
  # Convert classification values to numeric
  classification_predictions$.pred_class <- as.numeric(as.character(classification_predictions$.pred_class))
  
  # Store the bootstrap predictions
  nitzschia_longissima_classification_predictions_bootstrap[[i]] <- classification_predictions
  nitzschia_longissima_regression_predictions_bootstrap[[i]] <- regression_predictions
}

# Create an empty vector/list to store the hurdle predictions
nitzschia_longissima_hurdle_predictions_bootstrap <- c()

# Iterate over each bootstrap sample
for (i in 1:length(nitzschia_longissima_metrics_bootstrap$id)) {
  # Get the classification and regression predictions for the current bootstrap sample
  classification_predictions <- nitzschia_longissima_classification_predictions_bootstrap[[i]]
  regression_predictions <- nitzschia_longissima_regression_predictions_bootstrap[[i]]
  
  # Combine the predictions based on the hurdle model
  hurdle_predictions <- ifelse(
    classification_predictions$.pred_class == 0, 0, 
    exp(regression_predictions$.pred)
  )
  
  # Store the hurdle predictions
  nitzschia_longissima_hurdle_predictions_bootstrap[[i]] <- hurdle_predictions
}

nitzschia_longissima_hurdle_predictions_bootstrap

# calculate R-squared (R2)

rsq_boot <- c()

for (i in 1:length(nitzschia_longissima_metrics_bootstrap$id)) {
  rsq_boot[i] <- cor(
    nitzschia_longissima_hurdle_predictions_bootstrap[[i]], 
    nitzschia_longissima_metrics_bootstrap$splits[[i]]$data[nitzschia_longissima_metrics_bootstrap$splits[[i]]$in_id, 7])^2 
}

paste0(
  "R-squared = ", 
  round(mean(rsq_boot),3), 
  ", 95% CI: (", 
  round(mean(rsq_boot)-1.96*sd(rsq_boot),3), 
  ", ", 
  round(mean(rsq_boot)+1.96*sd(rsq_boot),3),
  ")"
)

# calculate Mean Absolute Error (MAE)

mae_boot <- c()

for (i in 1:length(nitzschia_longissima_metrics_bootstrap$id)) {
  mae_boot[i] <- 
    mean(
      abs(
        nitzschia_longissima_metrics_bootstrap$splits[[i]]$data[nitzschia_longissima_metrics_bootstrap$splits[[i]]$in_id, 7] -
          nitzschia_longissima_hurdle_predictions_bootstrap[[i]]
        )
      )
}

paste0(
  "MAE = ", 
  round(mean(mae_boot),3), 
  ", 95% CI: (", 
  round(mean(mae_boot)-1.96*sd(mae_boot),3), 
  ", ", 
  round(mean(mae_boot)+1.96*sd(mae_boot),3),
  ")"
)

# calculate RMSE

rmse_boot <- c()

for (i in 1:length(nitzschia_longissima_metrics_bootstrap$id)) {
  rmse_boot[i] <- 
    sqrt(
      mean(
        (nitzschia_longissima_metrics_bootstrap$splits[[i]]$data[nitzschia_longissima_metrics_bootstrap$splits[[i]]$in_id, 7] -
          nitzschia_longissima_hurdle_predictions_bootstrap[[i]])^2
      )
    )
}

paste0(
  "RMSE = ", 
  round(mean(rmse_boot),3), 
  ", 95% CI: (", 
  round(mean(rmse_boot)-1.96*sd(rmse_boot),3), 
  ", ", 
  round(mean(rmse_boot)+1.96*sd(rmse_boot),3),
  ")"
)

#############################################################################
############################ Variable importance ############################
#############################################################################

# classification vip:

nitzschia_longissima_classification_fit |>  
  extract_fit_parsnip() |> 
  vip(geom = "point") + 
  labs(title = "Variable importance") +
  theme_bw()

# classification pdp:

nitzschia_longissima_classification_vip_temp <-
  nitzschia_longissima_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "Temp",
    train    = nitzschia_longissima_train_classification
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_classification_vip_no3 <-
  nitzschia_longissima_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "NO3",
    train    = nitzschia_longissima_train_classification
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_classification_vip_dfe <-
  nitzschia_longissima_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "dFe",
    train    = nitzschia_longissima_train_classification
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_classification_vip_par <-
  nitzschia_longissima_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "PAR",
    train    = nitzschia_longissima_train_classification
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_classification_vip_p_s <-
  nitzschia_longissima_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "P_s",
    train    = nitzschia_longissima_train_classification
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_classification_vip_si_s <-
  nitzschia_longissima_classification_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "Si_s",
    train    = nitzschia_longissima_train_classification
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_classification_vip_temp +
  nitzschia_longissima_classification_vip_no3 +
  nitzschia_longissima_classification_vip_dfe +
  nitzschia_longissima_classification_vip_par +
  nitzschia_longissima_classification_vip_p_s +
  nitzschia_longissima_classification_vip_si_s

# regression vip:

nitzschia_longissima_regression_fit |>  
  extract_fit_parsnip() |> 
  vip(geom = "point") +
  labs(title = "Variable importance") +
  theme_bw()

# regression pdp:

nitzschia_longissima_regression_vip_temp <-
  nitzschia_longissima_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "Temp",
    train    = nitzschia_longissima_train_regression
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_regression_vip_no3 <-
  nitzschia_longissima_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "NO3",
    train    = nitzschia_longissima_train_regression
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_regression_vip_dfe <-
  nitzschia_longissima_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "dFe",
    train    = nitzschia_longissima_train_regression
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_regression_vip_par <-
  nitzschia_longissima_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "PAR",
    train    = nitzschia_longissima_train_regression
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_regression_vip_p_s <-
  nitzschia_longissima_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "P_s",
    train    = nitzschia_longissima_train_regression
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_regression_vip_si_s <-
  nitzschia_longissima_regression_fit |> 
  extract_fit_parsnip() |> 
  partial(
    pred.var = "Si_s",
    train    = nitzschia_longissima_train_regression
  ) |> 
  autoplot() +
  theme_bw()

nitzschia_longissima_regression_vip_temp +
  nitzschia_longissima_regression_vip_no3 +
  nitzschia_longissima_regression_vip_dfe +
  nitzschia_longissima_regression_vip_par +
  nitzschia_longissima_regression_vip_p_s +
  nitzschia_longissima_regression_vip_si_s

### End