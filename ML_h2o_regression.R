#=====================================================================================================#
#=====================================================================================================#
#==================================== H2O MACHINE LEARNING REGRESSION ================================#
#=====================================================================================================#
#=====================================================================================================#


#==================================================================================================#
######################################### INFORMATION ##############################################
#==================================================================================================#


#Date: 26th MAY 2020
#Last update: 26th MAY 2020

#R version: 3.6.1 (2019-07-05)
#H2O version: 3.30.0.1

#It is a large function to have a control of some options and to perform 
#different tests to compare between them.

#It needs:
# 1. Training data
# 2. Test data
# 3. Names of continuous variables
# 4. OPTIONS 


#It returns:
# 1. Metrics: MSE, MAE and R² of chosen algorithms
# 2. Predicted vs observed values of chosen algorithms
# 3. OPTIONAL: Variable importance 


#==================================================================================================#
############################################# PACKAGES #############################################
#==================================================================================================#


### Required packages. If we do not have those packages --> install
pkgs <- c("devtools", "h2o", "h2oEnsemble", "formattable","rlist","htmltools",
          "webshot", "smotefamily")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) install.packages(pkg)
  
}


### Load packages
library(h2o) # Needs java version <14
library(DataExplorer)
#library(h2oEnsemble)
library(formattable)
library(ggplot2)
library(tidyverse)
library(rlist)
library(data.table)
#library(DMwR)
library(htmltools)#table
library(webshot)#table
library(smotefamily)


#==================================================================================================#
############################################ FUNCTIONS ############################################# 
#==================================================================================================#


########################## NORMALIZATION BETWEEN 0-1 ########################## 

#na.rm = T to ignore NA, but setting then

range01<-function(x, ...){
  (x-min(x, ...))/(max(x, ... )-min(x, ...))}


##############################  MACHINE LEARNING ############################## 



ML_h2o_regression <- function(data_train,
                   data_test,
                   outcome,
                   distribution_reg = "gaussian",
                   algorithms = c("NB"),
                   train_per = 0.75,
                   folds = 10,
                   std_data = FALSE,
                   normalization = FALSE,
                   name = NULL,
                   var_imp = FALSE,
                   hyperparams_report = FALSE){
  
  
  
  ############################## CHECK POINT  ############################## 
  
  #Check if algorithms are available
  avai_alg <- c("GBM", "GLM", "DRF", "XGBoost", "DL")
  for (alg in algorithms){
    if (!(alg %in% avai_alg)){
      print(paste(alg,
                  "is not available to perform the analysis. Please, select another algorithm: ",
                  avai_alg))
      stop()}}
  
  #Check if outcome is a real column name
  if (!(outcome %in% colnames(data_train))){
    print(paste(outcome,
                "is not a column name of data. Please, check it and set another one."))
    stop()}
  
  #### Check if directory already exists -> overwrite!!!!!!!!
  
  print("Arguments appear correct.")
  print("-------------------------------------------------------------")
  
  
  
  
  ############################ NORMALIZATION ###############################
  
  if(normalization == TRUE){
    print("Normalization...")
    #Train
    norm_train <- as.data.frame(apply(data_train,2,function(x) range01(as.numeric(x),
                                                                       na.rm = T)))
    data_train <- norm_train
    
    #Test
    norm_test <- as.data.frame(apply(data_test,2,function(x) range01(as.numeric(x),
                                                                     na.rm = T)))
    data_test<-norm_test
    
    #Clean up
    remove(norm_train,norm_test)
  }
  
  print("-------------------------------------------------------------")
  
  
  
  ########################## FROM R TO H2O INIT #############################
  
  
  
  ### Load data to h2o
  data_train <- as.h2o(data_train,destination_frame = "data_train")
  data_test <- as.h2o(data_test, destination_frame = "data_test")
  
  
  ### Differentiate outcome and predictiv variables
  
  
  
  #CHECK
  print("H2O has been initiated...")
  
  ########################### INPUTS AND OUTCOME ###########################
  
  #Divide predictors and outcome
  names_pred_vars <- setdiff(h2o.colnames(data_train), outcome)
  
  #CHECK
  print("Data partition: predictors vs outcome DONE")
  
  
  
  ############################### ALGORITHMS  ##############################
  data_train[[outcome]]<-as.numeric(data_train[[outcome]])
  data_test[[outcome]]<-as.numeric(data_test[[outcome]])
  
  ### Dataframe to keep the results of the measures
  df_test_measures <- data.frame(matrix(ncol = 3,
                                        nrow = length(algorithms)))
  colnames(df_test_measures)<-c("r2", "MSE", "MAE")
  rownames(df_test_measures)<- algorithms
  
  pred_vs_observations <-list()
  
  #===================================================> GBM
  
  ######### GRADIENT BOOSTING ###########
  
  if ("GBM" %in% algorithms){
    
    print("GBM has started.")
    method <- "GBM"
    
    
    
    ### Other criteria
    hyper_params_GBM = list(
      categorical_encoding = "auto", #c("auto", "one_hot_explicit", "sort_by_response"),
      #col_sample_rate = c(7,1,1),
      #histogram_type = c("UniformAdaptive", "QuantilesGlobal", "RoundRobin"),
      learn_rate = 0.5, #c(0.3, 0.1, 0, 0.001.01),
      learn_rate_annealing = 0.99, #c(0.8, 0.9, 0.995, 1),
      min_split_improvement = c(1e-8, 1e-3),
      max_depth = c(5,20),#c(10, 20, 40, 60, 120),
      min_rows = 1, #c(1,30,100),
      nbins = c(20,100),#c(30,100,300),
      nbins_cats = 500, #c(64, 256, 1024),
      ntrees = c(200,300)#c(100, 300, 700, 1000)#, #early stopping
      #sample_rate = seq(0.4, 0.7, 1, 1)
      #stopping_metric = "logloss" 
    )
    
    
    search_criteria_gbm = list(strategy = "RandomDiscrete",
                               max_runtime_secs = 24*3600,
                               max_models = 30
    )
    
    # Grid-search
    gbm_md = h2o.grid(algorithm = "gbm", 
                      grid_id = "gbm_grid0",
                      x = names_pred_vars,
                      y = outcome, 
                      training_frame = data_train,
                      hyper_params = hyper_params_GBM, #TARDA MUCHO, PROBAR PRIMERO
                      #balance_classes = ,
                      search_criteria = search_criteria_gbm,
                      score_tree_interval = 10,
                      distribution = distribution_reg,
                      #fold_column = 
                      #fold_assignment =
                      #huber_alpha = 
                      #max_abs_leafnode_pred = ,
                      #offset_column = 
                      #pred_noise_bandwith = 
                      #quantile_alpha = 
                      score_each_iteration = TRUE,
                      nfolds = folds,
                      #keep_cross_validation_predictions = TRUE,
                      #stopping_metric = "AUC",
                      #stopping_tolerance = 1e-3,
                      #stopping_rounds = 5,
                      seed = 123
                      #tweedie_power = 
                      #weights_column = 
    )
    
    
    gbm_grid <- h2o.getGrid(grid_id = "gbm_grid0", sort_by = "mse", decreasing = FALSE)
    best_gbm_model_id <- gbm_grid@model_ids[[1]]
    best_gbm <- h2o.getModel(best_gbm_model_id)

    
    #print(best_gbm)
    h2o.saveModel(best_gbm, paste0(result_pth,"/",method,name), force=TRUE)
    
    ################## Variable importance
    if (var_imp == TRUE){
      GBM_var_imp <- h2o.varimp(best_gbm)
    }
    
    ################## Hyperparameters
    
    if(hyperparams_report == TRUE){
      hyperp_gbm <- best_gbm@model[["model_summary"]]
      print(hyperp_gbm)
    }
    
    
    
    ################## Predictions
    
    scores_test <- h2o.performance(best_gbm, data_test)
    
    
    pred_test <- h2o.predict(best_gbm, data_test)
    pred_vs_actual <- as.data.frame(h2o.cbind(data_test[[outcome]],pred_test))
    pred_vs_observations <-list.append(pred_vs_observations, pred_vs_actual)
    
    
    # MEASURES INTO A TABLE
    df_test_measures[method,"r2"] <- scores_test@metrics$r2
    df_test_measures[method,"MSE"]<- scores_test@metrics$MSE
    df_test_measures[method,"MAE"]<-scores_test@metrics$mae
    
    
    
    print("GBM model ended.")
    
    print("===========================================================================")
  }
  
  
  
  #===================================================> GLM
  
  ####### GLM - ELASTIC NET #########
  
  if("GLM" %in% algorithms){
    
    print("GLM has started...")
    method = "GLM"
    
    
    
    ### Other criteria
    hyper_params_GLM = list(alpha = c(0,0.1, 0.5, 0.9,1),
                            stopping_metric = "rmse")
    					#Regularization distribution between L1 and L2 --> Ridge - Elastic Net - Lasso
    
    search_criteria_glm = list(strategy = "RandomDiscrete",
                               max_runtime_secs = 24*3600,
                               max_models = 30
                               )
    
    print("Grid search started.")
    glm_md = h2o.grid(algorithm = "glm", 
                      grid_id = "glm_gridXX",
                      training_frame = data_train, 
                      x = names_pred_vars, 
                      y = outcome,
                      hyper_params = hyper_params_GLM,
                      search_criteria = search_criteria_glm,
                      #beta_epsilon = ,
                      #beta_constraints = ,
                      #compute_p_values = , only if lambda=0 and not penalization. remove_collinear_columns recommended
                      family = distribution_reg, #"gaussian", "multinomial", "poisson", "gamma", "twediee", "quasibinomial"
                      fold_assignment = "AUTO",
                      #fold_column = 
                      #gradient_epsilon = ,# For L-BFGS only
                      #ignored_columns = , only for python
                      #interactions = ,
                      #interaction_pairs = ,
                      intercept = FALSE,   ###
                      #ignore_const_cols = TRUE,   ###
                      #lambda = ,#regularization strength
                      #lambda_min_ratio = ,
                      #link = c("family_default", "identity", "logit", "log", "inverse", "tweedie","ologit", "oprobit", "ologlog"),
                      lambda_search = TRUE,
                      #max_iterations = 100,    ###
                      #max_active_predictors = ,
                      missing_values_handling = "MeanImputation",  ###
                      nfolds = 10,
                      #nlambdas = 4,
                      non_negative = FALSE,   ###
                      #objective_epsilon = , if objective value < this --> stops
                      #offset_columns = 
                      #prior = ,#(0,1)
                      #remove_collinear_columns = TRUE, 
                      seed = 123,
                      score_each_iteration = TRUE,
                      solver = "AUTO", #### "AUTO", "IRLSM", "L_BFGS", "COORDINATE_DESCENT_NAIVE", COORDINATE DESCENT
                      standardize = FALSE,
                      #stopping_metric = "rmse",
                      stopping_tolerance = 1e-3,
                      stopping_rounds = 5
                      #tweedie_link_power =
                      #tweedie_variance_power = 
                      #tweedie_values_handling =                      
                      #weights_column = 
    )
    print("Grid search finishes.")
    print("-------------------------------------------------------------")
    
    
    glm_md <- h2o.getGrid(grid_id = "glm_gridXX", sort_by = "rmse", decreasing = FALSE)
    best_glm_model_id = glm_md@model_ids[[1]]
    best_glm <- h2o.getModel(best_glm_model_id)
    
    #print(best_gbm)
    h2o.saveModel(best_glm, paste0(result_pth,"/",method,name), force=TRUE)
    
    if (var_imp == TRUE){
      GLM_var_imp <- h2o.varimp(best_glm)}
    
    ################## Hyperparameters
    
    if(hyperparams_report == TRUE){
      hyperp_glm <- best_glm@model[["model_summary"]]
      print(hyperp_glm)
    }
    
    
    ################## Predictions
    
    pred_test <- h2o.predict(best_glm, data_test)
    pred_vs_actual <- as.data.frame(h2o.cbind(data_test[[outcome]],pred_test))
    pred_vs_observations <-list.append(pred_vs_observations, pred_vs_actual)
    
    scores_test <- h2o.performance(best_glm, data_test)
    
    
    # MEASURES INTO A TABLE
    df_test_measures[method,"r2"] <- scores_test@metrics$r2
    df_test_measures[method,"MSE"]<- scores_test@metrics$MSE
    df_test_measures[method,"MAE"]<-scores_test@metrics$mae
    
  
    
    print("GLM model ended.")
    print("===========================================================================")
    
    
  }
  
  
  
  
  #===================================================> RANDOM FOREST
  
  if ("DRF" %in% algorithms){
    
    ####### DRF - RANDOM FOREST #######
    
    print("Random forest has started...")
    print("-------------------------------------------------------------")
    
    stoppingMetric <- "rmse"
    method <- "DRF"
    
    
    hyper_params_DRF <- list(
      categorical_encoding = "auto", #c("auto", "one_hot_explicit", "sort_by_response"),
      #histogram_type = c("UniformAdaptive", "QuantilesGlobal", "RoundRobin"),
      max_depth = c(5,20), #c(20, 60, 120, 200),
      min_split_improvement = seq(1e-8, 1e-3),  ###
      min_rows = 10,#c(1,3,10),   ###
      ntrees = c(100,300),
      #ntrees = c(50,100),#c(100 , 200, 500, 1000), #early stopping REDUCIDO PORQUE SINO ES ETERNO. PARA PRUEBAS!!!!
      nbins = c(20,100),#c(30, 100, 200),     ##
      nbins_cats = 500, #c(64, 256, 1024),   ###
      #nbins_top_level = 
      #sample_rate_per_class = seq(0.3, 0.5, 0.7), ###
      stopping_metric = "logloss"#, "missclassification")
    )
    
    search_criteria_drf <- list(strategy = "RandomDiscrete",
                                max_runtime_secs = 24*3600,
                                max_models = 30)
    
    print("Grid started.")
    print("-------------------------------------------------------------")
    drf_md <- h2o.grid(algorithm = "randomForest", 
                       grid_id = "drf_grid",
                       training_frame = data_train, 
                       x = names_pred_vars,
                       y = outcome,
                       search_criteria = search_criteria_drf,
                       binomial_double_trees = TRUE,  ###
                       #balance_classes = ,
                       hyper_params = hyper_params_DRF, 
                       #class_sampling_factors = class_sampling_,
                       #col_sample_rate_per_tree = 
                       #col_sample_rate_change_per_level = 
                       #fold_assignment = 
                       #fold_columns = 
                       #ignored_columns = 
                       ignore_const_cols = TRUE,  ###
                       nfolds = folds,
                       #max_after_balance_size = 
                       mtries = -1,     #columns to randomly select at each level. If -1,
                       #n_variables=square root of n_columns
                       seed = 123,
                       #sample_rate = 
                       #sample_rate_per_class =
                       #sample_rate_per_tree = 
                       score_tree_interval = 20,  ###
                       #stopping_metric = "AUC",
                       stopping_tolerance = 0.01
                       #stopping_rounds = 5
                       #weight_columns = 
    )
    
    drf_grid <- h2o.getGrid(grid_id = "drf_grid", sort_by = "rmse", decreasing = FALSE)
    best_drf_model_id <- drf_grid@model_ids[[1]]
    best_drf <- h2o.getModel(best_drf_model_id)
    h2o.saveModel(best_drf, paste0(result_pth,"/",method,name), force=TRUE)
    
    ################## Variable importance
    if (var_imp == TRUE){
      DRF_var_imp <- h2o.varimp(best_drf)}
    
    
    ################## Hyperparameters
    
    if(hyperparams_report == TRUE){
      hyperp_drf <-best_drf@model[["model_summary"]]
      print(hyperp_drf)
    }
    
    
    print("Random forest ended")
    print("-------------------------------------------------------------")
    
    ################## Predictions
    
    scores_test <- h2o.performance(best_drf, data_test)
    
    pred_test <- h2o.predict(best_drf, data_test)
    pred_vs_actual <- as.data.frame(h2o.cbind(data_test[[outcome]],pred_test))
    pred_vs_observations <-list.append(pred_vs_observations, pred_vs_actual)
    
    # MEASURES INTO A TABLE
    df_test_measures[method,"r2"] <- scores_test@metrics$r2
    df_test_measures[method,"MSE"]<- scores_test@metrics$MSE
    df_test_measures[method,"MAE"]<-scores_test@metrics$mae
    
    
  
    
    print("===========================================================================")
  }
  
  ########################## EXTREME GRADIENT BOOSTING
  
  if ("XGBoost" %in% algorithms){
    method = "XGBoost"
    hyper_params_xgb = list(
      categorical_encoding = "auto",#c("auto", "one_hot_explicit", "sort_by_response"),
      #col_sample_rate = c(7,1,1),
      booster = "gbtree",
      gamma = c(0,10,100),
      min_child_weight= c(1,10,100), #bigegr number,algorithm more conservative 
      min_split_improvement = c(1e-8, 1e-3),
      #lambda ##default 1
      #alpha  ##default 0
      max_depth = c(5,20),#c(10, 20, 40, 60, 120),
      learn_rate = 0.3, #c(0.3, 0.1, 0, 0.001.01),
      min_rows = 10,#c(1,30,100),
      ntrees = c(50,100)
      #ntrees = c(50,100)#c(100, 300, 700, 1000)#, #early stopping
      #sample_rate = seq(0.4, 0.7, 1, 1)
      #stopping_metric = "logloss" #LO CAMBIÉ
    )
    
    xgb_md <- h2o.grid(algorithm = "xgboost", 
                       grid_id = "xgb_grid",
                       training_frame = data_train, 
                       x = names_pred_vars,
                       y = outcome,
                       #search_criteria = search_criteria_drf,
                       #balance_classes = balance,
                       hyper_params = hyper_params_xgb, 
                       #class_sampling_factors = class_sampling_,
                       #col_sample_rate_per_tree = 
                       #col_sample_rate_change_per_level = 
                       #fold_assignment = 
                       #fold_columns = 
                       #ignored_columns = 
                       ignore_const_cols = TRUE,  ###
                       nfolds = folds,
                       distribution =distribution_reg,
                       #max_after_balance_size = 
                       #n_variables=square root of n_columns
                       seed = 123,
                       #sample_rate = 
                       #sample_rate_per_class =
                       #sample_rate_per_tree = 
                       score_tree_interval = 20,  ###
                       #stopping_metric = "AUC",
                       stopping_tolerance = 0.01#,
                       #stopping_rounds = 5
                       #weights_column = 
    )
    
    xgb_grid <- h2o.getGrid(grid_id = "xgb_grid", sort_by = "rmse", decreasing = FALSE)
    best_xgb_model_id <- xgb_grid@model_ids[[1]]
    best_xgb <- h2o.getModel(best_xgb_model_id)
    h2o.saveModel(best_xgb, paste0(result_pth,"/",method,name), force=TRUE)
    
    ################## Variable importance
    if (var_imp == TRUE){
      XGB_var_imp <- h2o.varimp(best_xgb)}
    
    ################## Hyperparameters
    
    if(hyperparams_report == TRUE){
      hyperp_xgb <- best_xgb@model[["model_summary"]]
      print(hyperp_xgb)
    }
    
    
    
    print("XGBoost ended.")
    print("-------------------------------------------------------------")
    
    ################## Predictions
    
    scores_test <- h2o.performance(best_xgb, data_test)
    
    pred_test <- h2o.predict(best_xgb, data_test)
    pred_vs_actual <- as.data.frame(h2o.cbind(data_test[[outcome]],pred_test))
    pred_vs_observations <-list.append(pred_vs_observations, pred_vs_actual)
    
    # MEASURES INTO A TABLE
    df_test_measures[method,"r2"] <- scores_test@metrics$r2
    df_test_measures[method,"MSE"]<- scores_test@metrics$MSE
    df_test_measures[method,"MAE"]<-scores_test@metrics$mae
    
 
    
    print("===========================================================================")
  }
  
  if ("DL" %in% algorithms){
    print("DL started")
    method = "DL"
    # Definition of the grid
    hyper_params_DL = list(                       
      activation=c("Rectifier","Tanh"),
      epochs = c(10, 100, 500, 1000),
      hidden = list(c(100,100,100), c(200,100,100,50), c(500,500,200,200,100,200,200,500),
                    c(500,300,200,100,50,50,50,100,200,300,400),
                    c(500,400,350,300,150,100,100,50,50,150,100,200,100,400,200)),
      #c(1500,1000,1000,750,500,500,600,900,2000,1500)),
      l1 = c(0, 0.15, 0.8),   ####
      l2 = c(0, 0.15, 0.8),   ###
      loss = c("Automatic", "Quadratic"), ###"Huber", "Absolute", "Quantile"
      stopping_metric = "rmse",
      rho = c(0.3, 0.1, 0.01) ### #adaptive learning rate decay
    )
    
    search_criteria_dl = list(strategy = "RandomDiscrete",
                              max_runtime_secs = 24*3600,
                              max_models = 30)
    
    #Grid-search
    dl_md = h2o.grid( algorithm = "deeplearning", 
                      grid_id = "dl_grid",
                      x = names_pred_vars,
                      y = outcome, 
                      training_frame = data_train,
                      hyper_params = hyper_params_DL, 
                      search_criteria = search_criteria_dl,
                      #activation = "Tanh",
                      adaptive_rate = TRUE,
                      #average_activation = 
                      autoencoder = FALSE,
                      #balance_classes = TRUE,
                      categorical_encoding = "AUTO",
                      classification_stop = 0.01,
                      #class_sampling_factors =
                      #col_sample_rate = 0.8,
                      distribution = distribution_reg,    #"bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie", "laplace", "quantile", "huber"
                      elastic_averaging = TRUE,  # To improve distributed model convergence
                      #elastic_averaging_moving_rate = 
                      #elastic_averaging_regularization =  
                      #epsilon = 1e-8,  #adaptive learning rate smoothin factor (to avoid divisions by zero and allow progress)
                      export_weights_and_biases = FALSE,
                      fast_mode = TRUE,
                      #force_load_balance = 
                      #fold_column = 
                      fold_assignment = "Modulo", #### "RANDOM", "STRATIFIED", "MODULO"
                      #ignored_columns = 
                      ignore_const_cols = TRUE,
                      #input_dropout_ratio = 
                      initial_weight_distribution = "UniformAdaptive", ### "Uniform", "Normal".
                      #initial_weight_scale = 
                      #initial_weights = 
                      #initial_biases = 
                      #hidden_dropout_ratios =
                      #huber_alpha = 0.9,  #desired quantile. Only for Humber/m-regression
                      #loss = "Automatic",  ####
                      #max_after_balance_size =
                      #max_w2 =
                      #max_categorical_features =
                      #missing_values_handling = "MeanImputation",
                      #momentum_start = 
                      #momentum_ramp = 
                      #momentum_stable =
                      nfolds = folds,
                      #nesterov_accelerated_gradient =
                      #offset_column = 
                      #overwrite_with_best_model =
                      #quantile_alpha = 
                      #quite_mode =
                      #rate = 
                      #rate_annealing = 
                      #rate_decay = 
                      #refression_stop =
                      replicate_training_data = FALSE,  ###
                      reproducible = FALSE,
                      #rho = 0.9,   #Adaptive learning rate time decay factor
                      #sample_rate = 0.8, #bigger for small datasets. Add also cv in these cases. For large datasets it is not needed
                      seed = 123,
                      score_each_iteration = TRUE,
                      score_duty_cycle = 0.2, #Maximum duty cycle fraction for scoring (lower: more training, higher: more scoring).
                      #score_validation_samples = 0,
                      score_interval = 3,
                      score_training_samples = 10000,
                      #score_validation_samples = 10000, ## downsample validation set for faster scoring
                      #score_duty_cycle = 0.025,         ## don't score more than 2.5% of the wall time
                      standardize = TRUE,   
                      #stopping_metric = "deviance",
                      stopping_tolerance = 1e-3,
                      stopping_rounds = 3,
                      single_node_mode = FALSE,
                      #shuffle_training_data = TRUE,
                      #sparse = TRUE,
                      #sparsity_beta = 
                      train_samples_per_iteration = -2,  # auto-search
                      #target_ratio_comp_to_comp = 
                      #tweedie_power = 
                      use_all_factor_levels = TRUE,
                      variable_importances = TRUE
                      #weights_column =
    )
    
    
    dl_grid = h2o.getGrid(grid_id = "dl_grid", sort_by = "rmse", decreasing = FALSE)
    best_dl_model_id = dl_grid@model_ids[[1]]
    best_dl = h2o.getModel(best_dl_model_id)
    h2o.saveModel(best_dl, paste0(result_pth,"/",method,name), force=TRUE)
    
    ################## Variable importance
    if (var_imp == TRUE){
      DL_var_imp <- h2o.varimp(best_dl)}
    
    ################## Hyperparameters
    
    if(hyperparams_report == TRUE){
      hyperp_dl <- best_dl@model[["model_summary"]]
      print(hyperp_dl)
    }
    
    
    ################## Predictions
    
    scores_test <- h2o.performance(best_dl, data_test)
    
    pred_test <- h2o.predict(best_dl, data_test)
    pred_vs_actual <- as.data.frame(h2o.cbind(data_test[[outcome]],pred_test))
    pred_vs_observations <-list.append(pred_vs_observations, pred_vs_actual)
    
  
    # MEASURES INTO A TABLE
    df_test_measures[method,"r2"] <- scores_test@metrics$r2
    df_test_measures[method,"MSE"]<- scores_test@metrics$MSE
    df_test_measures[method,"MAE"]<-scores_test@metrics$mae
  }

  ############################# RETURN RESULTS #############################
  

  ### Return the results
  if(var_imp == FALSE){
    return(list(df_test_measures, pred_vs_observations))}
  
  if(var_imp == TRUE){
  	prov <- list(df_test_measures, df_toROC)

  	if ("GBM" %in% algorithms){
  		prov <- list.append(prov, GBM_var_imp)
  	}

  	if ("GLM" %in% algorithms){
  		prov <- list.append(prov, GLM_var_imp)
  	}


  	if ("DRF" %in% algorithms){
  		prov <- list.append(prov, DRF_var_imp)
  	}

  	if ("XGBoost" %in% algorithms){
  		prov <- list.append(prov, XGB_var_imp)
  	}
    

    if ("DL" %in% algorithms){
    	prov <- list.append(prov, DL_var_imp)
    }

    return(prov)}
  
}

#==================================================================================================#
############################################# CALLING ############################################## 
#==================================================================================================#

h2o.init()