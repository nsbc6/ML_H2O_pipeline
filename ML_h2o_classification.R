
#=====================================================================================================#
#=====================================================================================================#
#================================= H2O MACHINE LEARNING CLASSIFICATION ==============================#
#=====================================================================================================#
#=====================================================================================================#


#==================================================================================================#
######################################### INFORMATION ##############################################
#==================================================================================================#


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
# 1. Metrics: AUC, TPR, TNR and logloss of chosen algorithms
# 2. TPR-FPR data to do ROC curve
# 3. OPTIONAL: Variable importance 


#==================================================================================================#
############################################# PACKAGES #############################################
#==================================================================================================#

### Required packages. If we do not have those packages --> install
pkgs <- c("devtools", "h2o", "h2oEnsemble", "formattable","rlist","htmltools",
          "webshot", "smotefamily", "data.table", "tidyverse")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) install.packages(pkg)
  
}


### Load packages
library(h2o) # Needs java version <14
library(DataExplorer)
#library(h2oEnsemble)
library(formattable)#table
library(ggplot2)#plot 
library(tidyverse)
library(data.table)
library(rlist)
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


          
ML_h2o_class <- function(data_train,
                   		data_test,
                   		outcome,
                   		algorithms = c("NB"),
                   		train_per = 0.75,
                   		folds = 5,
                   		std_data = FALSE,
                   		oversampling = FALSE, continuous_vars = NULL,dupl = 1,
                   		undersampling = FALSE,
                   		normalization = FALSE,
                   		name = NULL,
                   		var_imp = FALSE,
                   		hyperparams_report = FALSE){
  
  
  print("############################# MACHINE LEARNING INITIATED #############################")
  ############################## CHECK POINT  ############################## 
  
  #Check if algorithms are available
  avai_alg <- c("NB", "GBM", "GLM", "DRF", "XGBoost")
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
  
  
  
  
  ############################ OVER/UNDERSAMPLING ###############################
  
  
  ####################3## Oversampling
  
  if (oversampling == TRUE){
    
    ### Package DMwR
    
    ## Oversampling using SMOTE
    #prov_data_train <- SMOTE(vi_scr_sex.x ~.,
    #                         data_train,
    #                         perc.over = over,
    #                         perc.under = under)
    
    
    ### Package smotefamily
    
    #if(is.null(continuous_vars)){
    #  print("Oversampling option requires continuous variables' index. Please,set it.")
    #  stop()    #}
    
    print("Advise: Categorical variables need numeric codification of natural numbers (0,1,2...")
    print(paste0("Minority class will be oversampled by: x",dupl + 1,"."))
    numeric_data_train<- as.data.frame(apply(data_train,2,as.numeric))
    prov_data_train <- SMOTE(numeric_data_train, numeric_data_train[[outcome]],
                             dup_size = dupl)
    smoted_df <- as.data.frame(prov_data_train$data)
    smoted_df[["class"]] <- NULL #additional column generated
      
    #Round to the closest integer in binary variables
    #smoted_df[,-continuous_vars]<- as.data.frame(sapply(smoted_df[,-continuous_vars],
    #                                                    function(x) {round(x, digits = 0)}))
    #Colnames
    colnames(smoted_df) <- colnames(data_train)
    
    #Transform the variables as factors. Recovering data_train but oversampled
    #data_train<- as.data.frame(apply(smoted_df,2,as.factor))
    data_train<- smoted_df

    #data_train <- prov_data_train #substitute by balanced train dataframe
    
    #Clean
    #remove(prov_data_train)
    
    print("Majority class has been oversampled.")
    print("-------------------------------------------------------------")
              
  }
            
  ################## Undersampling
  
  if (undersampling == TRUE){
    #indexes class 1
    class_1 <- which(data_train[[outcome]] == 0)
    #number class 1
    num_class_1<- length(class_1)
    #indexes class 2
    class_2 <- which(data_train[[outcome]] == 1)
    #number class 2
    num_class_2 <- length(class_2)
    
    if (num_class_1 > num_class_2){
      print("Class 1 will be undersampled.")
      #Select randomply
      to_remove <- sample(class_1, num_class_1-num_class_2)
      data_train <- data_train[-to_remove, ]
    }
    if (num_class_2 > num_class_1){
      print("Class 2 will be undersampled.")
      #Select randomply
      to_remove <- sample(class_2, num_class_2-num_class_1)
      data_train <- data_train[-to_remove, ]
    }
    
  }
  
  
          
  ############################ NORMALIZATION ###############################
  
  if(normalization == TRUE){
    print("Normalization...")
    #Train
    norm_train <- as.data.frame(apply(data_train,2,function(x) range01(as.numeric(x),
                                                                       na.rm = T)))
    norm_train[[outcome]]<- as.factor(norm_train[[outcome]])
    data_train <- norm_train
    
    #Test
    norm_test <- as.data.frame(apply(data_test,2,function(x) range01(as.numeric(x),
                                                                     na.rm = T)))
    norm_test[[outcome]]<- as.factor(norm_test[[outcome]])
    data_test<-norm_test
    
    #Clean up
    remove(norm_train,norm_test)
  }
  
  print("-------------------------------------------------------------")
  
  ####################### VARS TYPES ###################################
  idx_cont<-vector()
  for (cont in names_continuous_variables){
    idx_cont <- append(idx_cont, which(colnames(data_train)==cont))
  }
        
  #data_train[,-idx_cont]<- as.data.frame(apply(data_train[,-idx_cont], 2,as.factor))
  #data_test[,-idx_cont]<- as.data.frame(apply(data_test[,-idx_cont], 2,as.factor))
  
  ########################## FROM R TO H2O INIT #############################
  
      
            
  ### Load data to h2o
  data_train[[outcome]]<-as.factor(data_train[[outcome]])
  data_train <- as.h2o(data_train,destination_frame = "data_train")
  data_test[[outcome]]<-as.factor(data_test[[outcome]])
  data_test <- as.h2o(data_test, destination_frame = "data_test")
          
            
  ########################### INPUTS AND OUTCOME ###########################
          
  #Divide predictors and outcome
  names_pred_vars <- setdiff(h2o.colnames(data_train), outcome)
          
  #CHECK
  print("Data partition: predictors vs outcome DONE")
            
            
            
  ############################### ALGORITHMS  ##############################
          
  ### Dataframe to keep the results of the measures
  df_test_measures <- data.frame(matrix(ncol = 4,
                                        nrow = length(algorithms)))
  colnames(df_test_measures)<-c("AUC", "TPR", "TNR","logloss")
  rownames(df_test_measures)<- algorithms
          
  #Empty list to save results to ROC curve
  models_script <- list()
          
            
            
  
  #===================================================> NAIVE BAYES (nb)
  ################## NAIVE BAYES (nb) ################
            
  if ("NB" %in% algorithms){
    print("Naive Bayes model has started...")
            
    #Build model
    method <- "NB"
            
    ################### Hyperparameters
            
    ################### Other options
    hyper_params_NB <- list(laplace = c(0.01, 0.5, 1, 3, 5))
    search_criteria_nb <- list(strategy = "RandomDiscrete",
                               max_runtime_secs = 24*3600,
                               max_models = 30)
              
              
    ################### Grid search
    nb_model <- h2o.grid(algorithm = "naivebayes",
                         grid_id = "nb_grid_vir",
                         x = names_pred_vars, 
                         y = outcome,
                         training_frame = data_train,
                         #balance_classes = balance,  
                         hyper_params = hyper_params_NB,
                         #class_sampling_factors = c(0.5, 1.),
                         search_criteria = list(strategy = "Cartesian"),#search_criteria_nb,
                         #min_sdev = 0.001,
                         #eps_sdev = 0,
                         #min_prob = 0.001,
                         #eps_prob = 0,
                         nfolds = folds,
                         #fold_assignment = "AUTO", #c("AUTO", "Random", "Modulo", "Stratified"),
                         #stopping_metric = "logloss",
                         #stopping_tolerance = 1e-3,
                         score_each_iteration = TRUE,
                         seed = 123
    )
    
    print("Finished grid search.")
    
    ################### Extract best model
    nb_grid <- h2o.getGrid(grid_id = "nb_grid_vir", sort_by = "auc", decreasing = FALSE)
    nb_best_model_ID <- nb_grid@model_ids[[1]]
    nb_best_model <- h2o.getModel(nb_best_model_ID)
    h2o.saveModel(nb_best_model, paste0(result_pth,"/",method,name), force=TRUE)
    #CHECK
    
    print("Naive Bayes model ended.")
    print("-------------------------------------------------------------")
    
    
    ################ Variable importance
    
    if (var_imp == TRUE){
      NB_var_imp <- h2o.varimp(nb_best_model)
    }
    
    ################## Hyperparameters
    
    if(hyperparams_report == TRUE){
      hyperp_NB <- nb_best_model@model[["model_summary"]]
      print(hyperp_NB)
    }
    
    ################## Predictions
    
    scores_test <- h2o.performance(nb_best_model, data_test)
    
    
    # MEASURES INTO A TABLE
    df_test_measures[method,"AUC"] <- scores_test@metrics$AUC
    df_test_measures[method,"logloss"]<- scores_test@metrics$logloss
    cm <- as.data.frame(scores_test@metrics$cm[2])
    df_test_measures[method,"TPR"]<-cm[1,1]/(cm[1,1]+cm[1,2])
    df_test_measures[method,"TNR"]<-cm[2,2]/(cm[2,2]+cm[2,1])
    
    
    # ROC curve
    #df_toROC <-data.frame()
    df_toROC_nb<- as.data.frame(cbind(scores_test@metrics$thresholds_and_metric_scores[,"tpr"],
                                      scores_test@metrics$thresholds_and_metric_scores[,"fpr"]))
    colnames(df_toROC_nb)<- c('tpr','fpr')
    df_toROC_nb<- add_column(df_toROC_nb, model = "NB")
    
    models_script <- list.append(models_script,df_toROC_nb)
    
    print("===========================================================================")
    
  }
  
  
  
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
                      hyper_params = hyper_params_GBM,
                      #balance_classes = ,
                      search_criteria = search_criteria_gbm,
                      score_tree_interval = 10,
                      #distribution = "multinomial",
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
    
    
    gbm_grid <- h2o.getGrid(grid_id = "gbm_grid0", sort_by = "auc", decreasing = FALSE)
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
      hyperp_gbm <- as.data.frame(best_gbm@model[["model_summary"]])
      print(hyperp_gbm)
    }
    
    
    ################## Predictions
    
    scores_test <- h2o.performance(best_gbm, data_test)
    
    
    # MEASURES INTO A TABLE
    df_test_measures[method,"AUC"] <- scores_test@metrics$AUC
    df_test_measures[method,"logloss"]<- scores_test@metrics$logloss
    cm <- as.data.frame(scores_test@metrics$cm[2])
    df_test_measures[method,"TPR"]<-cm[1,1]/(cm[1,1]+cm[1,2])
    df_test_measures[method,"TNR"]<-cm[2,2]/(cm[2,2]+cm[2,1])
    
    # ROC curve
    #df_toROC <-data.frame()
    df_toROC_gbm<- as.data.frame(cbind(scores_test@metrics$thresholds_and_metric_scores[,"tpr"],
                                       scores_test@metrics$thresholds_and_metric_scores[,"fpr"]))
    colnames(df_toROC_gbm)<- c('tpr','fpr')
    df_toROC_gbm<- add_column(df_toROC_gbm, model = "GBM")
    
    models_script <- list.append(models_script,df_toROC_gbm)
    
    
    print("GBM model ended.")
    
    print("===========================================================================")
  }
  
  
  
  #===================================================> GLM
  
  ####### GLM - REGULARIZATION #########
  
  if("GLM" %in% algorithms){
    
    print("GLM has started...")
    method = "GLM"
    
    
    
    ### Other criteria
    hyper_params_GLM = list(alpha = c(0, 0.1, 0.5, 0.9, 1)
    )#Regularization distribution between L1 and L2 --> Ridge - Elastic Net - Lasso

    search_criteria_glm = list(strategy = "RandomDiscrete",
                               max_runtime_secs = 24*3600,
                               max_models = 30)
    
    print("Grid search started.")
    glm_md <- h2o.grid(algorithm = "glm",
                       grid_id = "glm_gridXX",
                       training_frame = data_train, 
                       x = names_pred_vars, 
                       y = outcome,
                       #balance_classes = ,
                       #class_sampling_factors = class_sampling_,
                       #hyper_params = hyper_params_GLM,
                       #search_criteria = search_criteria_glm,
                       #beta_epsilon = ,
                       #beta_constraints = ,
                       #compute_p_values = , only if lambda=0 and not penalization. remove_collinear_columns recommended
                       family = "binomial", #"gaussian", "multinomial", "poisson", "gamma", "twediee", "quasibinomial"
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
                       #missing_values_handling = "MeanImputation",  ###
                       nfolds = folds,
                       #nlambdas = 4,
                       non_negative = FALSE,   ###
                       #objective_epsilon = , if objective value < this --> stops
                       #offset_columns = 
                       #prior = ,#(0,1)
                       #remove_collinear_columns = TRUE, 
                       seed = 123,
                       score_each_iteration = TRUE,
                       solver = "AUTO", #### "AUTO", "IRLSM", "L_BFGS",
                       #"COORDINATE_DESCENT_NAIVE", COORDINATE DESCENT
                       standardize = FALSE#,
                       #stopping_metric = "auc",
                       #stopping_tolerance = 1e-3,
                       #stopping_rounds = 5
                       #tweedie_link_power =
                       #tweedie_variance_power = 
                       #tweedie_values_handling =                      
                       #weights_column = 
    )
    
    print("Grid search finishes.")
    print("-------------------------------------------------------------")
    
    
    glm_md <- h2o.getGrid(grid_id = "glm_gridXX", sort_by = "auc", decreasing = FALSE)
    best_glm_model_id = glm_md@model_ids[[1]]
    best_glm <- h2o.getModel(best_glm_model_id)
    
    #print(best_m)
    h2o.saveModel(best_glm, paste0(result_pth,"/",method,name), force=TRUE)
    
    ################ Variable importance
    
    if (var_imp == TRUE){
      GLM_var_imp <- h2o.varimp(best_glm)}
    
    ################## Hyperparameters
    
    if(hyperparams_report == TRUE){
      hyperp_glm <- as.data.frame(best_glm@model[["model_summary"]])
      print(hyperp_glm)
    }
    
    ################## Predictions
    
    scores_test <- h2o.performance(best_glm, data_test)
    
    
    # MEASURES INTO A TABLE
    df_test_measures[method,"AUC"] <- scores_test@metrics$AUC
    df_test_measures[method,"logloss"]<- scores_test@metrics$logloss
    cm <- as.data.frame(scores_test@metrics$cm[2])
    df_test_measures[method,"TPR"]<-cm[1,1]/(cm[1,1]+cm[1,2])
    df_test_measures[method,"TNR"]<-cm[2,2]/(cm[2,2]+cm[2,1])
    
    # ROC curve
    #df_toROC <-data.frame()
    df_toROC_glm<- as.data.frame(cbind(scores_test@metrics$thresholds_and_metric_scores[,"tpr"],
                                       scores_test@metrics$thresholds_and_metric_scores[,"fpr"]))
    colnames(df_toROC_glm)<- c('tpr','fpr')
    df_toROC_glm<- add_column(df_toROC_glm, model = "GLM")
    
    models_script <- list.append(models_script,df_toROC_glm)
    
    
    print("GLM model ended.")
    print("===========================================================================")
    
    
  }
  
  
  
  
  #===================================================> RANDOM FOREST
  
  if ("DRF" %in% algorithms){
    
    ####### DRF - RANDOM FOREST #######
    
    print("Random forest has started...")
    print("-------------------------------------------------------------")
    
    stoppingMetric <- "auc"
    method <- "DRF"
    
    
    hyper_params_DRF <- list(
      categorical_encoding = "auto", #c("auto", "one_hot_explicit", "sort_by_response"),
      #histogram_type = c("UniformAdaptive", "QuantilesGlobal", "RoundRobin"),
      max_depth = c(5,20), #c(20, 60, 120, 200),
      min_split_improvement = seq(1e-8, 1e-3),  ###
      min_rows = 10,#c(1,3,10),   ###
      ntrees = c(100,300),#c(100 , 200, 500, 1000), #early stopping 
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
    
    drf_grid <- h2o.getGrid(grid_id = "drf_grid", sort_by = "auc", decreasing = FALSE)
    best_drf_model_id <- drf_grid@model_ids[[1]]
    best_drf <- h2o.getModel(best_drf_model_id)
    h2o.saveModel(best_drf, paste0(result_pth,"/",method,name), force=TRUE)
    
    ################## Variable importance
    if (var_imp == TRUE){
      DRF_var_imp <- h2o.varimp(best_drf)}
    
    ################## Hyperparameters
    
    if(hyperparams_report == TRUE){
      hyperp_drf <- as.data.frame(best_drf@model[["model_summary"]])
      print(hyperp_drf)
    }
    
    
    print("Random forest ended")
    print("-------------------------------------------------------------")
    
    ################## Predictions
    
    scores_test <- h2o.performance(best_drf, data_test)
    
    
    # MEASURES INTO A TABLE
    df_test_measures[method,"AUC"] <- scores_test@metrics$AUC
    df_test_measures[method,"logloss"]<- scores_test@metrics$logloss
    cm <- as.data.frame(scores_test@metrics$cm[2])
    df_test_measures[method,"TPR"]<-cm[1,1]/(cm[1,1]+cm[1,2])
    df_test_measures[method,"TNR"]<-cm[2,2]/(cm[2,2]+cm[2,1])
    
    # ROC curve
    #df_toROC <-data.frame()
    df_toROC_drf<- as.data.frame(cbind(scores_test@metrics$thresholds_and_metric_scores[,"tpr"],
                                       scores_test@metrics$thresholds_and_metric_scores[,"fpr"]))
    colnames(df_toROC_drf)<- c('tpr','fpr')
    df_toROC_drf<- add_column(df_toROC_drf, model = "DRF")
    
    models_script <- list.append(models_script,df_toROC_drf)
    
    
    print("===========================================================================")
  }
  
  ################## EXTREME GRADIENT BOOSTING ####################
  
  if ("XGBoost" %in% algorithms){
    method = "XGBoost"
    print("XGBoost has started.")
    hyper_params_xgb = list(
      categorical_encoding = "auto",#c("auto", "one_hot_explicit", "sort_by_response"),
      #col_sample_rate = c(7,1,1),
      booster = "gbtree",
      gamma = c(0,10,100),
      min_child_weight= c(0,10,100), #bigegr number,algorithm more conservative 
      min_split_improvement = c(1e-8, 1e-3),
      #lambda ##default 1
      #alpha  ##default 0
      max_depth = c(5,20),#c(10, 20, 40, 60, 120),
      learn_rate = 0.3, #c(0.3, 0.1, 0, 0.001.01),
      min_rows = 10,#c(1,30,100),
      ntrees = c(50,100)#c(100, 300, 700, 1000)#, #early stopping
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
    
    xgb_grid <- h2o.getGrid(grid_id = "xgb_grid", sort_by = "auc", decreasing = FALSE)
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
    
    
    # MEASURES INTO A TABLE
    df_test_measures[method,"AUC"] <- scores_test@metrics$AUC
    df_test_measures[method,"logloss"]<- scores_test@metrics$logloss
    cm <- as.data.frame(scores_test@metrics$cm[2])
    df_test_measures[method,"TPR"]<-cm[1,1]/(cm[1,1]+cm[1,2])
    df_test_measures[method,"TNR"]<-cm[2,2]/(cm[2,2]+cm[2,1])
    
    # ROC curve
    #df_toROC <-data.frame()
    df_toROC_xgb<- as.data.frame(cbind(scores_test@metrics$thresholds_and_metric_scores[,"tpr"],
                                       scores_test@metrics$thresholds_and_metric_scores[,"fpr"]))
    colnames(df_toROC_xgb)<- c('tpr','fpr')
    df_toROC_xgb<- add_column(df_toROC_xgb, model = "XGBoost")
    
    models_script <- list.append(models_script,df_toROC_xgb)
    
    
    print("===========================================================================")
  }
  
  ############################# RETURN RESULTS #############################
  
  ### Join all the measures to build the ROC curve into an unique dataframe
  
  df_toROC <- as.data.frame(do.call(rbind, models_script))
  
  ### Return the results
  if(var_imp == FALSE){
    return(list(df_test_measures, df_toROC))}
  
  
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
    

    return(prov)}
  
}

#==================================================================================================#
############################################# CALLING ############################################## 
#==================================================================================================#

h2o.init()