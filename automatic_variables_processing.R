  ###==================================================================================###
  ###======================================INFORMATION=================================###
  ###==================================================================================###
  
  
  
  
  ###By: Nuria Sánchez de la Blanca Carrero
  ### R script to process  large data  variables automaticaly using R
  ### Started at:19th November 2019
  ### Last updating: 24th January 2020
  
  
  # Created with 3.5 version of R. This script provides an analysis of features to
  # clean them, with the aim of become a clean predictive variables as much as possible.
  # It is necessary to divide previously outcomes and predictive variables. Predictive
  #variables are provided by the user to perform the script. It is necessary an ID's
  # variable located on the first column.
  
  ###==================================================================================###
  ###=====================================PACKAGES=====================================###
  ###==================================================================================###
  
  
  library(DataExplorer)
  library(tidyverse)
  library(plyr); library(dplyr)
  library(mlr)
  library(rio)
  library(devtools)
  library(FactoMineR)
  library(VIM)
  library(factoextra)
  library(ggplot2)
  library(rlist)
  
  
  
  ###==================================================================================###
  ###======================================VARIABLES===================================###
  ###==================================================================================###
  
  
  #Here users set the working directory 
  workingDirectory <- "./"
  
  #Users write the file of variables they want to filter
  #IDs will be taking into account
  predictive_variables <- non_normalized_all_vars[,-1]
  
  
  #Uncomment the following line if yo want to remove some additional column you do not
  # want or change number
  #predictive_variables <-predictive_variables[, -1]
  
  #directory to save results-->OUTPUTS. PROVIDED BY USER.
  saveData <- "./ALL_no_norm_categbin_SEX"
  OUTPUT_file <- "ALL_no_norm_categbin_FILT_SEX.csv"
  
  
  
  ###==================================================================================###
  ###==================================== CODE ========================================###
  ###==================================================================================###
  
  
  
  ###====================================DIRECTORIES===================================###
  #setting working directory
  setwd(workingDirectory)
  
  
  #OUTPUTS
  dir.create(saveData)
  
  #changing the work directory
  setwd(saveData)
  
  
  ###==================================== FUNCTIONS ===================================###
  
  
  ########    1
  
  # This function returns the list of indexes(in a vector) and/or number of variables that
  # have more than  a given number of missing values. User sets:
  #   1-Maximum percentage of missing values he/she wants
  #   2-Data frame
  #   3-If he/she wants or not the number of those variables(TRUE/FALSE).
  #   4-If he/she wants or not the vector where those variables are (TRUE/FALSE).
  
  count_vars_moreMV<- function(percentage, data_frame, number=FALSE, indexes=TRUE){
    #Missing values  percentage
    less_predVars <- colMeans(is.na(data_frame))
    
    #if we want the indexes where thoses variables are
    if(indexes==TRUE){
      tmp_index <- vector()
      for (p in less_predVars){
        if (p > percentage){
          tmp_index <-  append(tmp_index, which (less_predVars == p))}
      }
    return(unique(tmp_index))
    }    
    
    #if we only want the number of those variables
    if(number==TRUE){
      for (p in less_predVars){
        if (p > percentage){
          tmp_index <-  append(tmp_index, p)}}
      return(length(tmp_index))
    }
    
    #variables name
    if (name==TRUE){
      sapply(pp,  function (x) colnames(data_frame[x])) #!!!!!!!!!!!!
    }
    
  }
  
  
  ########    2
  
  # This function returns rows with a given percentage. If user wants, it can provide the quantity
  # of this and/or the ID's or keys.
  # of missing values. User provides:
  #   1-Maximum percentage of missing values he/she wants
  #   2-Data frame
  #   3-If he/she wants or not the number of those patients (TRUE/FALSE). FALSE by default.
  #   4-If he/she wants or not the vector where those patients are (TRUE/FALSE). FALSE by default.
  
  perc_MV_patients <- function(percentage, user_dataFrame, ids = FALSE, number =FALSE){
    #return lines
    line_sel <- vector()
    means_ind_MV <- apply(user_dataFrame, 1, function(x) mean(is.na(x)))
    for (mean in means_ind_MV){if (mean >  percentage){
        tmp_place <- which(means_ind_MV==mean, arr.ind = TRUE)
        line_sel <- append(line_sel, tmp_place[1])
      }
    }
    
    if (is_empty(line_sel) == TRUE ){
      line_sel<- list()
    } else{
      #only continue if there is something to do (patients with this percentage)
      list_of_returns <- list("Lines"=unique(line_sel))
  
      #return ids
      if (ids==TRUE){
        ids <- vector()
        for (line in unique(line_sel)){
          ids <- append(ids,as.matrix(user_dataFrame[line,1]))} #as.matrix,a trick to localize position
        list_of_returns <- c(list("IDs"=ids), list_of_returns)
      }
      
     #return the number
      if (number== TRUE){
        list_of_returns <- c(list("Number"=length(unique(line_sel))), list_of_returns)
      }
      
      #if any of the statement is TRUE: return the list with only an element
      return(list_of_returns)
    }
  }
  
  
  
  ###========================== NON- INFORMATIVE VARIABLES ============================###
  
  # In this section, variables with only one characteristic will be dropped. NAs are not traited
  #as an extra level. We consider that those features will not explain the variability
  # (non informative variables).
  
  ncf_pred_vars <- removeConstantFeatures(predictive_variables , perc = 0, na.ignore = TRUE,
                                          show.info = FALSE)
  
  if (is_empty(ncf_pred_vars)){
    ncf_pred_vars<-predictive_variables
    F1_n_Vars<- ncol(ncf_pred_vars)
    F1_n_Pats <- nrow(ncf_pred_vars)
    #Clean up
    remove(rm_noV_pred_vars, all_pred_vars)
  }else{
    #We also keep the variables that we dropped at this point:
    all_pred_vars <- colnames(predictive_variables)
    rm_noV_pred_vars <- vector()
    for (name in all_pred_vars){
      if (!(name %in% colnames(ncf_pred_vars))){
        rm_noV_pred_vars <- append(rm_noV_pred_vars, name)}
    }
    write(rm_noV_pred_vars, file = "./rm_noVariability_pred_vars.txt")
    F1_n_Vars<- ncol(ncf_pred_vars)
    F1_n_Pats <- nrow(ncf_pred_vars)
   
   #Clean up
    remove(rm_noV_pred_vars, all_pred_vars)
  }
  
  
  
  
  ###========================= VARIABLES AND MISSING VALUES ===========================###
  
  #All of missing values are transformed to "NA"
  ncf_pred_vars[ncf_pred_vars == ""] = NA
  
  
  ## ===============> >80% OF MISSING VALUES
  
  #call function
  indx_more80 <- count_vars_moreMV(0.8,ncf_pred_vars,indexes = TRUE)
  
  if(is_empty(indx_more80)){
    #if any, maintain the dataset
    noMV_pred_vars<-ncf_pred_vars
    F2_n_vars <- ncol(noMV_pred_vars)
    F2_n_pats<- nrow(noMV_pred_vars)
    #clean
    remove(indx_more80, ncf_pred_vars)
  }else{
    #remove variables
    noMV_pred_vars <- drop_columns(ncf_pred_vars, indx_more80) 
    #keep the dropped variables in a text file
    write( colnames(subset (ncf_pred_vars, select = indx_more80)),
           file = "./More_80p_MV.txt")
    
    #clean
    remove(indx_more80, ncf_pred_vars)
    #keep columns' and rows' numbers
    F2_n_vars <- ncol(noMV_pred_vars)
    F2_n_pats<- nrow(noMV_pred_vars)
    
  }
  
  
  ## ===============> >40% OF MISSING VALUES
  
  #call function
  indx_more40 <- count_vars_moreMV(0.4, noMV_pred_vars,indexes = TRUE)
  
  if(is_empty(indx_more40)){
    filter3MV_all<- noMV_pred_vars
    F3_n_vars<- ncol(filter3MV_all)
    F3_n_pats <- nrow(filter3MV_all)
    #clean
    remove(indx_more40, noMV_pred_vars)
  }else{
    #remove variables
    filter3MV_all <- drop_columns(noMV_pred_vars, indx_more40) 
    #keep the dropped variables in a text file
    write( colnames(subset (noMV_pred_vars, select = indx_more40)),
           file = "./40_70pMV.txt")
    F3_n_vars<- ncol(filter3MV_all)
    F3_n_pats <- nrow(filter3MV_all)
    
    #clean
    remove(indx_more40, noMV_pred_vars)
    
  }
  
  
  
  ###======================== INDIVIDUALS AND MISSING VALUES ==========================###
  
  more_80MV_pats <- perc_MV_patients(0.8, filter3MV_all,ids = TRUE)
  if ( is_empty(more_80MV_pats) ==FALSE){
    #write(unlist(more_80MV_pats$IDs),"./AUT_analysis_data/Pat_more80per_MV.txt")
    filter4 <- filter3MV_all[(-(unlist(more_80MV_pats$Lines))), ]
    F4_n_vars <- ncol(filter4)
    F4_n_pats<- nrow(filter4)
   
     ## Repeat again the variable filtering process
    
    #   More than 80% of missing values
    #call function
    indx_more80 <- count_vars_moreMV(0.8,filter4,indexes = TRUE)
    
    if(is_empty(indx_more80)==FALSE){
      #remove variables
      filter5 <- drop_columns(filter4, indx_more80) 
      #keep the dropped variables in a text file
      write( colnames(subset (filter4, select =  indx_more80)),
             file = "./More_80p_MV_PART2.txt")
      F5_n_vars <- ncol(filter5)
      F5_n_pats<- nrow(filter5)
      #clean
      remove(indx_more80)
      
      #   More than 40% of missing values
      #call function
      indx_more40 <- count_vars_moreMV(0.4, filter5,indexes = TRUE)
      
      if (is_empty(indx_more40)==FALSE){
        #remove variables
        filter6 <- drop_columns(filter5, indx_more40) 
        #keep the dropped variables in a text file
        write( colnames(subset (filter5, select = indx_more40)),
               file = "./40-80pMV_PART2.txt")
        #clean
        remove(indx_more40)
        F6_n_vars <- ncol(filter6)
        F6_n_pats <- nrow(filter6)
  
        write.csv(filter6, OUTPUT_file, row.names = FALSE)
        
      }else{
        #No variables with > 40%NA
        F6_n_vars <- F5_n_vars
        F6_n_pats<- F5_n_pats
        
        write.csv(filter5, OUTPUT_file, row.names = FALSE)
        
      }
    }else{
      #No variables with >80% NA
      F5_n_vars <- F4_n_vars
      F5_n_pats<- F4_n_pats
      F6_n_vars <- F4_n_vars
      F6_n_pats<- F4_n_pats
      
      write.csv(filter4, OUTPUT_file, row.names = FALSE)
      
    }
  }else{
    #No patients removed
    print("There's no any patient with more than 0.8 missing values.")
    F4_n_vars <- F3_n_vars
    F4_n_pats<- F3_n_pats
    F5_n_vars <- F4_n_vars
    F5_n_pats<- F4_n_pats
    F6_n_vars <- F4_n_vars
    F6_n_pats<- F4_n_pats
    
    write.csv(filter3MV_all, OUTPUT_file, row.names = FALSE)
  }
  
  #remove
  
  
  ###==================================================================================###
  ###=================================== GRAPHS =======================================###
  ###==================================================================================###
  
  
  results_filts_vars <- c(ncol(predictive_variables), F1_n_Vars,F2_n_vars,F3_n_vars, F4_n_vars,
                          F5_n_vars,F6_n_vars)
  results_filts_patients <- c(nrow(predictive_variables),F1_n_Pats,F2_n_pats,F3_n_pats,
                              F4_n_pats,F5_n_pats,F6_n_pats)
  filters <- c("All", "F1","F2", "F3", "F4", "F5", "F6")
  
  #Dataframe to build the graph
  df_summary <- data.frame(filters, results_filts_vars, results_filts_patients)
  
  df_summary %>% ggplot(aes(x=filters)) +  
    geom_bar(aes( y = results_filts_patients, colour="Patients"), 
             color="grey", fill ="grey", stat = "identity") +
    geom_label(aes( y = results_filts_patients,label =results_filts_patients))+
    geom_line( aes(y = results_filts_vars), color = "blue", group=1) + 
    geom_point(aes(y= results_filts_vars, colour="Variables"), color="blue")+
    scale_y_continuous(name="Patients", sec.axis = sec_axis(~., name = "Variables"))+
    geom_label(aes( y = results_filts_vars,label =results_filts_vars),vjust=-0.7)+
    xlab("Filters")+
    ggtitle("Results of automatic processing")+
    theme(
      axis.title.x = element_text(size=15),
      title = element_text(size=18),
      plot.title = element_text(hjust = 0.5),
      axis.title.y.left = element_text(size = 15,color = "dark grey"),
      axis.title.y.right = element_text(size=15, color = "blue"))
  ggsave("./AUT_process_changes.jpg",width = 7.7,height = 5.25)
  
  
  
  ###==================================================================================###
  ###=================================== SUMMARY ======================================###
  ###==================================================================================###
  
  
  #text_feat_summary <- paste0("Total of predictive variables from the study: ",
  #             (ncol(predictive_variables)-1),"\n","Variables after remove those non-informatives: ",
  #             (ncol(ncf_pred_vars)-1),"\n","Variables after remove variables with >80% of missing values: ",
  #             (ncol(noMV_pred_vars)-1),"\n" ,
  #             "After removing those with 40-80% of missing values: ",(ncol(filter3MV_all)-1),"\n",
  #             "Number of patients without filtering: ", (nrow(predictive_variables)), "\n", 
  #"Number of patients after removing duplicates (by ID): ", (nrow(filter3MV_all)) ,"\n",
  #"Patients with more than 80% of missing values that were removed: ", length(tmp_index_80),"\n",
  #"Patients with more than 70% of missing values that were NOT removed: ",length(tmp_index_70), sep = "\n")
  
  #write(text_feat_summary,file = "./AUT_analysis_data/SUM_automatic_variable_processing.txt", sep = "\n")
  
  
