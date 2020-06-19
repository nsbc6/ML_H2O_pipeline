# This function performs imputation with or without previous clustering previous to machine learning techniques:
# - Continuous variables: Median
# - Discrete: Majority voting

#Users have to:
# 1- Set training and test datasests
# 2- If you want previous clustering, the tecnique (cl_method = "kmeans" or "hierarchicla").
#    Checking best options. It is advisable the use of clValid to test how many clusters will provide the 
#    best quality and the technique. Altough, you can explore if data would have subgroups of course.
# 3- Names of continuous and discrete (factors) variables in your data.
# 4- If clustering, how many clusters you want.


#Due to performance, some variables will be removed because: A) NA is the majority class or B) when cluster is
#done, only NA form one variable could be grouped.
# Their names will be kept into a .txt in working directory.
# Clustering functions have previous set.seed(1234) 





imputation<- function(train_withNA,
                      test_withNA,
                      clusters = "Yes",
                      cl_method = "kmeans",
                      continuous, factors, name,
                      centers = 2){
  
  ###### INTERN FUNCTIONS
  majority_voting <- function (x, ...){
    counts <- as.data.frame(table(x))
    counts <- as.data.frame(apply(counts,2,as.numeric))
    if(ncol(counts) == 1){
      major_num <- counts[1,1]
    } else {
      major_num <- counts[which( counts[,2] == max(as.numeric(counts[,2]))),1]
    }
    return(major_num)
  }
  
  clust_meth_imp<- function(data_no_NA, method = "kmeans", centers){
    if("kmeans" %in% method){
      set.seed(1234)
      km_r <- kmeans(data_no_NA,centers)
      return(km_r)
    }
    
    if("hierarchical" %in% method){
      set.seed(1234)
      h_clust_complete <- hclust(dist(data_no_NA))
      hclus_r <- cutree(h_clust_complete, centers)
      return(hclus_r)
    }
  }
  
  ####### INTERN FUNCTIONS ENDS
  print(".")
  
  ###### TRAIN
  
  
  
  ###### CLUSTERING
  
  if (clusters == "Yes"){
    
    idx <- which(colMeans(is.na(train_withNA))==0)
    vars_no_NA<-train_withNA[,idx]
    
    #set.seed(123)
    #### K-MEANS
    if("kmeans" %in% cl_method){
      imp <- clust_meth_imp(vars_no_NA, method = "kmeans",centers)
      for (i in 1:centers){
        assign(paste0("pats_clus_tr_", i), cbind(vars_no_NA[which(imp$cluster == i),],
                                                 train_withNA[which(imp$cluster == i),-idx]))
      }
      
      list_df<- mget(ls(pattern = "pats_clus_tr_"))
      
    }
    
    #### HIERARCHICAL CLUSTERING
    
    if("hierarchical" %in% cl_method){
      imp <- clust_meth_imp(vars_no_NA, method = "hierarchical",centers)
      print(length(imp))
      for (number in 1:centers){
        assign(paste0("pats_clus_tr_", number), cbind(vars_no_NA[which(imp == number),],
                                                      train_withNA[which(imp == number),-idx]))
      }
      
      list_df<- mget(ls(pattern = "pats_clus_tr_"))
      
    }
    
    #Common path clusters
    l_no_NA<- list()
    for(df in list_df){
      df <-as.data.frame(df)
      vars_na <-which(colMeans(is.na(df))> 0)
      for (var in colnames(df[vars_na])){
        #print(var)
        indx_var <-which(colnames(df)==var)
        vector_var <- df[,indx_var]
        if(var %in% continuous){
          vector_var[is.na(vector_var)] <- median(as.numeric(vector_var), na.rm = T)
          df[[var]]<-vector_var
        }
        if(var %in% factors){
          vector_var[is.na(vector_var)] <- majority_voting(vector_var, na.rm = T)
          df[[var]]<-vector_var
        }
      }
      l_no_NA<- list.append(l_no_NA, df)
    }
    
    train_data_imputed <- bind_rows(l_no_NA, .id = "column_label")
    train_data_imputed$column_label<-NULL
  }
  
  
  ### IMPUTATION WITHOUT CLUSTERS
  if (clusters == "No"){
    train_data_imputed <- train_withNA
    vars_na <-which(colMeans(is.na(train_data_imputed))> 0)
    for (var in colnames(train_data_imputed[vars_na])){
      #print(var)
      indx_var <-which(colnames(train_data_imputed)==var)
      vector_var <- train_data_imputed[,indx_var]
      if(var %in% continuous){
        vector_var[is.na(vector_var)] <- median(as.numeric(vector_var), na.rm = T)
        train_data_imputed[[var]]<-vector_var
      }
      if(var %in% factors){
        vector_var[is.na(vector_var)] <- majority_voting(vector_var, na.rm = T)
        train_data_imputed[[var]]<-vector_var
      }
    }
  }
  
  idx_to_rm<-vector()
  idx_to_rm<- which(colMeans(is.na(train_data_imputed))>0)
  
  if(!(is_empty(idx_to_rm))){
    train_data_imputed<-train_data_imputed[,-idx_to_rm]
    write(names(idx_to_rm), paste0("vars_rm_train",name,".txt"))
    print("Columns were removed from train")
  }
  
  print(".")
  
  
  ############################## TEST
  
  idx <- which(colMeans(is.na(test_withNA))==0)
  vars_no_NA<-test_withNA[,idx]
  
  ###### CLUSTERING
  if (clusters == "Yes"){
    ### K-means
    if("kmeans" %in% cl_method){
      imp <- clust_meth_imp(vars_no_NA, method = "kmeans",centers)
      for (i in 1:centers){
        assign(paste0("pats_clus_te_", i), cbind(vars_no_NA[which(imp$cluster == i),],
                                                 test_withNA[which(imp$cluster == i),-idx]))
      }
      list_df<- mget(ls(pattern = "pats_clus_te_"))
    }
    
    #### Hierarchical clustering
    if("hierarchical" %in% cl_method){
      imp <- clust_meth_imp(vars_no_NA, method = "hierarchical",centers)
      
      for (number in 1:centers){
        assign(paste0("pats_clus_te_", number), cbind(vars_no_NA[which(imp == number),],
                                                      test_withNA[which(imp == number),-idx]))
      }
      list_df<- mget(ls(pattern = "pats_clus_te_"))
    }
    
    # COMMON
    l_no_NA<- list()
    for(df in list_df){
      df <-as.data.frame(df)
      vars_na <-which(colMeans(is.na(df))> 0)
      for (var in colnames(df[vars_na])){
        #print(var)
        indx_var <-which(colnames(df)==var)
        vector_var <- df[,indx_var]
        if(var %in% continuous){
          vector_var[is.na(vector_var)] <- median(as.numeric(vector_var), na.rm = T)
          df[[var]]<-vector_var
        }
        if(var %in% factors){
          vector_var[is.na(vector_var)] <- majority_voting(vector_var, na.rm = T)
          df[[var]]<-vector_var
        }
      }
      l_no_NA<- list.append(l_no_NA, df)
    }
    test_data_imputed <- bind_rows(l_no_NA, .id = "column_label")
    test_data_imputed$column_label<-NULL
  }
  
  #NO CLUSTERS
  if (clusters == "No"){
    test_data_imputed <- test_withNA
    for (var in colnames(test_data_imputed[vars_na])){
      #print(var)
      indx_var <-which(colnames(test_data_imputed)==var)
      vector_var <- test_data_imputed[,indx_var]
      if(var %in% continuous){
        vector_var[is.na(vector_var)] <- median(as.numeric(vector_var), na.rm = T)
        test_data_imputed[[var]]<-vector_var
      }
      if(var %in% factors){
        vector_var[is.na(vector_var)] <- majority_voting(vector_var, na.rm = T)
        test_data_imputed[[var]]<-vector_var
      }
    }
  }
  
  idx_to_rm_2<-vector()
  idx_to_rm_2<- which(colMeans(is.na(test_data_imputed))>0)
  
  if(!(is_empty(idx_to_rm_2))){
    test_data_imputed<-test_data_imputed[,-idx_to_rm_2]
    
    write(names(idx_to_rm_2), paste0("vars_rm_test",name,".txt"))
  }
  
  print(".")
  
  #### Checking
  
  if(!(ncol(train_data_imputed == ncol(test_data_imputed)))){
    print("Warning: Different column names between training and test sets.")
  }
  print(".")
  
  
  #### Results
  return(list(train_data_imputed, test_data_imputed))
}
