
#######################################################################
##########################  PACKAGES ##################################
#######################################################################
library(ggplot2)
library(tidyverse)
library(gghighlight)
library(jcolors)
library(RColorBrewer) 

########################################################################

plot_regPath_h2o<- function(rp_h2o_object, chosen_lambda =0.02954){
  coef <- as.data.frame(rp_h2o_object$coefficients)
  rm <- which(colnames(coef) == "Intercept")
  coef<-coef[,-rm]
  lambdas <- as.vector(rp_h2o_object$lambdas)
  pos <-1
  nlamb <-nrow(coef)
  df_plot<- as.data.frame(matrix(nrow = nlamb * ncol(coef), ncol = 3))
  colnames(df_plot)<- c("variables_names", "coefficients", "lambda")
  for (row in c(1:nlamb)){
    for (col in c(1:ncol(coef))){
      df_plot[pos,1]<- colnames(coef[col])
      df_plot[pos,2]<- coef[row,col]
      df_plot[pos,3]<- lambdas[row]
      pos<-pos+1
    }
  }
  colourCount <- length(unique(df_plot$variables_names))
  getPalette <- colorRampPalette(brewer.pal(9, "Set1"))
  
  plotito <- ggplot(df_plot, aes(lambda, coefficients,
                                    color = variables_names)) + 
    geom_line() + 
    scale_x_log10() + 
    xlab("Lambda (log scale)") +
    ylab("Coefficients")+
    scale_color_manual(values = getPalette(colourCount))+
    annotate(geom = "point", x = chosen_lambda, y = 0, colour = "black", size = 1) + 
    annotate(geom = "point", x = chosen_lambda, y = 0) + 
    annotate(geom = "text", x = chosen_lambda, y = 0, label = "Lambda chosen",
             hjust = "left",vjust=-1.5)+
    theme(legend.position="none")
  return(plotito)
}

plot_rp <-plot_regPath_h2o(reg_path)
svg("Reg_path_prueba.svg")
plotito 
dev.off()


