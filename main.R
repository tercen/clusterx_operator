library(tercen)
library(dplyr)
library(ClusterX)
library(reshape2)
 
ctx <-tercenCtx()

seed <- NULL
if(!ctx$op.value('seed') < 0) seed <- as.integer(ctx$op.value('seed'))

set.seed(seed)

data <- ctx %>% 
  select(.ci, .ri, .y) %>% 
  reshape2::acast(.ci ~ .ri, value.var='.y', fill=NaN, fun.aggregate=mean) 

colnames(data) <- paste('c', colnames(data), sep='')

dimReduction <- NULL
if (ctx$op.value('dimReduction') != "NULL") dimReduction <- ctx$op.value('dimReduction')

dataX <- ClusterX(data, dimReduction=dimReduction, outDim=as.integer(ctx$op.value('outDim')))

data.frame(.ci = seq(from=0,to=length(dataX$cluster)-1),
           cluster=paste0("cluster",dataX$cluster)) %>%
  ctx$addNamespace() %>%
  ctx$save()