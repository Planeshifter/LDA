# TODO: Add comment
# 
# Author: Philipp
###############################################################################

setwd("C:/Users/Philipp/Documents/GitHub/LDA")
rm(list=ls())
library(Rcpp)
sourceCpp("LDA_class.cpp")
source("LDA_RClass.R")
load("my_corpus.RData")
temp <- myLDA(my_corpus_small,K=20,alpha=1)
temp$collapsedGibbs(100,0,1)