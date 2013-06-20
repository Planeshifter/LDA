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
my_corpus_very_small <- my_corpus_small[1:25]
temp <- myLDA(my_corpus_very_small,K=4,alpha=1)
temp$collapsedGibbs(100,0,1)
temp$NichollsMH(100,0,1)