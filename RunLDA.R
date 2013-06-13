# TODO: Add comment
# 
# Author: Philipp
###############################################################################

rm(list=ls())
library(Rcpp)
sourceCpp("Custom LDA/LDA_class.cpp")
source("Custom LDA/LDA_RClass.R")
load("my_corpus.RData")
temp <- myLDA(my_corpus_small,K=20)
