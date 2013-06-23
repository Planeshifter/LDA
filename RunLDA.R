# TODO: Add comment
# 
# Author: Philipp
###############################################################################

setwd("C:/Users/Philipp/Documents/GitHub/LDA")
rm(list=ls())
Sys.setenv("PKG_LIBS"="C:/Users/Philipp/boost_1_53_0/stage/lib/libboost_regex-vc110-mt-1_53.lib")
library(Rcpp)
Rcpp:sourceCpp("LDA_class.cpp")
Rcpp::sourceCpp('RegExps.cpp')

source("LDA_RClass.R")
load("my_corpus.RData")
my_corpus_very_small <- my_corpus_small[1:50]
temp <- myLDA(my_corpus_very_small,K=4,alpha=1)
temp$collapsedGibbs(100,0,1)
temp$NichollsMH(100,0,1)
temp$LangevinMHSampling(1000,100,10)