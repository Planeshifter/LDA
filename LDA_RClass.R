# TODO: Add comment
# 
# Author: Philipp
###############################################################################

library(Rcpp)

myLDA_class <- setRefClass("myLDA_class",
		fields=list(
				corpus = "ANY",
				w = "list",  # list of length d which contains the words for each doc as a list
				w_num = "list", # list of length d which contains the word indices instead of the actual words
				z = "list", # list of the hidden topic labels for each word in each doc
				z_res = "list", 
				z_list = "list", # list of sampled z's as returned from Gibbs sampler
				alpha = "numeric", # hyper-parameter for Dirichlet distr of theta
				beta = "numeric", # hyper-parameter of Dirichlet distr of phi
				K = "numeric",   # K number of Topics 
				W = "numeric",  # W number of unique Words
				D = "numeric", # D number of Documents
				Vocabulary = "character", # Character Vector of all unique words
				nw = "matrix", # nw_ij number of word i assigned to topic j
				nd = "matrix", # nd_dj number of times topic j appears in doc d
				nw_sum = "numeric", # nw_sum_j total number of words assigned to topic j
				nd_sum = "numeric" # doc length of doc d

		),
		methods=list(
				preProcessText = function()
				{
					require(tm)                 
					.self$corpus <-  tm_map(.self$corpus, tolower)
					.self$corpus <-  tm_map(.self$corpus, removeWords, words=stopwords())
					.self$corpus <-  tm_map(.self$corpus, removeNumbers)
					.self$corpus <-  tm_map(.self$corpus, removePunctuation)
					# .self$corpus <-  tm_map(.self$corpus, stemDocument)
					.self$corpus <-  tm_map(.self$corpus, stripWhitespace)
				},
				
				initSampling = function()
				{
					# nw_ij number of times word wi is assigned to topic j
					# nw_sum_j number of words assigned to topic j
					# nd_jd number of times topic j appears in doc d
					# nd_sum_d number of topics in document di 
					alpha <- .self$alpha
					beta <- .self$beta
					z <- .self$z
					w <- .self$w_num
					K <- .self$K
					D <- .self$D
					W <- .self$W
					nd <- .self$nd
					nw <- .self$nw
					
					nd_sum <- rep(0,times=D)
					nw_sum <- rep(0,times=K) 
					
					for (d in 1:D)
					{
						
						# initialize random topics
						z[[d]] <- sample(x=K,size=length(w[[d]]), replace=TRUE)
						# get doc length
						nd_sum[d] <- length(z[[d]])
						
						for (i in 1:nd_sum[d])
						{
							wtemp <- w[[d]][i]
							topic <- z[[d]][i]
							
							# number of instances of word i assigned to topic j
							nw[wtemp,topic] <- nw[wtemp,topic] + 1;
							# number of words in document i assigned to topic j
							nd[d,topic]  <- nd[d,topic] + 1;
							#  total number of words assigned to topic j           
							nw_sum[topic] <- nw_sum[topic] + 1;
						}
					}     
					
					.self$nd_sum <- nd_sum
					.self$nw_sum <- nw_sum
					.self$nd <- nd
					.self$nw <- nw
					.self$z <- z                     
					
				},
										
				extractWords = function()
				{                           
					w <- lapply(.self$corpus, function(x) strsplit(x," +")[[1]])
					w <- lapply(w,function(x){x[!x%in%""]})
					w <- lapply(w, function(x) x[nchar(x)>3])
					z <- vector("list", .self$D)  
					
					.self$w <- w
					.self$z <- z
					.self$W <- length(unique(unlist(w)))
					.self$Vocabulary <- unique(unlist(w))
					
					nw <- matrix(0,nrow=W,ncol=K)
					.self$nw <- nw
					
					nd <- matrix(0,nrow=D,ncol=K)                   
					.self$nd <- nd
				},
								
				w_to_numeric = function()
				{
					w <- .self$w
					w_num <- lapply(w, function(x) match(x,.self$Vocabulary)) 
					.self$w_num <- w_num
				},
								
				initialize = function(corpus, alpha, beta, K, D, old_myLDA = NULL)
				{
					
					if (is.null(old_myLDA))
					{
						.self$corpus <- corpus
						.self$alpha <- alpha
						.self$beta <- beta
						.self$K <- K
						.self$D <- D
						.self$preProcessText()  
						.self$extractWords()
						.self$w_to_numeric()
						.self$initSampling()
					}
					else
					{
						.self$corpus <- old_myLDA$corpus
						.self$alpha <- old_myLDA$alpha
						.self$beta <- old_myLDA$beta
						.self$K <- old_myLDA$K
						.self$D <- old_myLDA$D 
						.self$W <- old_myLDA$W
						.self$Vocabulary <- old_myLDA$Vocabulary
						
						.self$nd <- old_myLDA$nd
						.self$nw <- old_myLDA$nw
						.self$nw_sum <- old_myLDA$nw_sum
						.self$nd_sum <- old_myLDA$nd_sum
												
						.self$w <- old_myLDA$w
						.self$w_num <- old_myLDA$w_num
						.self$z <- old_myLDA$z
						
					}
				}
		)
)
myLDA <- function(artList, alpha=NULL, beta=0.1, K=20)
{
	
	if (is.null(alpha)) alpha <- 50 / K
	
	require(newscrapeR)
	D <- length(artList)
	corpus <- ArtListToCorpus(artList)
	
	intermediate = new("myLDA_class", corpus=corpus, alpha=alpha, beta=beta, K=K,D=D);
	LDA <- LDA_module$LDA
	lda <- new(LDA,intermediate)
	return(lda)
}


migrate.myLDA <- function(myLDAobj)
{
	ret <- myLDA_class$new(old_myLDA = myLDAobj)
	ret
}


