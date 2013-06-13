# TODO: Add comment
# 
# Author: Philipp
###############################################################################


library(ggplot2)

plotChangingZ(LDA)
	{
    nonchangingZ <- vector()
	phi <- LDA$getPhiList()	
	for (i in 2:length(phi))
		{
		nonchangingZ <- sum(phi[i]=phi[i-1])	
		}
	}