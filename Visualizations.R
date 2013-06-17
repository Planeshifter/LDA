# TODO: Add comment
# 
# Author: Philipp
###############################################################################


library(ggplot2)

plotChangingZ <- function(LDA)
	{
    require(ggplot2)
    nonchangingZ <- vector()
	z <- LDA$getZList()	
	len <- length(z)-1
	for (i in 1:len)
		{
		current = z[[i+1]]
		old = z[[i]]
		sum_d = 0;
		for (d in 1:LDA$D)
		{
		sum_d = sum_d + sum(!current[[d]]==old[[d]])
		}	
		nonchangingZ[i] = sum_d
		}
	df1 = data.frame(iter=1:len, changingz = nonchangingZ)
	ggplot(df1,aes(x=iter,y=changingz))+geom_line()+labs(x="Iteration",y="Number of z that switched labels")+
			theme_bw()
	}