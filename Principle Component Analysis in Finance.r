library(xts)
library(ggplot2)
library(reshape2)
library(ggthemes)
library(plotly)

df<- read.csv('48_Industry_Portfolios.csv',skip=11,stringsAsFactors=F)
df<- df[1:1098,]
df<- data.frame(lapply(df,as.numeric))
df<- df[df[,1]>200000& df[,1]< 201800,]
rownames(df) <- df$X
df<- df[,-1]/100


n<- dim(df)[1]
k<- dim(df)[2]

head(df)

pca<- prcomp(df,scale.=T,center=T)  
omega<- cov(df) # calculate covariance matrix of X
P<- pca$rotation   #eigenvector
lambda<- pca$sdev^2    # eigenvalue

Pnorm<- apply(P,2,function(x) x/sum(x))  #normalize weights
df_price<- apply(df+1,2,cumprod) #temp
eigen.port<- df_price %*% Pnorm   #calculate eigen portfolios

options(repr.plot.width=7, repr.plot.height=3.5)
par(mfrow=c(2,5),mar=c(3.5,2,2,2))
for (i in 1:10){
    barplot(Pnorm[,i],main=colnames(P)[i],col='#3a539b',border = NA)
    #$text(plt, par("usr")[3], labels = rownames(P), srt = 45, adj = c(1.1,1.1), xpd = TRUE, cex=0.7)
}

plotdf1<- melt(data.frame(df_price,date=as.yearmon(rownames(df),format='%Y%m')),id='date')
plotdf2<- melt(data.frame(eigen.port[,c('PC1','PC2','PC3','PC4')],date=as.yearmon(rownames(df_price),format='%Y%m')),id='date')

head(plotdf1)

options(repr.plot.width=7, repr.plot.height=3)
g <- ggplot(plotdf1,aes(x=date,y=value,group=variable),size=2)+ 
    geom_line(alpha=0.1)+ xlab('')+ylab('')+
    geom_line(data=plotdf2,alpha=0.8,aes(x=date,y=value,col=variable,group=variable)) +
    theme_bw()
ggplotly(g)

frac<- rep(0,k)
for (i in 1:k){
    frac[i]<- sum(lambda[1:i])/sum(lambda)
}
options(repr.plot.width=5, repr.plot.height=3)
p = qplot(1:k,frac,geom='line',ylab='Fraction',xlab='Number of Eigen-Portfolios')+geom_point()+theme_bw()
ggplotly(p)

m<- 4
betahat<- (P %*% diag(lambda))[,1:m]

Dhat<- diag(diag(omega - betahat %*% t(betahat)))

DhatInv<- solve(Dhat)
A<- solve(t(betahat)%*%DhatInv%*%betahat)
fhat<- matrix(0,nrow=dim(df)[1],ncol=m)
for (i in 1:dim(df)[1]){
  fhat[i,]<- t(A%*%(t(betahat)%*%DhatInv%*%t(as.matrix(df[i,]))))
}

factorprice<- as.data.frame(apply(fhat+1,2,cumprod),row.names = as.Date(as.yearmon(rownames(df),format='%Y%m')))
colnames(factorprice) <- c('Factor1','Factor2','Factor3','Factor4')
factorprice$date <- as.Date(rownames(factorprice),format='%Y-%m-%d')
options(repr.plot.width=5, repr.plot.height=6)
#matplot(factorprice,type='l')


plot_ly(factorprice, x = ~date, y = ~Factor1, name = 'Factor1', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~Factor2, name = 'Factor2', mode = 'lines') %>%
  add_trace(y = ~Factor3, name = 'Factor3', mode = 'lines') %>%
  add_trace(y = ~Factor4, name = 'Factor4', mode = 'lines') 

