
source("logistic_fit_joint.R") from https://github.com/teisseyrep/Pulogistic
source("functions.R") from https://github.com/teisseyrep/Pulogistic
library('glmnet')
#SET PARAMETERS c:
c = 0.3 #label frequency
n = 1000 #sample size
#n=2000
p<-1200
#p=2000
#p0=20
p0=5
m=1000
rep1<-100;
delta<-0.5*(log(p)/n)^1/3
#delta<-1/n*(log(p)/n)^1/3
lambda<-2*delta
#delta<-0.5*(log(p)/n)^1/3
#lambda<-2*delta

Pred=matrix(0,nrow=rep1, ncol=3)


set.seed(1000)


for(i in 1:rep1) 
{
print(i);

Xtest<-matrix(rnorm(m*p),nrow=m)


beta=1*c(rep(1,p0),rep(0,p-p0))
ind=1:p0
beta0 = 1
x = matrix(0, nrow = n, ncol = p)
for (j in 1:p) {
  x[, j] = rnorm(n, 0, 1)
}
y = numeric(n)
for (i1 in 1:n) {
  lc = beta0+sum(x[i1, ] * beta)
  prob = sigma(lc)
  y[i1] = rbinom(1, 1, prob)
}

#CREATE SURROGATE VARIABLE S:

s = numeric(n)
for (i1 in 1:n) {
  if (y[i1] == 1) {
    s[i1] = rbinom(1, 1, c)
  }
}

beta_true = c(beta0,beta)

###############################logistic model

prtest=exp(cbind(Xtest,1)%*%c(beta,beta0))

Ytest=0
for (j in 1:m){
Ytest[j]=sample(c(1,0),1,prob=c(prtest[j]/(1+prtest[j]),1/(1+prtest[j])))
}



############logistic loss


obj2<-cv.glmnet(x,s,standardize=TRUE, intercept=TRUE,family="binomial", nfolds=10)
obj3<-glmnet(x,s,standardize=TRUE, intercept=TRUE,family="binomial",lambda=lambda)


betasx<-coefficients(obj2,s=obj2$lambda.min)
betasy<-betasx[-1]
nosnik<-which(abs(betasy)>delta)
x1<-x[,nosnik]
s1<-s

betasx1<-coefficients(obj3,s=lambda)
betasy1<-betasx1[-1]
nosnik1<-which(abs(betasy1)>delta)
x11<-x[,nosnik1]
s11<-s




#JOINT METHOD:
par_joint = logistic_fit_joint(x1, s1)$par
beta_joint = par_joint[-length(par_joint)]
c_joint = par_joint[length(par_joint)]


betas12=beta_joint[-1]
betas02=beta_joint[1]


est2=ifelse(Xtest[,nosnik]%*%betas12 + betas02>0,1,0)
Pred[i,1]=sum(est2==Ytest)/m

par_joint1 = logistic_fit_joint(x11, s11)$par
beta_joint1 = par_joint1[-length(par_joint1)]
c_joint1 = par_joint1[length(par_joint1)]

betas13=beta_joint1[-1]
betas03=beta_joint1[1]



est3=ifelse(Xtest[,nosnik1]%*%betas13 + betas03>0,1,0)
Pred[i,2]=sum(est3==Ytest)/m


betas14=beta_true[-1]
betas04=beta_true[1]


est4=ifelse(Xtest%*%betas14 + betas04>0,1,0)
Pred[i,3]=sum(est4==Ytest)/m


}

apply(Pred,2,mean)
apply(Pred,2,sd)









