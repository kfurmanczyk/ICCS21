
source("logistic_fit_joint.R") # from https://github.com/teisseyrep/Pulogistic
library('glmnet')
library('spls')
library('HiDimDA')
library('caret')
library('AdaSampling')

#DATA SETS
data(lymphoma)
Xdane1<-data.frame(x=lymphoma$x,y=ifelse(lymphoma$y==0,0,1)) #y 0-1
data(prostate)
Xdane2<-data.frame(x=prostate$x,y=prostate$y)
Xdane4<-data.frame(x=AlonDS[,2:2001],y=ifelse(AlonDS[,1]=="colonc",1,0))
data(segmentationData)
Xdane6<-data.frame(x=dhfr[,2:229],y=ifelse(dhfr[,1]=="active",1,0))



#SET PARAMETER c:

c = 0.3 #label frequency
rep1<-100;

Pred=matrix(0,nrow=rep1, ncol=5)

for(i in 1:rep1) 
{
print(i);


Xdane<-Xdane1 # CURRENT DATA SET

l1<-length(Xdane[,1])
sam<-sample(1:l1,floor(0.8*l1),F)
Xtrain<-Xdane[sam,]
Xtest0<-Xdane[-sam,]
Ytrain<-Xtrain$y
Ytest<-Xtest0$y
p1<-length(Xdane[1,])
Xtest<-Xtest0[,-p1]
p<-length(Xdane[1,])-1
k<-p
sam1<-sample(1:p,k,F)
Xtest<-Xtest[,sam1]
n<-l1

lambda<-0.5*(log(k)/n)^1/3
delta<-lambda


#CREATE SURROGATE VARIABLE S:
s0 = numeric(n)
for (i1 in 1:n) {
  if (Xdane$y[i1] == 1) {
    s0[i1] = rbinom(1, 1, c)
  }
}

Xtrain1=Xtrain[,-p1] # zbior predyktorow 
x=Xtrain1[,sam1]
m=length(Ytest)

############logistic loss
s<-s0[sam]
xc<-as.matrix(x)
obj2<-cv.glmnet(xc,s,standardize=TRUE, intercept=TRUE,family="binomial", nfolds=10)
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
w11<-as.matrix(x1)
par_joint = logistic_fit_joint(w11, s1)$par
beta_joint = par_joint[-length(par_joint)]
c_joint = par_joint[length(par_joint)]


betas12=beta_joint[-1]
betas02=beta_joint[1]


est2=ifelse(Xtest[,nosnik]%*%betas12 + betas02>0,1,0)
Pred[i,1]=sum(est2==Ytest)/m


w11<-as.matrix(x11)
par_joint1 = logistic_fit_joint(w11, s11)$par
beta_joint1 = par_joint1[-length(par_joint1)]
c_joint1 = par_joint1[length(par_joint1)]

betas13=beta_joint1[-1]
betas03=beta_joint1[1]


wy<-as.matrix(Xtest[,nosnik1])

est3=ifelse(wy%*%betas13 + betas03>0,1,0)
Pred[i,2]=sum(est3==Ytest)/m


dane=data.frame(s,x)
sx<-s0[-sam]
dane1=data.frame(sx,Xtest)

Ps <- rownames(dane)[which(dane[,1] == 1)]
Ns <- rownames(dane)[which(dane[,1] == 0)]
model1<-adaSample(Ps, Ns, train.mat=dane, test.mat=dane1,classifier = "svm")
model2<-adaSample(Ps, Ns, train.mat=dane, test.mat=dane1,classifier = "knn")
model3<-adaSample(Ps, Ns, train.mat=dane, test.mat=dane1,classifier = "logit")


est1a=ifelse(model1[,"P"]>0.5,1,0)
est2a=ifelse(model2[,"P"]>0.5,1,0)
est3a=ifelse(model3[,"P"]>0.5,1,0)

test=data.frame(Ytest,Xtest)

Pred[i,3]=sum(est1a==Ytest)/m
Pred[i,4]=sum(est2a==Ytest)/m
Pred[i,5]=sum(est3a==Ytest)/m


}

apply(Pred,2,mean)
apply(Pred,2,sd)



