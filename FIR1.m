clc;
clear all;
close all;
clf;
size1=1000;
iter=200;
blk_size=8;
muu=0.05;
w_sys=[8 4 6 5]';l=length(w_sys);
w1=zeros(l,iter);
for it=1:iter
 x=randn(1,size1);
 %n=randn(1,size1);%noise
 w=zeros(l,1);
 u=zeros(1,l);%regressor vector
 temp=zeros(1,l);
 for i=1:(size1/blk_size)-1%block number is i
 temp=zeros(1,l);
 for r=0:blk_size-1
 u(1,l-2:l)=u(1,1:l-1);%2:4=1:3
 u(1,1)=x(i*blk_size+r);
 desired(i*blk_size+r)=u*w_sys+randn(1,1);
%noise added
 estimate(i*blk_size+r)=u*w;
 err(it,i*blk_size+r)=desired(i*blk_size+r)-estimate(i*blk_size+r);
 temp=temp+err(it,i*blk_size+r).*u;
 end
 w=w+muu.*temp';%(muu'/blk_size)=muu
 end
 w1(:,it)=w;
 clc;
 fprintf('current iteration is %d ',it);
end
w=mean(w1,2);
err_sqr=err.^2;
err_mean=mean(err_sqr,1);
err_max=err_mean./max(err_mean);
err_dB=10*log10(err_max);
plot(err_dB);
title('BLOCK LMS Algorithm');
ylabel('Mean Square Error');
xlabel('No of Iterations');
w_sys
w