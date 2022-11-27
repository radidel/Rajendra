#!/usr/bin/env python
# coding: utf-8

# # Description Of Courswork

# # Problme 1: Noraml Distribution 

# a Monte Carlo simulation is generated to analyse the potential range of outcomes for page view statistics. The page views are expressed in logarithmic format, in accordance with the chosen distribution. A Mo.nte Carlo simulation is a useful tool for predicting future results by calculating a formula multiple times with different random inputs. This is a process you can execute in Excel but it is not simple to do without some VBA or potentially expensive third party plugins. Using numpy and pandas to build a model and generate multiple potential results and analyze them is relatively straightforward. The other added benefit is that analysts can run many scenarios by changing the inputs and can move on to much more sophisticated models in the future if the needs arise. Finally, the results can be shared with non-technical users and facilitate discussions around the uncertainty of the final result.

# Library 

# In[40]:


import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy.stats import uniform
from scipy.stats import expon


#  Generating random numbers from a standard normal distribution

# Sometimes, we like to produce the same random numbers repeatedly. For example, when a professor is explaining how to estimate the mean, standard deviation, skewness, and kurtosis of five random numbers, it is a good idea that students could generate exactly the same values as their instructor. Another example would be that when we are debugging our Python program to simulate a stock’s movements, we might prefer to have the same intermediate numbers. For such cases, we use the seed() function as follows:

# In[20]:


sp.random.seed(12345)
x=sp.random.normal(0.05,0.1,50) 
print(x[0:5]) 


# A histogram is used intensively in the process of analyzing the properties of datasets. To generate a histogram for a set of random values drawn from a normal distribution with specified mean and standard deviation, we have the following code:

# In[34]:


import scipy as sp
import matplotlib.pyplot as plt
sp.random.seed(12345)
x=sp.random.normal(0.08,0.2,1000)
plt.hist(x, 15, density=True)
plt.show()


# # Problme 2:
#     

# 1. PDF:
#     	f(x)={█((λe^(-λx))/(e^(-λ)-e^(-2λ) ),if x∈[1,2]@0,otherewise)┤
#               Now ∫_(-∞)^∞▒f(x)dx=∫_1^2▒〖(λe^(-λx))/(e^(-λ)-e^(-2λ) ) dx〗+0
#                                           =λ/(e^(-λ)-e^(-2λ) ) [e^(-λx)/(-λ)]_1^2
#                          =(-1)/(e^(-λ)-e^(-2λ) ) (e^(-2λ)-e^(-λ) )
#                          =1/(e^(-λ)-e^(-2λ) ) (e^(-λ)-e^(-2λ) )=1
# Since ∫_(-∞)^∞▒f(x)dx=1
# So f(x) is pdf
# 

# 2. Inverse Transfer Method:

# In simulation theory, generating random variables become one of the most important “building block”, where these random variables are mostly generated from Uniform distributed random variable. One of the methods that can be used to generate the random variables is the Inverse Transform method. In this article, I will show you how to generate random variables (both discrete and continuous case) using the Inverse Transform method in Python.

# In[41]:


def exponential_inverse_trans(n=1,mean=1):
    U=uniform.rvs(size=n)
    X=-mean*np.log(1-U)
    actual=expon.rvs(size=n,scale=mean)
    
    plt.figure(figsize=(12,9))
    plt.hist(X, bins=50, alpha=0.5, label="Generated r.v.")
    plt.hist(actual, bins=50, alpha=0.5, label="Actual r.v.")
    plt.title("Generated vs Actual %i Exponential Random Variables" %n)
    plt.legend()
    plt.show()
    return X


# In[42]:


cont_example1=exponential_inverse_trans(n=100,mean=4)
cont_example2=exponential_inverse_trans(n=500,mean=4)
cont_example3=exponential_inverse_trans(n=1000,mean=4)


# 3. Sample gerenrate using von neumann's alogithims:  sampling is a means of generating random numbers that belong to a particular distribution.

# So we can see that the reason this is so straightforward is that we get samples according to the function by simply throwing away the right number of samples when the function has a smaller value. In our function, this means if we get a small 
# x
#  value, we’d normally keep the sample (and indeed the distribution is pretty flat for 
# x
# <
# 0.5
# ), but for values close to 
# x
# =
# 1
# , we’d throw them out most of the time!
# 
# Let’s write a very simple function that gets us a single sample using this method. See how for rejection sampling, we have to specify an initial box to pick from (the x bounds and the maximum y bound). This does make the algorithm unsuited for distributions that have an infinite range of 
# x
#  values, a pity!

# In[49]:


def sample(function, xmin=0, xmax=1, ymax=1.2):
    while True:
        x = np.random.uniform(low=xmin, high=xmax)
        y = np.random.uniform(low=0, high=ymax)
        if y < function(x):
            return x


# Let us use this inefficient algorithm to generate a ten thousand samples, and plot their distribution to make sure it looks like its working

# In[50]:


samps = [sample(f) for i in range(10000)]

plt.plot(xs, ys, label="Function")
plt.hist(samps, density=True, alpha=0.2, label="Sample distribution")
plt.xlim(0, 1), plt.ylim(0, 1.4), plt.xlabel("x"), plt.ylabel("y"), plt.legend();


# In[51]:


def batch_sample(function, num_samples, xmin=0, xmax=1, ymax=1.2, batch=1000):
    samples = []
    while len(samples) < num_samples:
        x = np.random.uniform(low=xmin, high=xmax, size=batch)
        y = np.random.uniform(low=0, high=ymax, size=batch)
        samples += x[y < function(x)].tolist()
    return samples[:num_samples]

samps = batch_sample(f, 10000)

plt.plot(xs, ys, label="Function")
plt.hist(samps, density=True, alpha=0.2, label="Sample distribution")
plt.xlim(0, 1), plt.ylim(0, 1.4), plt.xlabel("x"), plt.ylabel("f(x)"), plt.legend();


# In[52]:


def gauss(x):
    return np.exp(-np.pi * x**2)

xs = np.linspace(-10, 10, 1000)
ys = gauss(xs)

plt.plot(xs, ys)
plt.fill_between(xs, ys, 0, alpha=0.2)
plt.xlabel("x"), plt.ylabel("f(x)");


# In[53]:


def batch_sample_2(function, num_samples, xmin=-10, xmax=10, ymax=1):
    x = np.random.uniform(low=xmin, high=xmax, size=num_samples)
    y = np.random.uniform(low=0, high=ymax, size=num_samples)
    passed = (y < function(x)).astype(int)
    return x, y, passed

x, y, passed = batch_sample_2(gauss, 10000)

plt.plot(xs, ys)
plt.fill_between(xs, ys, 0, alpha=0.2)
plt.scatter(x, y, c=passed, cmap="RdYlGn", vmin=-0.1, vmax=1.1, lw=0, s=2)
plt.xlabel("x"), plt.ylabel("y"), plt.xlim(-10, 10), plt.ylim(0, 1);

print(f"Efficiency is only {passed.mean() * 100:0.1f}%")


# 4.prefered method for gernrating a sample from f is inverse tranfrom method. becuase von neumenns have less efficiency 

# # Problme 3: Black-Scholes

# The Black-Scholes model was first introduced by Fischer Black and Myron Scholes in 1973 in the paper "The Pricing of Options and Corporate Liabilities". Since being published, the model has become a widely used tool by investors and is still regarded as one of the best ways to determine fair prices of options.
# 
# The purpose of the model is to determine the price of a vanilla European call and put options (option that can only be exercised at the end of its maturity) based on price variation over time and assuming the asset has a lognormal distribution.

# In Black-Scholes formulas, the following parameters are defined.
# 
# S, the spot price of the asset at time t
# T, the maturity of the option. Time to maturity is defined as T−t
# K, strike price of the option
# r, the risk-free interest rate, assumed to be constant between t and T
# σ, volatility of underlying asset, the standard deviation of the asset returns

# In[54]:


import numpy as np
import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing
init_printing()


# In[55]:


def euro_vanilla_call(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call


# In[56]:


euro_vanilla_call(50, 100, 1, 0.05, 0.25)


# In[57]:


def euro_vanilla_put(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
    return put
euro_vanilla_put(50, 100, 1, 0.05, 0.25)


# The next function can be called with 'call' or 'put' for the option parameter to calculate the desired option

# In[58]:


def euro_vanilla(S, K, T, r, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        
    return result
euro_vanilla(50, 100, 1, 0.05, 0.25, option = 'put')


# Sympy implementation for Exact Results

# In[59]:


def euro_call_sym(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    N = Normal('x', 0.0, 1.0)
    
    d1 = (sy.ln(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    d2 = (sy.ln(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    
    call = (S * cdf(N)(d1) - K * sy.exp(-r * T) * cdf(N)(d2))
    
    return call
euro_call_sym(50, 100, 1, 0.05, 0.25)


# In[60]:


def euro_put_sym(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    N = systats.Normal(0.0, 1.0)
    
    d1 = (sy.ln(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    d2 = (sy.ln(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    
    put = (K * sy.exp(-r * T) * N.cdf(-d2) - S * N.cdf(-d1))
    
    return put


# In[61]:


def sym_euro_vanilla(S, K, T, r, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    N = Normal('x', 0.0, 1.0)
    
    d1 = (sy.ln(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    d2 = (sy.ln(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
    
    if option == 'call':
        result = (S * cdf(N)(d1) - K * sy.exp(-r * T) * cdf(N)(d2))
    if option == 'put':
        result = (K * sy.exp(-r * T) * cdf(N)(-d2) - S * cdf(N)(-d1))
        
    return result


# In[62]:


sym_euro_vanilla(50, 100, 1, 0.05, 0.25, option = 'put')


# In[ ]:





# # Problem 4: continuous time stochastic process

# In[65]:


from symbulate import *
get_ipython().run_line_magic('matplotlib', 'inline')


# # Random processes
# A random process (a.k.a. # stochastic process) is an indexed collection of random variables defined on some probability space. The index often represents "time", which can be either discrete or continuous.
# 
# A discrete time stochastic process is a collection of countably many random variables, e.g.  Xn  for  n=0,1,2,… . For each outcome in the probability space, the outcome of a discrete time stochastic process is a sequence (in  n ). (Remember Python starts indexing at 0. The zero-based-index is often natural in stochastic process contexts in which there is a time 0, i.e.  X0  is the initial value of the process.)
# A continuous time stochastic process is a collection of uncountably many random variables, e.g.  Xt  for  t≥0 . For each outcome in the probability space, the outcome of a discrete time stochastic process is a function (a.k.a. sample path) (of  t ).

# # RandomProcess and TimeIndex
# Much like RV, a RandomProcess can be defined on a ProbabilitySpace. For a RandomProcess, however, the TimeIndex must also be specified. TimeIndex takes a single parameter, the sampling frequency fs. While many values of fs are allowed, the two most common inputs for fs are
# 
# TimeIndex(fs=1), for a discrete time process  Xn,n=0,1,2,… .
# TimeIndex(fs=inf), for a continuous time process  X(t),t≥0 .

# # Defining a RandomProcess explicity as a function of time
# A random variable is a function  X  which maps an outcome  ω  in a probability space  Ω  to a real value  X(ω) . Similarly, a random process is a function  X  which maps an outcome  ω  and a time  t  in the time index set to the process value at that time  X(ω,t) . In some situations, the function defining the random process can be specified explicitly.
# 
# Example. Let  X(t)=A+Bt,t≥0  where  A  and  B  are independent with  A∼  Bernoulli(0.9) and  B∼  Bernoulli(0.7). In this case, there are only 4 possible sample paths.
# 
# X(t)=0 , when  A=0,B=0 , which occurs with probability  0.03 
# X(t)=1 , when  A=1,B=0 , which occurs with probability  0.27 
# X(t)=t , when  A=0,B=1 , which occurs with probability  0.07 
# X(t)=1+t , when  A=1,B=1 , which occurs with probability  0.63 
# The following code defines a RandomProcess X by first defining an appropriate function f. Note that an outcome in the probability space consists of an  A,B  pair, represented as  ω0  and  ω1  in the function. A RandomProcess is then defined by specifying: the probability space, the time index set, and the  X(ω,t)  function.

# In[ ]:


def f(omega, t):
    return omega[0] + omega[1] * t

X = RandomProcess(Bernoulli(0.9) * Bernoulli(0.7), TimeIndex(fs=inf), f)
X.sim(100).plot(tmin=0, tmax=2)


# # Process values at particular time points
# The value  X(t)  (or  Xn ) of a stochastic process at any particular point in time  t  (or  n ) is a random variable. These random variables can be accessed using brackets []. Note that the value inside the brackets represents time  t  or  n . Many of the commands in the previous sections (Random variables, Multiple random variables, Conditioning) are useful when simulating stochastic processes.
# 
# Example. Let  X(t)=A+Bt,t≥0  where  A  and  B  are independent with  A∼  Bernoulli(0.9) and  B∼  Bernoulli(0.7).
# 
# Find the distribution of  X(1.5) , the process value at time  t=1.5 .

# In[ ]:


def f(omega, t):
    return omega[0] * t + omega[1]

X = RandomProcess(Bernoulli(0.9) * Bernoulli(0.7), TimeIndex(fs=inf), f)

X[1.5].sim(10000).plot()


# # Mean function
# The mean function of a stochastic process  X(t)  is a deterministic function which maps  t  to  E(X(t)) . The mean function can be estimated and plotted by simulating many sample paths of the process and using .mean().

# In[ ]:


paths = X.sim(1000)
plot(paths)
plot(paths.mean(), 'r')


# In[ ]:


#The variance function maps  t  to  Var(X(t)) ; similarly for the standard deviation function. These functions can be used to give error bands about the mean function.

# This illustrates the functionality, but is not an appropriate example for +/- 2SD
plot(paths)
paths.mean().plot('--')
(paths.mean() + 2 * paths.sd()).plot('--')
(paths.mean() - 2 * paths.sd()).plot('--')


# # Defining a RandomProcess incrementally
# There are few situations like the linear process in the example above in which the random process can be expressed explicitly as a function of the probability space outcome and the time value. More commonly, random processes are often defined incrementally, by specifying the next value of the process given the previous value.
# 
# Example. At each point in time  n=0,1,2,…  a certain type of "event" either occurs or not. Suppose the probability that the event occurs at any particular time is  p=0.5 , and occurrences are independent from time to time. Let  Zn=1  if an event occurs at time  n , and  Zn=0  otherwise. Then  Z0,Z1,Z2,…  is a Bernoulli process. In a Bernoulli process, let  Xn  count the number of events that have occurred up to and including time  n , starting with 0 events at time 0. Since  Zn+1=1  if an event occurs at time  n+1  and  Zn+1=0  otherwise,  Xn+1=Xn+Zn+1 .
# 
# The following code defines the random process  X . The probability space corresponds to the independent Bernoulli random variables; note that inf allows for infinitely many values. Also notice how the process is defined incrementally through  Xn+1=Xn+Zn+1 .

# In[ ]:


P = Bernoulli(0.5)**inf
Z = RV(P)
X = RandomProcess(P, TimeIndex(fs=1))

X[0] =  0
for n in range(100):
    X[n+1] = X[n] + Z[n+1]


# In[ ]:


#The above code defines a random process incrementally. Once a RandomProcess is defined, it can be manipulated the same way, regardless of how it is defined.

X.sim(1).plot(alpha = 1)

