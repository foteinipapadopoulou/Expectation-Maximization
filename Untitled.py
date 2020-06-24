#!/usr/bin/env python
# coding: utf-8

# ## Μεγιστοποίηση πιθανοφάνειας μίξης Gaussian κατανομών 
# 
# 

# In[2]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
from PIL import Image


# Η εικόνα που θα εισάγουμε ως input είναι ένας πίνακας που το κάθε στοιχείο είναι ένα pixel

# In[3]:


image=img.imread('im.jpg')
plt.imshow(image)
array=np.array(image)
array.shape


# Αλλάζουμε τις διαστάσεις του πίνακα σε 690*550,3

# In[4]:


new_img = array.reshape((array.shape[0]*array.shape[1]), array.shape[2])
new_img=new_img/255



# ### Αρχικοποίηση των παραμέτρων : σk^2,πk,μk

# In[5]:


def initialize_mean_cov_pi(X,k):
    N=X.shape[0]
    D=X.shape[1]
    
    mean=np.array([np.mean(X, axis=0),]*k) + np.random.randn(k,D)
    print ("Dimensions mean:",mean.shape)
    print(mean)
  
    cov_2=np.zeros(k)
    for k_ in range(k):
        sum1=0
        for n in range(N):
            for d in range(D):
                sum1+=np.square(X[n][d]-mean[k_][d])
        cov_2[k_]=sum1/N
    
    
    print ("Dimensions cov_2:",cov_2.shape)
    print(cov_2)
    
    a = []
    pi = []
    for i in range(k):
        a.append(np.random.uniform(np.min(X),np.max(X)))
    sum_a = sum(a)
    for x in range(k):
        pi.append(a[x] / sum_a)
    pi=np.array(pi)
    print ("Dimensions pi:",pi.shape)
    print(pi)
    
    return mean ,cov_2, pi





# In[157]:


def EM(X,mean,cov_2,pi):

    N, D = X.shape
    K = mean.shape[0]
    Jold = np.inf
    maxIters = 100#number of iterations
    tol = 1e-6#tolerance of convergence
    costs = []
    
    for it in range(maxIters): 
        J = 0#total cost
        print("iteration: ",it)

        # Step 1 -- Calculate a priori possibilities
        gamma = np.zeros((N,K))
        gamma = e_step(X,N,K,mean,cov_2,pi)
        # Step 2 -- Update gaussian parameters
        print("Update Gaussian parameters")
        pi, mean ,cov_2 = m_step(N,D,K,gamma,mean,cov_2,pi)
        
        # Calculate log-likelihood
        print("Calculate log-likelihood")
        J = calculate_log_likelihood(N,D,K,mean,cov_2,pi)
        # Step 3 -- Check for convergence
        costs.append(J)
        print("Iteration #{}, Cost function value: {}".format(it, J))
        if np.abs(J - Jold) < tol:
            break
        Jold = J
    plot_costs(costs)
    np.save("costs_k_" + str(K),costs)
    return X, mean, cov_2 , pi ,gamma


# In[158]:


def Normal(X,n,k,mean,cov_2,pi):
    D=X.shape[1]
    c=1
    try:
        for d in range(D):
            a=1/(np.sqrt(2*np.pi*cov_2[k]))
            e=np.square(X[n][d]-mean[k][d])
            b=a*np.exp((-1/(2*cov_2[k]))*e)
            c=c*b
    except IndexError:
        print("Error with d:",d," n: ",n," k: ",k,"X.shape:", X.shape, "mean.shape ",mean.shape)
    return c
    


# In[159]:


def gamma_fun(X,N,K,mean,cov_2,pi):
    gamma = np.zeros((N,K))
    D = mean.shape[1]
    print("Calculate a priori possibilities")
    for k in range(K):
        for n in range(N):            
            a=pi[k]*Normal(X,n,k,mean,cov_2,pi)
            b=sum(pi[j]*Normal(X,n,j,mean,cov_2,pi) for j in range(K))
           
            gamma[n][k]=a/b
    return gamma 


# In[160]:


def plot_costs(costs):
    plt.plot(costs)
    plt.ylabel('log likelihood')
    plt.xlabel('iterations ')
    plt.title("Log likelihood ")
    plt.show()


#  ### Ε-step
#  Υπολογισμός του γ(k) ( πιθανότητας δεδομένου ότι ανήκει στην κατηγορια k)
#  

# In[161]:


def e_step(X,N,K,mean,cov_2,pi):
    return gamma_fun(X,N,K,mean,cov_2,pi)


# ### M-step 
# Ανανέωση των παραμέτρων των gaussian κατανομών

# In[162]:


def update_pi(N,k,gamma,mean,cov_2,pi):
    sum1=0
    for n in range(N):
        sum1+=gamma[n][k]
    pi[k]=sum1/N
    return pi


# In[163]:


def update_mean(N,D,k,gamma,mean,cov_2,pi):
    sum1=0
    sum2=0
    for d in range(D):
        sum1=sum(gamma[n][k]*X[n][d] for n in range(N))
        sum2=sum(gamma[n][k] for n in range(N))
        mean[k][d]=sum1/(D*sum2)
    return mean


# In[164]:


def update_cov_2(N,D,k,gamma,mean,cov_2,pi,mean_new):
    sum1=0
    sum2=0
    for n in range(N):
        sum2+=gamma[n][k]
        for d in range(D):
            sum1+=gamma[n][k]*np.square(X[n][d]-mean_new[k][d])
    cov_2[k]= sum1/(D*sum2)  
    return cov_2


# In[165]:


def m_step(N,D,K,gamma,mean,cov_2,pi):
    for k in range(K):
        pi_new=update_pi(N,k,gamma,mean,cov_2,pi)
        mean_new=update_mean(N,D,k,gamma,mean,cov_2,pi)
        cov_2_new=update_cov_2(N,D,k,gamma,mean,cov_2,pi,mean_new)
    return pi_new,mean_new,cov_2_new


# In[166]:


def calculate_log_likelihood(N,D,K,mean,cov_2,pi):
    sum1=0
    sum2=0
    for n in range(N):
        sum1=sum(pi[j]*Normal(X,n,j,mean,cov_2,pi) for j in range(K))
        sum2+=np.log(sum1)
    return sum2


K=4
mean , cov_2 , pi = initialize_mean_cov_pi(new_img,K)
X=new_img

meaninit=np.copy(mean)
X,mean,cov_2,pi,gamma=EM(X,mean,cov_2,pi)
np.save("gamma_file_k_"+str(K),gamma)
np.save("mean_file_k_"+str(K),mean)
print( "Final k clusters" )
print( mean )
print( "Inital k clusters" )
print( meaninit )

rec_img=np.zeros((X.shape[0],X.shape[1]))
for n in range(X.shape[0]):
    rec_img[n] = mean[gamma[n].argmax()]

plt.imshow(np.reshape(rec_img,(array.shape[0],array.shape[1],array.shape[2])))
plt.show()
from numpy import linalg as LA
N=rec_img.shape[0]
error=np.square(LA.norm(new_img-rec_img))/N
print(error)
