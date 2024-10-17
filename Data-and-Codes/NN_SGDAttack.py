# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:18:42 2024

@author: saionkr2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math

Resnet_weights = np.load('resnet_cifar10_w.npy')
Resnet_bias = np.load('resnet_20_cifar10_biases.npy')
cifar10_labels = np.load('label_cifar10.npy')
Resnet_ideal_inputs = np.load('resnet_cifar10_x_unscaled.npy')
Resnet_weights[Resnet_weights==0] = -1

Binary_WL_inputs = np.load('Input_col.npy')

Binary_unscaled_inputs = Binary_WL_inputs[0::2,:]

Binary_unscaled_inputs[Binary_unscaled_inputs==0] = -1

Binary_unscaled_inputs = Binary_unscaled_inputs.T

FIG_SIZE = (6,6)
GRID_ALPHA = 0.5
COLORS = {'r': 'r', 'g': 'lawngreen', 'b':'b', 'c': 'cyan', 'k': 'k', 'y':'y', 'm':'m', 'a':'salmon','l':'darkviolet','p':'teal'}
OUTPUT_DIR = 'figures'

colors = 'bgrckymalp'
marker = 'odos'
marker_size = 100 

font = {'family': 'arial',
        'color':  'black',
        'weight': 'regular',
        'size': 16,
        }

matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'

#Evaluating Network Accuracy
NN_NOI = 10000
absolute_outputs = np.zeros([NN_NOI,10])
#absolute_outputs_Bw = np.zeros([NOI,10,4])
computed_labels = np.zeros([NN_NOI])
start_ind = 0
for i in range(start_ind,NN_NOI+start_ind):
    for j in range(10):
        for k in range(7):
            for l in [0,1]:
                if(k==0):
                    absolute_outputs[i-start_ind,j] = absolute_outputs[i-start_ind,j] - np.dot(Resnet_ideal_inputs[i,:,6-k],Resnet_weights[j,:,3-l])*2**(-k-1)*2**(-l-1) 
                else:
                    absolute_outputs[i-start_ind,j] = absolute_outputs[i-start_ind,j] + np.dot(Resnet_ideal_inputs[i,:,6-k],Resnet_weights[j,:,3-l])*2**(-k-1)*2**(-l-1) 
                               
    computed_labels[i-start_ind] = int(np.argmax(absolute_outputs[i-start_ind,:]*2+Resnet_bias))   
accuracy_fp = (np.sum(computed_labels == cifar10_labels[start_ind:])/NN_NOI)*100 
print(accuracy_fp) 

NOAE = 10
NOI = 2000
N = 64
Bw = 4
times = 100
TE = 10 #Total number of epochs
code_gain = np.array([0.85, -0.81,  0.99,  0.66,  0.75,  0.09,  0.63,  0.54,
                      0.8 ,  0.09,  0.6 ,  0.61,  0.75,  0.04,  0.58,  0.59,  0.83,
                      0.16,  0.91,  0.52,  0.7 ,  0.05,  0.73,  0.47,  0.94,  0.08,
                      0.93,  0.61,  0.95, -0.27,  0.43,  0.15,  1.18, -0.54,  1.09,
                      0.58,  0.88,  0.33,  0.83, 0])


code_offset = [1.9 , -12.77,  14.08,   7.06,   6.11,   1.09,  -0.53,
         6.21,   5.53,   0.29,   3.41,   6.01,   5.26,   0.45,  -3.34,
         5.58,   6.52,   0.46,   9.82,   4.48,   5.08,   0.85,   2.54,
         4.48,  19.3 ,  -0.66,  14.9 ,   7.2 ,  20.17,  -5.22,  -8.23,
         3.18,  32.26,  -8.83,   6.1 ,   3.31,   9.99,   1.93,   9.99, 0]

                    
adc_ind = np.arange(66,66+40)
unscaled_measured_outputs_v1 = np.load('Apr6_2024_OLSAttack_1000_8p3MHz_N64_ADC65-105.npz')
unscaled_measured_outputs_v2 = np.load('Apr7_2024_OLSAttack_1k-2k_8p3MHz_N64_ADC65-105.npz')

unscaled_measured_outputs_v1 = unscaled_measured_outputs_v1[unscaled_measured_outputs_v1.files[0]][adc_ind,:,:]
unscaled_measured_outputs_v2 = unscaled_measured_outputs_v2[unscaled_measured_outputs_v2.files[0]][adc_ind,:,:]

unscaled_measured_outputs = np.zeros([40,NOI,times])
unscaled_measured_outputs[:,:1000,:] = unscaled_measured_outputs_v1
unscaled_measured_outputs[:,1000:2000,:] = unscaled_measured_outputs_v2

unscaled_measured_outputs = unscaled_measured_outputs*2-N  
for i in range(NOAE*Bw):
    unscaled_measured_outputs[i,:,:] = unscaled_measured_outputs[i,:,:]*code_gain[i] + code_offset[i]
unscaled_measured_outputs[unscaled_measured_outputs>63]=63  
unscaled_measured_outputs[unscaled_measured_outputs<-64]=-64

MSB_training_set = unscaled_measured_outputs[0::4,:,:]
MSBm1_training_set = unscaled_measured_outputs[2::4,:,:]

def isNaN(num):
    return num != num

def mean_square_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def ber(w_true, w_pred):
    l2_error = np.linalg.norm(w_true - w_pred,2)
    weight_norm = np.linalg.norm(w_true,2)
    return 100*l2_error**2/(weight_norm**2*4)

def gradient(w, a, b, x, y_true):
    # Compute gradient of the loss function with respect to w
    x_ones = np.ones(64)
    y_pred =  np.dot(x, w) + np.dot(x_ones,np.multiply(a,w**2)) + np.dot(x,np.multiply(b,w**3))
    error = y_true - y_pred
    #print(y_pred,a,np.dot(x, w))
    grad_w = -2 * np.dot(x.T+2*np.multiply(a,w).T+3*np.multiply(np.multiply(b,x),w**2).T, error) #/ len(y_true)
    grad_a = -2 * np.dot((w**2).T, error) #/ len(y_true)
    grad_b = -2 * np.dot(np.multiply(x,w**3).T, error) #/ len(y_true)
    return grad_w, grad_a, grad_b

def stochastic_gradient_descent(x, y, learning_rate=0.0001, epochs=TE):
    # Initialize weights randomly
    w = np.random.randint(2, size=(x.shape[1])).astype('float64')
    a = np.random.randn(x.shape[1])
    b = np.random.randn(x.shape[1])
    #print(x.shape[1])
    x_ones = np.ones(64)
    w_est = np.zeros([64,x.shape[0]*epochs])
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(y))
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        # Update weights for each data point
        for i in range(len(y)):
        #for i in range(5):
            grad_w, grad_a, grad_b = gradient(w, a, b, x_shuffled[i], y_shuffled[i])
            w -= learning_rate * grad_w
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b
            w_est[:,epoch*len(y)+i] = w    
        # Print loss every epoch
        y_pred = np.dot(x, w) + np.dot(x_ones,np.multiply(a,w**2)) + np.dot(x,np.multiply(b,w**3))
        loss = mean_square_loss(y, y_pred)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")
        
    return w_est

#%%

predicted_weights_MSB = np.zeros([10,64,NOI*TE,times])
l2_error_MSB = np.zeros([10,NOI*TE])

starting_pt = 1
for i in range(10):
    for k in range(100):
        print(i,k)
        predicted_weights_MSB[i,:,:,k] = stochastic_gradient_descent(Binary_unscaled_inputs[:NOI,:], MSB_training_set[i,:NOI,k])

#%%    
predicted_weights_MSB[predicted_weights_MSB<=0] = -1
predicted_weights_MSB[predicted_weights_MSB>0] = 1
predicted_weights_MSB_avg = np.mean(predicted_weights_MSB,axis=3)
predicted_weights_MSB_avg[predicted_weights_MSB_avg<=0] = -1
predicted_weights_MSB_avg[predicted_weights_MSB_avg>0] = 1
for i in range(10):
    for j in range(NOI*TE):
        l2_error_MSB[i,j] = np.linalg.norm(Resnet_weights[i,:,3]-predicted_weights_MSB_avg[i,:,j],ord=2)

#%%        

weight_norm_MSB = np.zeros(10)
for i in range(10):
    weight_norm_MSB[i] = np.linalg.norm(Resnet_weights[i,:,3], ord=2)

predicted_weights_MSBm1 = np.zeros([10,64,NOI*TE,times])

for i in range(10):
    for k in range(100):
        print(i,k)
        predicted_weights_MSBm1[i,:,:,k] = stochastic_gradient_descent(Binary_unscaled_inputs[:NOI,:], MSBm1_training_set[i,:NOI,k])

#%%
predicted_weights_MSBm1[predicted_weights_MSBm1<=0] = -1
predicted_weights_MSBm1[predicted_weights_MSBm1>0] = 1
predicted_weights_MSBm1_avg = np.mean(predicted_weights_MSBm1,axis=3)    
predicted_weights_MSBm1_avg[predicted_weights_MSBm1_avg<=0] = -1
predicted_weights_MSBm1_avg[predicted_weights_MSBm1_avg>0] = 1

l2_error_MSBm1 = np.zeros([10,NOI*TE])
for i in range(10):
    for j in range(NOI*TE):
        l2_error_MSBm1[i,j] = np.linalg.norm(Resnet_weights[i,:,2]-predicted_weights_MSBm1_avg[i,:,j],ord=2)
#%%       
    
weight_norm_MSBm1 = np.zeros(10)
for i in range(10):
    weight_norm_MSBm1[i] = np.linalg.norm(Resnet_weights[i,:,2], ord=2)

loc_idx = 3
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
plt.yticks(fontname = "Arial",weight='regular') 
plt.xticks(fontname = "Arial",weight='regular') 
ax.set_xlabel(r'# of input vectors $M$', fontsize = 25,fontdict=font)
ax.set_ylabel(r'bit-error rate (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
ax.set_ylim([-4,38])
colors = ['green','blue','orange']
starting_pt = 0
ADCid = 6

ax.plot(np.arange(starting_pt,NOI),np.zeros(NOI-starting_pt), markerfacecolor='none', ms=18, color='black')
ax.plot(np.arange(starting_pt,NOI),np.zeros(NOI-starting_pt), markerfacecolor='none', ms=18, color='black')
ax.plot(np.arange(starting_pt,NOI),100*l2_error_MSBm1[4,starting_pt:NOI*TE:TE]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18,label=r'best',color=colors[0])
ax.plot(np.arange(starting_pt,NOI),100*l2_error_MSBm1[5,starting_pt:NOI*TE:TE]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18,label=r'worst',color=colors[1])
ax.plot(np.arange(starting_pt,NOI),100*l2_error_MSBm1[2,starting_pt:NOI*TE:TE]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18,label=r'typical',color=colors[2])

ax.text(0.15, 0.02, r'ideal',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, fontname='Arial')       
ax.text(0.8, 0.9, r'ADC column',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, fontname='Arial')    
plt.legend(loc='upper center', ncol=1,bbox_to_anchor=(0.8, 0.9),prop={'size': 20, 'family':'Arial'}                        
            ,edgecolor='black',columnspacing=0.5,handlelength=1.0,handletextpad=0.5) 
plt.savefig('L2_Error_MSBm1_SGD.pdf',bbox_inches='tight')

#%%
#Evaluating Network Accuracy
NN_NOI = 10000
startNOI = 0

reconstructed_unscaledflat_outputs = np.zeros([NN_NOI,NOI*TE,10])
reconstructed_unscaledflat_labels = np.zeros([NN_NOI,NOI*TE])

reconstructed_w_unscaledflat = np.zeros([10,64,NOI*TE,2])
reconstructed_w_unscaledflat[:,:,:,1] = predicted_weights_MSB_avg
reconstructed_w_unscaledflat[:,:,:,0] = predicted_weights_MSBm1_avg

for i in range(startNOI,NN_NOI+startNOI):
    for j in range(10):
        for k in range(7):
            for l in [0,1]:
                if(k==0):
                    reconstructed_unscaledflat_outputs[i-startNOI,:,j] = reconstructed_unscaledflat_outputs[i-startNOI,:,j] - np.matmul(Resnet_ideal_inputs[i,:,6-k],reconstructed_w_unscaledflat[j,:,:,1-l])*2**(-k-1)*2**(-l-1) 
                else:   
                    reconstructed_unscaledflat_outputs[i-startNOI,:,j] = reconstructed_unscaledflat_outputs[i-startNOI,:,j] + np.matmul(Resnet_ideal_inputs[i,:,6-k],reconstructed_w_unscaledflat[j,:,:,1-l])*2**(-k-1)*2**(-l-1) 
                       
    reconstructed_unscaledflat_labels[i-startNOI,:] = (np.argmax(reconstructed_unscaledflat_outputs[i-startNOI,:,:]*2+Resnet_bias,axis=1)).astype(int)   

accuracy_reconstructed_unscaledflat = np.zeros([NOI*TE])
for i in range(NOI*TE):
    accuracy_reconstructed_unscaledflat[i] = (np.sum(reconstructed_unscaledflat_labels[:,i] == cifar10_labels[startNOI:])/NN_NOI)*100 

#%%       
 
loc_idx = 3
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
plt.yticks(fontname = "Arial",weight='regular') 
plt.xticks(fontname = "Arial",weight='regular') 
ax.set_xlabel(r'# of input vectors $M$', fontsize = 25,fontdict=font)
ax.set_ylabel('CIFAR-10 accuracy (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
ax.set_ylim(0,100)
starting_pt = 1
ax.plot(np.arange(0,NOI),np.ones(NOI-0)*accuracy_fp, markerfacecolor='none', ms=18, color='black')
ax.plot(np.arange(0,NOI),accuracy_reconstructed_unscaledflat[starting_pt:NOI*TE:TE], markerfacecolor='none', ms=18, color='blue')
ax.text(0.3, 0.9, r'FX Accuracy 88%',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, font='Arial')
plt.savefig('CIFAR-10_reconstruction_Acc_SGD.pdf',bbox_inches='tight')

#%%
mismatch_prob = np.zeros([NOI*TE])

for j in range(NOI*TE):
    mismatch_prob[j] = (np.sum(computed_labels != reconstructed_unscaledflat_labels[:,j])/NN_NOI)

    
#%%    
loc_idx = 3
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
plt.yticks(fontname = "Arial",weight='regular') 
plt.xticks(fontname = "Arial",weight='regular') 
ax.set_xlabel(r'# of input vectors $M$', fontsize = 25,fontdict=font)
ax.set_ylabel(r'mismatch probability $p_m$ (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
ax.text(0.12, 0.075, r'ideal',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, fontname='Arial')  
ax.plot(np.arange(0,NOI),np.zeros(NOI-0), markerfacecolor='none', ms=18, color='black')
ax.plot(np.arange(0,NOI),mismatch_prob[starting_pt:NOI*TE:TE]*100, markerfacecolor='none', ms=18, color='blue')
plt.savefig('CIFAR-10_MismatchProb_SGD.pdf',bbox_inches='tight')