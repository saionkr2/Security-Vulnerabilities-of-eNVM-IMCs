#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:45:35 2023

@author: saionroy
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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

#%%
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

#%%
NOAE = 10
NOI = 2000
N = 64
Bw = 4
times = 100
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

#%%
unscaled_measured_outputs = unscaled_measured_outputs*2-N  
for i in range(NOAE*Bw):
    unscaled_measured_outputs[i,:,:] = unscaled_measured_outputs[i,:,:]*code_gain[i] + code_offset[i]
unscaled_measured_outputs[unscaled_measured_outputs>63]=63  
unscaled_measured_outputs[unscaled_measured_outputs<-64]=-64

#%%        

predicted_weights_MSB = np.zeros([10,64,NOI,times])
l2_error_MSB = np.zeros([10,NOI,times])

for i in range(10):
    for j in range(NOI):
        if(np.linalg.det(np.matmul(Binary_unscaled_inputs[:(j+1),:].T,Binary_unscaled_inputs[:(j+1),:]))==0):
            temp_mat = np.matmul(Binary_unscaled_inputs[:(j+1),:].T,Binary_unscaled_inputs[:(j+1),:]) + np.diag(np.ones(64))
        else:
            temp_mat = np.matmul(Binary_unscaled_inputs[:(j+1),:].T,Binary_unscaled_inputs[:(j+1),:])
        for k in range(times):
            predicted_weights_MSB[i,:,j,k] = np.matmul(np.linalg.inv(temp_mat),np.matmul(Binary_unscaled_inputs[:(j+1),:].T,np.mean(unscaled_measured_outputs[4*i,:(j+1),:(k+1)],axis=1)))
            predicted_weights_MSB[i,predicted_weights_MSB[i,:,j,k]<=0,j,k] = -1
            predicted_weights_MSB[i,predicted_weights_MSB[i,:,j,k]>0,j,k] = 1
            l2_error_MSB[i,j,k] = np.linalg.norm(Resnet_weights[i,:,3]-predicted_weights_MSB[i,:,j,k],ord=2)

predicted_weights_MSB_avg = np.mean(predicted_weights_MSB,axis=3)
predicted_weights_MSB_avg[predicted_weights_MSB_avg<=0] = -1
predicted_weights_MSB_avg[predicted_weights_MSB_avg>0] = 1

l2_error_MSB_avg = np.zeros([10,NOI])
for i in range(10):
    for j in range(NOI):
        l2_error_MSB_avg[i,j] = np.linalg.norm(Resnet_weights[i,:,3]-predicted_weights_MSB_avg[i,:,j],ord=2)

weight_norm_MSB = np.zeros(10)
for i in range(10):
    weight_norm_MSB[i] = np.linalg.norm(Resnet_weights[i,:,3], ord=2)

#%%        
predicted_weights_MSBm1 = np.zeros([10,64,NOI,times])
l2_error_MSBm1 = np.zeros([10,NOI,times])

for i in range(10):
    for j in range(NOI):
        if(np.linalg.det(np.matmul(Binary_unscaled_inputs[:(j+1),:].T,Binary_unscaled_inputs[:(j+1),:]))==0):
            temp_mat = np.matmul(Binary_unscaled_inputs[:(j+1),:].T,Binary_unscaled_inputs[:(j+1),:]) + np.diag(np.ones(64))
        else:
            temp_mat = np.matmul(Binary_unscaled_inputs[:(j+1),:].T,Binary_unscaled_inputs[:(j+1),:])
        for k in range(times):
            predicted_weights_MSBm1[i,:,j,k] = np.matmul(np.linalg.inv(temp_mat),np.matmul(Binary_unscaled_inputs[:(j+1),:].T,np.mean(unscaled_measured_outputs[4*i+2,:(j+1),:(k+1)],axis=1)))
            predicted_weights_MSBm1[i,predicted_weights_MSBm1[i,:,j,k]<=0,j,k] = -1
            predicted_weights_MSBm1[i,predicted_weights_MSBm1[i,:,j,k]>0,j,k] = 1
            l2_error_MSBm1[i,j,k] = np.linalg.norm(Resnet_weights[i,:,2]-predicted_weights_MSBm1[i,:,j,k],ord=2)

l2_error_MSBm1_avg = np.zeros([10,NOI])
weight_norm_MSBm1 = np.zeros(10)

predicted_weights_MSBm1_avg = np.mean(predicted_weights_MSBm1,axis=3)
predicted_weights_MSBm1_avg[predicted_weights_MSBm1_avg<=0] = -1
predicted_weights_MSBm1_avg[predicted_weights_MSBm1_avg>0] = 1

l2_error_MSBm1_avg = np.zeros([10,NOI])
for i in range(10):
    for j in range(NOI):
        l2_error_MSBm1_avg[i,j] = np.linalg.norm(Resnet_weights[i,:,2]-predicted_weights_MSBm1_avg[i,:,j],ord=2)
        
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
ax.set_ylim([-8,62])
ax.set_yscale('linear')
starting_pt = 64
colors = ['green','blue','orange']
ADCid = 6
ax.plot(np.arange(starting_pt,NOI),np.zeros(NOI-starting_pt), markerfacecolor='none', ms=18, color='black')
ax.plot(np.arange(starting_pt,NOI),100*l2_error_MSBm1_avg[4,starting_pt:NOI]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18,label=r'best',color=colors[0])
ax.plot(np.arange(starting_pt,NOI),100*l2_error_MSBm1_avg[2,starting_pt:NOI]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18,label=r'worst',color=colors[1])
ax.plot(np.arange(starting_pt,NOI),100*l2_error_MSBm1_avg[8,starting_pt:NOI]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18,label=r'typical',color=colors[2])

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
plt.savefig('L2_Error_MSBm1_OLS.pdf',bbox_inches='tight')


#%%
#Evaluating Network Accuracy
NN_NOI = 10000
startNOI = 0

reconstructed_unscaledflat_outputs = np.zeros([NN_NOI,NOI,10])
reconstructed_unscaledflat_labels = np.zeros([NN_NOI,NOI])
accuracy_reconstructed_unscaledflat = np.zeros([NOI])

reconstructed_w_unscaledflat = np.zeros([10,64,NOI,2])
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

for i in range(NOI):
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
starting_pt = 64
ax.plot(np.arange(starting_pt,NOI),np.ones(NOI-starting_pt)*accuracy_fp, markerfacecolor='none', ms=18, color='black')
ax.plot(np.arange(starting_pt,NOI),accuracy_reconstructed_unscaledflat[starting_pt:], markerfacecolor='none', ms=18, color='blue')
ax.text(0.3, 0.9, r'FX accuracy 88%',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, fontname='Arial')  
plt.savefig('CIFAR-10_ReconstructionAcc_OLS.pdf',bbox_inches='tight')

#%%  
  
mismatch_prob = np.zeros([NOI])

for j in range(NOI):
    mismatch_prob[j] = (np.sum(computed_labels != reconstructed_unscaledflat_labels[:,j])/NN_NOI)

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
ax.text(0.15, 0.075, r'ideal',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, fontname='Arial')  
ax.plot(np.arange(starting_pt,NOI),np.zeros(NOI-starting_pt), markerfacecolor='none', ms=18, color='black')
ax.plot(np.arange(starting_pt,NOI),mismatch_prob[starting_pt:]*100, markerfacecolor='none', ms=18, color='blue')
plt.savefig('CIFAR-10_MismatchProb_OLS.pdf',bbox_inches='tight')

