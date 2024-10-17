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
NOI = 64
N = 64
Bx = 7
Bw = 4
times = 100

code_gain = np.array([ 0.62317129, 0, 0.51780804, 0, 0.47781266, 0, 0.62270638, 0 
                      ,0.58682881, 0, 0.64486762, 0, 0.61976217, 0, 0.61160741, 0
                      ,0.5125321 , 0, 0.5322428, 0, 0.59029505, 0, 0.48260022, 0 
                      ,0.50180059, 0, 0.50763519, 0, 0.74344185, 0, 0.50997796, 0
                      ,0.54332991, 0, 0.57980052, 0, 0.49261643, 0, 0.43725824, 0])


code_offset = [ 4.67036563, 0, -3.8931104, 0, -5.358747, 0, -15.96330864, 0,
               -17.7442648 , 0, -19.10694047, 0, -13.32272529, 0,  -20.98637353, 0,
               -14.58098095, 0, -3.55808699, 0, -16.24643549, 0, -14.38848783, 0,
               -3.25649228, 0,  -3.71915168, 0, 4.09604674, 0, -5.69713235, 0,
               6.87622466, 0, -8.33377719, 0, -9.66202891, 0,  -9.93121553, 0]

                    
adc_ind = np.arange(66,66+40)
unscaled_measured_outputs = np.load('Apr3_2024_Attack_8p3MHz_N64_ADC65-105.npz')
unscaled_measured_outputs = unscaled_measured_outputs[unscaled_measured_outputs.files[0]][adc_ind,:,:]

unscaled_measured_outputs = unscaled_measured_outputs*2-N  
for i in range(NOAE*Bw-1):
    unscaled_measured_outputs[i,:,:] = unscaled_measured_outputs[i,:,:]*code_gain[i] + code_offset[i]
    
unscaled_measured_outputs[unscaled_measured_outputs>63]=63  
unscaled_measured_outputs[unscaled_measured_outputs<-64]=-64

unscaled_measured_outputs_per_row = np.zeros([40,NOI,times])
for i in range(NOI):
    unscaled_measured_outputs_per_row[:,i,:] =  unscaled_measured_outputs[:,-1,:] - unscaled_measured_outputs[:,i,:]

#%%
threstimes = 3
threshold = np.array([-1,0,1])
predicted_weights_MSB = np.zeros([10,NOI,times,threstimes])
l2_error_MSB = np.zeros([10,times,threstimes])

for i in range(10):
    for j in range(times):
        for k in range(threstimes):
            predicted_weights_MSB[i,:,j,k] = np.mean(unscaled_measured_outputs_per_row[4*i,:,:(j+1)],axis=1)
            predicted_weights_MSB[i,predicted_weights_MSB[i,:,j,k]<=threshold[k],j,k] = -1
            predicted_weights_MSB[i,predicted_weights_MSB[i,:,j,k]>threshold[k],j,k] = 1
            l2_error_MSB[i,j,k] = np.linalg.norm(Resnet_weights[i,:,3]-predicted_weights_MSB[i,:,j,k],ord=2)

weight_norm_MSB = np.zeros(10)
for i in range(10):
    weight_norm_MSB[i] = np.linalg.norm(Resnet_weights[i,:,3], ord=2)

#%%
predicted_weights_MSBm1 = np.zeros([10,NOI,times,threstimes])
l2_error_MSBm1 = np.zeros([10,times,threstimes])

for i in range(10):
    for j in range(times):
        for k in range(threstimes):
            predicted_weights_MSBm1[i,:,j,k] = np.mean(unscaled_measured_outputs_per_row[4*i+2,:,:(j+1)],axis=1)
            predicted_weights_MSBm1[i,predicted_weights_MSBm1[i,:,j,k]<=threshold[k],j,k] = -1
            predicted_weights_MSBm1[i,predicted_weights_MSBm1[i,:,j,k]>threshold[k],j,k] = 1
            l2_error_MSBm1[i,j,k] = np.linalg.norm(Resnet_weights[i,:,2]-predicted_weights_MSBm1[i,:,j,k],ord=2)

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
ax.set_xlabel(r'# of repeated queries $N_{\mathrm{rep}}$', fontsize = 25,fontdict=font)
ax.set_ylabel(r'bit-error rate (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
ADCid = 6
colors = ['green','blue','orange']
ax.plot(np.arange(1,101),np.zeros(100), markerfacecolor='none', ms=18, color='black')

ax.plot(np.arange(1,101),100*l2_error_MSBm1[4,:,1]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18,label=r'best'%threshold[0],color=colors[0])
ax.plot(np.arange(1,101),100*l2_error_MSBm1[2,:,1]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18,label=r'worst'%threshold[1],color=colors[1])
ax.plot(np.arange(1,101),100*l2_error_MSBm1[8,:,1]**2/(weight_norm_MSBm1[ADCid]**2*4), markerfacecolor='none', ms=18,label=r'typical'%threshold[2],color=colors[2])

ax.text(0.15, 0.09, r'ideal',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, fontname='Arial')       
ax.text(0.8, 0.9, r'ADC column',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, fontname='Arial')    
plt.legend(loc='upper center', ncol=1,bbox_to_anchor=(0.8, 0.9),prop={'size': 20, 'family':'Arial'}                        
            ,edgecolor='black',columnspacing=0.5,handlelength=1.0,handletextpad=0.5)
plt.savefig('L2_Error_MSBm1_Basis.pdf',bbox_inches='tight')

#%%
#Evaluating Network Accuracy
NN_NOI = 10000
startNOI = 0
reconstructed_unscaledflat_outputs = np.zeros([NN_NOI,times,10,threstimes])
reconstructed_unscaledflat_labels = np.zeros([NN_NOI,times,threstimes])
accuracy_reconstructed_unscaledflat = np.zeros([times,threstimes])

reconstructed_w_unscaledflat = np.zeros([10,N,times,2,threstimes])
reconstructed_w_unscaledflat[:,:,:,1,:] = predicted_weights_MSB
reconstructed_w_unscaledflat[:,:,:,0,:] = predicted_weights_MSBm1

for i in range(startNOI,NN_NOI+startNOI):
    for m in range(times):
        for th in range(threstimes):
            for j in range(10):
                for k in range(7):
                    for l in [0,1]:
                        if(k==0):
                            reconstructed_unscaledflat_outputs[i-startNOI,m,j,th] = reconstructed_unscaledflat_outputs[i-startNOI,m,j,th] - np.matmul(Resnet_ideal_inputs[i,:,6-k],reconstructed_w_unscaledflat[j,:,m,1-l,th])*2**(-k-1)*2**(-l-1) 
                        else:   
                            reconstructed_unscaledflat_outputs[i-startNOI,m,j,th] = reconstructed_unscaledflat_outputs[i-startNOI,m,j,th] + np.matmul(Resnet_ideal_inputs[i,:,6-k],reconstructed_w_unscaledflat[j,:,m,1-l,th])*2**(-k-1)*2**(-l-1) 
                               
            reconstructed_unscaledflat_labels[i-startNOI,m,th] = (np.argmax(reconstructed_unscaledflat_outputs[i-startNOI,m,:,th]*2+Resnet_bias,axis=0)).astype(int)   

for j in range(times):
    for th in range(threstimes):
        accuracy_reconstructed_unscaledflat[j,th] = (np.sum(reconstructed_unscaledflat_labels[:,j,th] == cifar10_labels[startNOI:])/NN_NOI)*100 

#%%
      
loc_idx = 3
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
plt.yticks(fontname = "Arial",weight='regular') 
plt.xticks(fontname = "Arial",weight='regular') 
ax.set_xlabel(r'# of repeated queries $N_{\mathrm{rep}}$', fontsize = 25,fontdict=font)
ax.set_ylabel('CIFAR-10 accuracy (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
ax.set_ylim(0,100)
colors = ['green','blue','orange']
ax.text(0.3, 0.9, r'FX accuracy 88%',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, fontname='Arial') 
ax.plot(np.arange(1,101),np.ones(100)*accuracy_fp, markerfacecolor='none', ms=18, color='black')
ax.plot(np.arange(1,101),accuracy_reconstructed_unscaledflat[:,1], markerfacecolor='none', ms=18,label=r'$\tau=$%d'%threshold[1],color=colors[1])
plt.savefig('CIFAR-10_ReconstructionAcc_Basis.pdf',bbox_inches='tight')

#%%
times = 100
NN_times = 10000
mismatch_prob = np.zeros([times, threstimes])

for j in range(times):
    for th in range(threstimes):
        mismatch_prob[j,th] = (np.sum(computed_labels != reconstructed_unscaledflat_labels[:,j,th])/NN_times)

#%%    
loc_idx = 3
ftsize = 15
fig, ax = plt.subplots(figsize = FIG_SIZE)
plt.yticks(fontname = "Arial",weight='regular') 
plt.xticks(fontname = "Arial",weight='regular') 
ax.grid(which = 'both', alpha = GRID_ALPHA, linestyle = 'dotted')
ax.set_xlabel(r'# of repeated queries $N_{\mathrm{rep}}$', fontsize = 25,fontdict=font)
ax.set_ylabel(r'mismatch probability $p_m$ (%)', fontsize = 25,fontdict=font)
ax.tick_params(axis='y', labelsize=15) 
ax.tick_params(axis='x', labelsize=15) 
#ax.set_ylim(0,100)
colors = ['green','blue','orange']
ax.text(0.15, 0.075, r'ideal',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=20, fontname='Arial') 
ax.plot(np.arange(1,times+1),np.zeros(times), markerfacecolor='none', ms=18, color='black')
ax.plot(np.arange(1,times+1),mismatch_prob[:,1]*100, markerfacecolor='none', ms=18,label=r'$\tau=$%d'%threshold[1],color=colors[1])
plt.savefig('CIFAR-10_MismatchProb_Basis.pdf',bbox_inches='tight')

