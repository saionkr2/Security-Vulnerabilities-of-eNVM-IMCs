# Security Vulnerabilities of eNVM-based IMCs
This repository contains the code for our paper titled On the Security Vulnerabilities of MRAM-based In-Memor Computing Architectures against Model Extraction Attacks by Saion K. Roy and Naresh R. Shanbhag, to appear in ICCAD 2024. The is for the proposed attacks in the paper, i.e., basis vector (BV), least squares (LS), and stochastic gradient descent (SGD) attack which uses 22nm MRAM-based IMC chip data to generate bank-level bit-error rate and network-level mismatch probability and prediction accuracy.

## About
This paper studies the security vulnerabilities of embedded non-volatile memory (eNVM)-based in-memory computing (IMC) architectures to model extraction attacks (MEAs). These attacks allow the reconstruction of private training data from trained model parameters thereby leaking sensitive user information. The presence of analog noise in eNVM-based IMC computation suggests that they may be intrinsically robust to MEA. However, we show that this conjecture is false. Specifically, we consider the scenario where an attacker aims to retrieve model parameters via input-output query access, and propose three attacks that exploit the statistics of the IMC computation. We demonstrate the efficacy of these attacks in extracting the model parameters of the last layer of a ResNet-20 network from the bitcell array of an MRAM-based IMC prototype in 22 nm process. Employing the proposed MEAs, the attacker obtains a CIFAR-10 accuracy within 0.1% of that of a N=64 dimensional, 7 b x 4 b fixed-point digital baseline. To the best of our knowledge, this is the first work to demonstrate MEAs for eNVM-based IMC on a real-life IC prototype. Our results indicate the critical importance of investigating the security vulnerabilities of IMCs in general, and eNVM-based IMCs, in particular.

## Environment
The following Python 3 packages are required to run the program
* numpy
* matplotlib

## Acknowledgements
This research was supported by SRC and DARPA funded JUMP 2.0 centers, COCOSYS and CUBIC.
