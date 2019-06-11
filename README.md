# FA-GAN

This repository contains the code for the paper "Graphical analysis of the progression of atrial arrhythmia through an 
ensemble of Generative Adversarial Network Discriminators", accepted by the EUSFLAT 2019 Conference in Praga (available soon).

# Abstract

Logs of arrhythmia episodes in patients with pacemakers are used to estimate the temporal progression of atrial arrhythmia. 
In order to attain an early detection, a stream of dates and episode lengths are fed to an array of detectors, each of which is 
responsive to a narrow range of arrhythmias. The outputs of these detectors are organized on a projection map, used by the specialist 
to assess the risk in the evolution of the patient. Each of the mentioned detectors is a Recurrent Neural Network(RNN), that is in 
turn the discriminating element of a Generative Adversarial Network (GAN) that has been trained to generate temporal sequences of 
values of the degrees of truth that the arrhythmia episodes are not isolated.

# Overview

AF(Atrial Fibrilation) is the most common type of arrhythmia in clinical practice. It is a type of irregular heartbeat in which 
the upper chambers of the heart tremble or fibrillate, resulting in an irregular and rapid heart rhythm. This can cause symptoms such 
as dizziness or chest pains, even in the worst cases it can increase the risk of clots due to the fact that blood circulation 
to the ventricles is obstructed, this, being a dangerous manifestation since if the clot reaches the bloodstream 
it can cause severe problems such as a stroke.


Treatment of the disease often involves the use of pacemakers to control the heart rate. These devices provide heart rhythm monitoring, 
being able to detect episodes of arrhythmia, specifically high atrial frequency, which normally correspond to episodes of AF. 
The availability of monitoring data is interesting in order to provide an early diagnosis.

The aim of this study is to develop a model capable of capturing the distinctive properties of AF progression in order to simulate 
its behaviour. From this model it will be possible to generate arrhythmia episodes of different characteristics that can be used 
as a training set of another model composed by the discriminating elements of a set of GAN networks. This simulation model is 
necessary because it would not be possible to obtain sufficiently large training sets with real data. Taking advantage of the 
potential of neural networks, if a GAN is trained from arrhythmias of specific characteristics, at the end there will be a 
discriminating element that will be able to distinguish arrhythmias of that type, from arrhythmias of any other. 
By training a set of GAN networks in such a way that each network is trained with a specific set of arrhythmias, at the end a model 
can be built that gathers a set of detectors of different types of AF.

The key to the success of this idea lies in the precision of the detectors, which must only react to episodes that coincide with 
a set of clinical situations similar to those with which they were trained. When this group of detectors is fed with records of a 
real patient's pacemaker, it is expected that only a few will recognize that the patient's arrhythmia is of the same type as 
the arrhythmia with which they were trained.

The simulator model uses a series of parameters to characterize the different types of arrhythmia, so the outputs of the detectors 
provide an estimate of the model parameters. As an illustrative example, suppose that arrhythmias are simulated that currently 
occur every year and whose duration is one hour, but that in the coming years will evolve to arrhythmias that occur every month 
and whose duration is three hours. It is expected that a discriminator trained with these simulated arrhythmias react to real 
cases for which the progression of the arrhythmia occurs at the same speed and not, for example, to other patients 
whose arrhythmias occur every six months, last two hours and have a stable prognosis.

It is thus proposed that the detectors be organized in a graphic map whose axes reflect the parameters with which they were trained. 
When the case of a real patient is passed, a specific area of the map is expected to be illuminated, encompassing a few detectors 
tuned to similar parameters. This will mean that they have reacted by recognizing the similarity of the arrhythmias of the 
patient with the synthetic arrhythmias with which they were trained; thus, the output of the map can be considered as a 
projection of the parameters of the model that best adapt to the criticality of the patient, thus being able to know the 
patient's state of health.


# Files in this Repository

- generateMarkov.py: it generates different types of arrhythmia episodes from a Markov model that depends mainly on the Alpha and Beta
parameters. Alpha represents the evolution of the AF and Beta the mean time between two arrhythmia episodes.
- data folder: it contains the data generated by generateMarkov.py 
- regular_experiment/experiment_markov.py: this script makes possible to train a GAN in order to learn generating arrhythmia episodes
and at the same time learn exactly their distribution. It makes use of other helper scripts, also inside the regular_experiment folder.
- trained_models folder: it contains some of the trained models 
- evaluate-sensibility-base.py: this script evaluate the sensibility of a detector (the discriminative part of a trained GAN model).
