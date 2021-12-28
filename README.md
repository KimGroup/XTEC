# Xray TEmperature Clustering (XTEC)

* Venderley et al. (https://arxiv.org/abs/2008.03275).

## Introduction

At synchrotron sources, such as the Advanced Photon Source (APS), it is now routinely possible to collect x-ray diffraction data from single crystals that contain thousands, and even tens of thousands, of Brillouin Zones, in well under half an hour. This means that detailed parametric studies, *e.g.*, as a function of temperature or magnetic field, can be completed in well under a day. These capabilities have arisen from the coupling of high x-ray brightness with new generations of photon-counting area detectors that combine fast frame rates with high dynamic range and low backgrounds.

These experimental methods are ideal for investigating the temperature dependence of structural correlations, whether short- or long-range. For example, if there is a structural phase transition, below which new superlattice peaks emerge owing to a reduction in symmetry, these experiments will contain information on the temperature dependence both of the order parameter, *i.e.*, the superlattice peak intensities, and the critical fluctuations above the transition. However, ensuring that all components of the order parameter, including secondary order parameters, or all relevant fluctuations have been identified is not generally possible by manual inspection alone. 

To meet this challenge, the application, *XRD Temperature Clustering (XTEC)*, uses unsupervised machine learning, in particular, the Gaussian Mixture Model, to extract from the billions of recorded pixels a reduced set of temperature trajectories that correspond to distinct physical processes in the material.  The trajectories are rescaled so that we can compare trajectories at different intensities, focusing on their temperature dependence rather than their absolute scale. A technique known as label smoothing averages cluster assignments of neighboring pixels to enforce local correlations. *XTEC* is able to extract both the temperature dependence and the Q-dependence of emergent order parameters, such as charge-density-wave modulations, without any prior input. It has also been used to separate superlattice peaks from the critical fluctuations that surround them.

## Methods

When the temperature `T`  is lowered below a certain threshold, the system can give way to an ordered state. Hence the temperature (`T`) evolution of the XRD intensity for reciprocal space point  `q`,  `I(q,T)` , must be qualitatively different if the given reciprocal space point `q` reflects order parameters or their fluctuations. Tracking the temperature evolution of thousands of Brillouin zones to identify systematic trends and correlations in any comprehensive manner is impossible to achieve manually without selection bias.

*X-TEC* is an unsupervised and interpretable machine learning (ML) algorithm that can identify the order parameters and their fluctuations from the voluminous data by clustering the temperature series associated with a given `q`: `I(q,T)` , according to qualitative features in the temperature dependence.

At the core of *XTEC* is a Gaussian Mixture Model (GMM) clustering to identify disctint temperature trajectories. The figure below shows a simplified illutration of GMM clustering behind *XTEC*.

![image](https://user-images.githubusercontent.com/72625766/121227481-9b6a1f80-c859-11eb-8de0-e4d01a637aa3.png)

To cluster distinct  `I(g)`  trajectories given the collection of series  `{I(g0),I(g1),…,I(gN−1)}`  (`N=2` in the above figure), the raw trajectories (in panel (a)) can be mapped to a simple Gaussian Mixture Model (GMM) clustering problem on a  N  dimensional space (panel (b)). In the above figure, GMM clustering identifies three distinct clusters color-coded as red, blue and green. From the GMM cluster mean and variance (panel (b)), we get the distinct trajectories of  `I(g)`  and their variance (panel (c)).

Note that  `g`  can be any parameter like temperature, time, energy etc. Hence apart from temperature series data, you can adapt *X-TEC* to analysie any other parametric dependence like time or energy series data.

## Installation

Released versions of *XTEC* can be installed using either

```
    $ pip install xtec
```

or by cloning the Git repository and installing from source:

```
    $ git clone https://github.com/KimGroup/XTEC
```
## Tutorials

There are three Jupyter notebooks, which can be downloaded from this
repository to see XTEC in action. They show how XTEC identifies a 
charge density wave (CDW) ordering in reciprocal space from temperature 
series voluminous XRD data collected at Advanced Photon Source on 
Sr<sub>3</sub>Rh<sub>4</sub>Sn<sub>13</sub>: a a quasi-skutterudite 
family which shows CDW ordering below a critical temperature. The  
tutorials are: 

1. `Tutorial_XTEC-d`: this performs simple GMM clustering, treating each 
pixel independently. This mode of XTEC can distinguish the diffuse scatering, 
hence ideal for probing fluctuations of order parameters.

2. `Tutorial_XTEC-s_with_label_smoothing`: ensures the cluster assignments
in neighbouring pixels are correlated (smoothed). This mode is better suited to 
probe order parameter peaks and their visualization in reciprocal space. 

3. `Tutorial_XTEC-s_with_peak_averaging`: Faster and cheaper version of 
label smoothing by assigning connected pixels of the peaks with their peak averaged 
intensities. Best suited to get order parameters quickly from large datasets.



## Data for tutorials 
The XRD data on Sr<sub>3</sub>Rh<sub>4</sub>Sn<sub>13</sub> collected at the Advanced
Photon Source is available for download at https://dx.doi.org/10.18126/iidy-30e7. 
Download the file `srn0_XTEC.nxs` (~32 GB) which has all the data needed for the 
tutorial notebooks.

## Contact info
Any questions and comments on the code, tutorials and the use of XTEC can be directed to
Krishnanand Mallayya (kmm537@cornell.edu)
