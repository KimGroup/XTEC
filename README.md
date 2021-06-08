Xray TEmperature Clustering (XTEC)

Venderley et al. (https://arxiv.org/abs/2008.03275).

............................................................................................

When the temperature  T  is lowered below a certain threshold, the system can give way to an ordered state. Hence the temperature ( T ) evolution of the XRD intensity for reciprocal space point  q,  I(q,T) , must be qualitatively different if the given reciprocal space point q reflects order parameters or their fluctuations. Tracking the temperature evolution of thousands of Brillouin zones to identify systematic trends and correlations in any comprehensive manner is impossible to achieve manually without selection bias.

XTEC is an unsupervised and interpretable ML algorithm that can identify the order parameters and their fluctuations from the voluminous data by clustering the temperature series associated with a given q: I(q,T) , according to qualitative features in the temperature dependence.

At the core of XTEC is a Gaussian Mixture Model (GMM) clustering to identify disctint temperature trajectories. The figure below shows a simplified illutration of GMM clustering behind XTEC.

![image](https://user-images.githubusercontent.com/72625766/121227481-9b6a1f80-c859-11eb-8de0-e4d01a637aa3.png)

To cluster distinct  I(g)  trajectories given the collection of series  {I(g0),I(g1),…,I(gN−1)}  (N=2 in this case), the raw trajectories (in panel (a)) can be mapped to a simple Gaussian Mixture Model (GMM) clustering problem on a  N  dimensional space (panel (b)). In this case, GMM clustering identifies three distinct clusters color-coded as red, blue and green. From the GMM cluster mean and variance (panel (b)), we get the distinct trajectories of  I(g)  and their variance (panel (c)).

Note that  g  can be any parameter like temperature, time, energy etc. Hence apart from temperature series data, you can adapt XTEC to analysie any other parametric dependence like time or energy series data.


.........................................................................


Check out the XTEC_Tutorial_Notebook.ipynb to see XTEC in action. This notebook shows how XTEC identifies a charge density wave (CDW) ordering in reciprocal space from temperature series voluminous XRD data collected at Advanced Photon Source on Sr3Rh4Sn13: a a quasi-skutterudite family which shows CDW ordering below a critical temperature.  
