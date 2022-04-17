
<p align="center">
    <img src="https://user-images.githubusercontent.com/15319503/163697964-5a9958f1-2f30-4da6-8cec-53ecc82965fe.png" alt="ai-earth-sciences" width="150" height="150">
  </a>
  <h1 align="center">Papers on AI in Earth Sciences 🌎</h1>
  <p align="center">📰 📄 A collection of research papers from the earth science community that use AI approaches.</p>
  <p align="center">
      <a href="https://twitter.com/javedali99"><img src="https://img.shields.io/badge/twitter-%231DA1F2.svg?&style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter@javedali"></a>
  <a href="https://www.linkedin.com/in/javedali18"><img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn@javedali"></a>
  <a href="mailto:javedali28@gmail.com"><img src="https://img.shields.io/badge/gmail-D14836?&style=for-the-badge&logo=gmail&logoColor=white" alt="javedali28@gmail.com"></a>
  </p>
  <br>
</p>



This repository contains a list of scientific papers from the Earth Sciences (Hydrology, Meteorology, Climate science etc.) that use AI (machine learning or deep learning) approaches. 

If you have any comments or suggestions for additions or improvements for this repository, submit an issue or a pull request. If you can’t contribute on GitHub, [send me an email](mailto:javedali28@gmail.com). 

If you find these resources useful, please give this repository a star ⭐️. 

 ---

### Table of Content

* [Hydrology](#hydrology)
* [Ecology](#ecology)
* [Meteorology](#meteorology)
* [Climate Science](#climate-science)
* [How To Read a Paper](#how-to-read-a-paper)


---


## Hydrology

- [Machine learning assisted hybrid models can improve streamflow simulation in diverse catchments across the conterminous US](https://iopscience.iop.org/article/10.1088/1748-9326/aba927)
  <details>
  <summary><b>Abstract</b></summary>
  
  >Incomplete representations of physical processes often lead to structural errors in process-based (PB) hydrologic models. Machine learning (ML) algorithms can reduce streamflow modeling errors but do not enforce physical consistency. As a result, ML algorithms may be unreliable if used to provide future hydroclimate projections where climates and land use patterns are outside the range of training data. Here we test hybrid models built by integrating PB model outputs with an ML algorithm known as long short-term memory (LSTM) network on their ability to simulate streamflow in 531 catchments representing diverse conditions across the Conterminous United States. Model performance of hybrid models as measured by Nash–Sutcliffe efficiency (NSE) improved relative to standalone PB and LSTM models. More importantly, hybrid models provide highest improvement in catchments where PB models fail completely (i.e. NSE < 0). However, all models performed poorly in catchments with extended low flow periods, suggesting need for additional research.
  </details>

- [Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks](https://hess.copernicus.org/articles/22/6005/2018/) 
  <details>
  <summary><b>Abstract</b></summary>
  
  >Rainfall–runoff modelling is one of the key challenges in the field of hydrology. Various approaches exist, ranging from physically based over conceptual to fully data-driven models. In this paper, we propose a novel data-driven approach, using the Long Short-Term Memory (LSTM) network, a special type of recurrent neural network. The advantage of the LSTM is its ability to learn long-term dependencies between the provided input and output of the network, which are essential for modelling storage effects in e.g. catchments with snow influence. We use 241 catchments of the freely available CAMELS data set to test our approach and also compare the results to the well-known Sacramento Soil Moisture Accounting Model (SAC-SMA) coupled with the Snow-17 snow routine. We also show the potential of the LSTM as a regional hydrological model in which one model predicts the discharge for a variety of catchments. In our last experiment, we show the possibility to transfer process understanding, learned at regional scale, to individual catchments and thereby increasing model performance when compared to a LSTM trained only on the data of single catchments. Using this approach, we were able to achieve better model performance as the SAC-SMA + Snow-17, which underlines the potential of the LSTM for hydrological modelling applications.
  </details>
 
- [Developing a Long Short-Term Memory (LSTM) based model for predicting water table depth in agricultural areas](https://doi.org/10.1016/j.jhydrol.2018.04.065)
  <details>
  <summary><b>Abstract</b></summary>
  
  >Predicting water table depth over the long-term in agricultural areas presents great challenges because these areas have complex and heterogeneous hydrogeological characteristics, boundary conditions, and human activities; also, nonlinear interactions occur among these factors. Therefore, a new time series model based on Long Short-Term Memory (LSTM), was developed in this study as an alternative to computationally expensive physical models. The proposed model is composed of an LSTM layer with another fully connected layer on top of it, with a dropout method applied in the first LSTM layer. In this study, the proposed model was applied and evaluated in five sub-areas of Hetao Irrigation District in arid northwestern China using data of 14 years (2000–2013). The proposed model uses monthly water diversion, evaporation, precipitation, temperature, and time as input data to predict water table depth. A simple but effective standardization method was employed to pre-process data to ensure data on the same scale. 14 years of data are separated into two sets: training set (2000–2011) and validation set (2012–2013) in the experiment. As expected, the proposed model achieves higher R2 scores (0.789–0.952) in water table depth prediction, when compared with the results of traditional feed-forward neural network (FFNN), which only reaches relatively low R2 scores (0.004–0.495), proving that the proposed model can preserve and learn previous information well. Furthermore, the validity of the dropout method and the proposed model’s architecture are discussed. Through experimentation, the results show that the dropout method can prevent overfitting significantly. In addition, comparisons between the R2 scores of the proposed model and Double-LSTM model (R2 scores range from 0.170 to 0.864), further prove that the proposed model’s architecture is reasonable and can contribute to a strong learning ability on time series data. Thus, one can conclude that the proposed model can serve as an alternative approach predicting water table depth, especially in areas where hydrogeological data are difficult to obtain.
  </details>
 
- [Uncertainty estimation with deep learning for rainfall–runoff modeling](https://doi.org/10.5194/hess-26-1673-2022)
   <details>
  <summary><b>Abstract</b></summary>
  
  >Deep learning is becoming an increasingly important way to produce accurate hydrological predictions across a wide range of spatial and temporal scales. Uncertainty estimations are critical for actionable hydrological prediction, and while standardized community benchmarks are becoming an increasingly important part of hydrological model development and research, similar tools for benchmarking uncertainty estimation are lacking. This contribution demonstrates that accurate uncertainty predictions can be obtained with deep learning. We establish an uncertainty estimation benchmarking procedure and present four deep learning baselines. Three baselines are based on mixture density networks, and one is based on Monte Carlo dropout. The results indicate that these approaches constitute strong baselines, especially the former ones. Additionally, we provide a post hoc model analysis to put forward some qualitative understanding of the resulting models. The analysis extends the notion of performance and shows that the model learns nuanced behaviors to account for different situations.
  </details>

- [Evaluation of artificial intelligence models for flood and drought forecasting in arid and tropical regions](https://doi.org/10.1016/j.envsoft.2021.105136)
   <details>
  <summary><b>Abstract</b></summary>
  
  >With the advancement of computer science, Artificial Intelligence (AI) is being incorporated into many fields to increase prediction performance. Disaster management is one of the main fields embracing the techniques of AI. It is essential to forecast the occurrence of disasters in advance to take the necessary mitigation steps and reduce damage to life and property. Therefore, many types of research are conducted to predict such events due to climate change in advance using hydrological, mathematical, and AI-based approaches. This paper presents a comparison of three major accepted AI-based approaches in flood and drought forecasting. In this study, fluvial floods are measured by the runoff change in rivers whereas meteorological droughts are measured using the Standard Precipitation Index (SPI). The performance of the Convolutional Neural Network (CNN), Long-Short Term Memory network (LSTM), and Wavelet decomposition functions combined with the Adaptive Neuro-Fuzzy Inference System (WANFIS) are compared in flood and drought forecasting, with five statistical performance criteria and accepted flood and drought indicators used for comparison, extending to two climatic regions: arid and tropical. The results suggest that the CNN performs best in flood forecasting with WANFIS for meteorological drought forecasting, regardless of the climate of the region under study. Besides, the results demonstrate the increased accuracy of the CNN in applications with multiple features in the input.
  </details>
  
- [Long short-term memory neural network (LSTM-NN) for aquifer level time series forecasting using in-situ piezometric observations](https://doi.org/10.1016/j.jhydrol.2021.126800)
   <details>
  <summary><b>Abstract</b></summary>
  
  >The application of neural networks (NN) in groundwater (GW) level prediction has been shown promising by previous works. Yet, previous works have relied on a variety of inputs, such as air temperature, pumping rates, precipitation, service population, and others. This work presents a long short-term memory neural network (LSTM-NN) for GW level forecasting using only previously observed GW level data as the input without resorting to any other type of data and information about a groundwater basin. This work applies the LSTM-NN for short-term and long-term GW level forecasting in the Edwards aquifer in Texas. The Adam optimizer is employed for training the LSTM-NN. The performance of the LSTM-NN was compared with that of a simple NN under 36 different scenarios with prediction horizons ranging from one day to three months, and covering several conditions of data availability. This paper’s results demonstrate the superiority of the LSTM-NN over the simple-NN in all scenarios and the success of the LSTM-NN in accurate GW level prediction. The LSTM-NN predicts one lag, up to four lags, and up to 26 lags ahead GW level with an accuracy (R2) of at least 99.89%, 99.00%, and 90.00%, respectively, over a testing period longer than 17 years of the most recent records. The quality of this work’s results demonstrates the capacity of machine learning (ML) in groundwater prediction, and affirms the importance of gathering high-quality, long-term, GW level data for predicting key groundwater characteristics useful in sustainable groundwater management.
  </details>
  
- [Groundwater level forecasting with artificial neural networks: a comparison of long short-term memory (LSTM), convolutional neural networks (CNNs), and non-linear autoregressive networks with exogenous input (NARX)](https://hess.copernicus.org/articles/25/1671/2021/)
   <details>
  <summary><b>Abstract</b></summary>
  
  >It is now well established to use shallow artificial neural networks (ANNs) to obtain accurate and reliable groundwater level forecasts, which are an important tool for sustainable groundwater management. However, we observe an increasing shift from conventional shallow ANNs to state-of-the-art deep-learning (DL) techniques, but a direct comparison of the performance is often lacking. Although they have already clearly proven their suitability, shallow recurrent networks frequently seem to be excluded from the study design due to the euphoria about new DL techniques and its successes in various disciplines. Therefore, we aim to provide an overview on the predictive ability in terms of groundwater levels of shallow conventional recurrent ANNs, namely non-linear autoregressive networks with exogenous input (NARX) and popular state-of-the-art DL techniques such as long short-term memory (LSTM) and convolutional neural networks (CNNs). We compare the performance on both sequence-to-value (seq2val) and sequence-to-sequence (seq2seq) forecasting on a 4-year period while using only few, widely available and easy to measure meteorological input parameters, which makes our approach widely applicable. Further, we also investigate the data dependency in terms of time series length of the different ANN architectures. For seq2val forecasts, NARX models on average perform best; however, CNNs are much faster and only slightly worse in terms of accuracy. For seq2seq forecasts, mostly NARX outperform both DL models and even almost reach the speed of CNNs. However, NARX are the least robust against initialization effects, which nevertheless can be handled easily using ensemble forecasting. We showed that shallow neural networks, such as NARX, should not be neglected in comparison to DL techniques especially when only small amounts of training data are available, where they can clearly outperform LSTMs and CNNs; however, LSTMs and CNNs might perform substantially better with a larger dataset, where DL really can demonstrate its strengths, which is rarely available in the groundwater domain though.
  </details>
  
- [Uncertainty assessment of LSTM based groundwater level predictions](https://doi.org/10.1080/02626667.2022.2046755)
   <details>
  <summary><b>Abstract</b></summary>
  
  >Due to the underlying uncertainty in groundwater level (GWL) modelling, point prediction of GWLs does not provide sufficient information. Moreover, the insufficiency of data on subjects such as illegal exploitation wells and wastewater pounds, which are untraceable, underlines the importance of evolved uncertainty in the groundwaters of the Ardabil plain. Thus, estimating prediction intervals (PIs) for groundwater modelling can be an important step. In this paper, PIs were estimated for GWLs of selected piezometers of the Ardebil plain in Iran using the artificial neural network (ANN)-based lower upper bound estimation (LUBE) method. The classic feedforward neural network (FFNN) and deep-learning-based long short-term memory (LSTM) were used. GWL data of piezometers and hydrological data (1992–2018) were applied for modelling. The results indicate that LSTM outperforms FFNN in both PI and point prediction tasks. LSTM-based LUBE was found to be superior to FFNN-based LUBE, providing an average 25% lower coverage width criterion (CWC). PIs estimated for piezometers with high transmissivity resulted in 50% lower CWC than PIs estimated for piezometers in areas with lower transmissivity.
  </details>
  
 <br>
    
## Ecology
    
- [Graph Learning for Inverse Landscape Genetics](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_32.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Inferring unknown edges from data at a graph’s nodes is a common problem across statistics and machine learning. We study a version that arises in the field of landscape genetics, where genetic similarity between organisms living in a heterogeneous landscape is explained by a graph that encodes the ease of dispersal through that landscape. Our main contribution is an efficient algorithm for inverse landscape genetics, the task of inferring edges in this graph based on the similarity of genetic data from populations at different nodes. This problem is important in discovering impediments to dispersal that threaten biodiversity and species survival. Drawing on influential work that models dispersal using graph effective resistances, we reduce the inverse landscape genetics problem to that of inferring graph edges from noisy measurements of these resistances. Then, building on edgeinference techniques for social networks, we develop an efficient first-order optimization method for solving the problem, which significantly outperforms existing techniques in experiments on synthetic and real genetic data.
  </details>
 
- [Segmentation of Soil Degradation Sites in Swiss Alpine Grasslands with Deep Learning](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_40.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Soil degradation is an important environmental problem which affects the Alpine
ecosystem and agriculture. Research results suggest that soil degradation in Swiss
Alpine grasslands has increased in recent years and it is expected to increase further
due to climate and land-use change. However, reliably quantifying the increase
in spatial extent of soil degradation is a challenging task. Although methods like
Object-based Image Analysis (OBIA) can provide precise detection of erosion
sites, an efficient large scale investigation is not feasible due to the labour intensive
nature and lack of transferability of the method. In this study, we overcome these
limitations by adapting the fully convolutional neural network U-Net trained on
high-quality training data provided by OBIA to enable efficient segmentation of
erosion sites in high-resolution aerial images. We find that segmentation results
of both methods, OBIA and U-Net, are generally in good agreement, but display
method specific difference, with an overall precision of 73% and recall of 84%.
Importantly, both methods indicate an increase in soil degradation for a case study
region over a 16-year period of 167% and 201% for OBIA and U-Net, respectively.
Furthermore, we show that the U-Net approach transfers well to new regions
(within our study region) and data from subsequent years, even when trained on a
comparably small training dataset. Thus the proposed approach enables large scale
analysis in Swiss Alpine grasslands and provides a tool for reliable assessment of
temporal changes in soil degradation.
  </details>
 
- [Meta-modeling large-scale spatial data using Convolutional Neural Networks](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_12.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Species connectivity models play an important role in ecological research and
biodiversity assessment. Unfortunately, simulations of connectivity models are
typically slow, therefore preventing the rapid iteration and updates of models when
evaluating different scenarios.
In this pilot study, we present the proof of concept of utilizing Deep Learning
methodologies as a novel approach in ecology for significantly reducing the prediction rate of species connectivity models.
  </details>
    
- [Understanding Climate Impacts on Vegetation with Gaussian Processes in Granger Causality](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_26.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Global warming is leading to unprecedented changes in our planet, with great
societal, economical and environmental implications, especially with the growing
demand of biofuels and food. Assessing the impact of climate on vegetation is
of pressing need. We approached the attribution problem with a novel nonlinear
Granger causal (GC) methodology and used a large data archive of remote sensing
satellite products, environmental and climatic variables spatio-temporally gridded
over more than 30 years. We generalize kernel Granger causality by considering
the variables cross-relations explicitly in Hilbert spaces, and use the covariance in
Gaussian processes. The method generalizes the linear and kernel GC methods,
and comes with tighter bounds of performance based on Rademacher complexity.
Spatially-explicit global Granger footprints of precipitation and soil moisture on
vegetation greenness are identified more sharply than previous GC methods.
  </details>
    
- [Interpreting the Impact of Weather on Crop Yield Using Attention](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_54.pdf)
  <details>
  <summary><b>Abstract</b></summary>
  
  >Accurate prediction of crop yield supported by scientific and domain-relevant
interpretations can improve agricultural breeding by providing monitoring across
diverse climatic conditions. The use of this information in plant breeding can help
provide protection against weather challenges to crop production, including erratic
rainfall and temperature variations. In addition to isolating the important time-steps,
researchers are interested to understand the effect of different weather variables on
crop yield. In this paper, we propose a novel attention-based model that can learn
the most significant variables across different weeks in the crop growing season
and highlight the most important time-steps (weeks) to predict the annual crop
yield. We demonstrate our model’s performance on a dataset based on historical
performance records from Uniform Soybean Tests (UST) in North America. The
interpretations provided by our model can help in understanding the impact of
weather variability on agricultural production in the presence of climate change
and formulating breeding strategies to circumvent these climatic challenges.
      
  </details>   
    
    
<br>    
    
## Meteorology 
 >coming soon
 
 
 
 <br>
    
## Climate Science
 
 >coming soon
  
  
<br>
  
## How To Read a Paper

_(Taken from [papers we love](https://github.com/papers-we-love/papers-we-love))_

Reading a paper is not the same as reading a blogpost or a novel. Here are a few handy resources to help you get started.

* [How to read an academic article](http://organizationsandmarkets.com/2010/08/31/how-to-read-an-academic-article/)
* [Advice on reading academic papers](https://userpages.umbc.edu/~akmassey/posts/2012-02-15-advice-on-reading-academic-papers.html)
* [How to read and understand a scientific paper](http://violentmetaphors.com/2013/08/25/how-to-read-and-understand-a-scientific-paper-2/)
* [Should I Read Papers?](http://michaelrbernste.in/2014/10/21/should-i-read-papers.html)
* [The Refreshingly Rewarding Realm of Research Papers](https://www.youtube.com/watch?v=8eRx5Wo3xYA)
