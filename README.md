
<p align="center">
    <img src="https://user-images.githubusercontent.com/15319503/163697964-5a9958f1-2f30-4da6-8cec-53ecc82965fe.png" alt="ai-earth-sciences" width="150" height="150">
  </a>
  <h1 align="center">Papers on AI in Earth Sciences 🌎</h1>
  <p align="center">📰 📄 A collection of research papers from the earth science community that use AI approaches.</p>
  
</p>

<br>

This repository contains a list of scientific papers from the Earth Sciences (Hydrology, Meteorology, Climate science etc.) that use AI (machine learning or deep learning) approaches. 

If you have any comments or suggestions for additions or improvements for this repository, submit an issue or a pull request. If you can’t contribute on GitHub, [send me an email](mailto:javedali28@gmail.com). 

If you find these resources useful, please give this repository a star ⭐️ 

<h3 align="center">:mailbox: Connect with me :mailbox:</h3>
 <p align="center">
      <a href="https://twitter.com/javedali99"><img src="https://img.shields.io/badge/twitter-%231DA1F2.svg?&style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter@javedali"></a>
  <a href="https://www.linkedin.com/in/javedali18"><img src="https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn@javedali"></a>
 <a href="https://javedali.net"><img src="https://img.shields.io/badge/Website%20-%2302569B.svg?&style=for-the-badge&logo=WordPress&logoColor=white" alt="LinkedIn@javedali"></a>
 </p>

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
  
    
- [A Machine Learner’s Guide to Streamflow Prediction](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_18.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Although often subconsciously, many people deal with water-related issues on
a daily basis. For instance, many regions rely on hydropower plants to produce
their electricity, and, at the extreme, floods and droughts pose one of the big environmental threats of climate change. At the same time, many machine learning
researchers have started to look beyond their field and wish to contribute to environmental issues of our time. The modeling of streamflow—the amount of water
that flows through a river cross-section at a given time—is a natural starting point
to such contributions: It encompasses a variety of tasks that will be familiar to
machine learning researchers, but it is also a vital component of flood and drought
prediction (among other applications). Moreover, researchers can draw upon large
open datasets, sensory networks, and remote sensing data to train their models. As
a getting-started resource, this guide provides a brief introduction to streamflow
modeling for machine learning researchers and highlights a number of possible
research directions where machine learning could advance the domain.
  </details>
  
- [A Deep Learning Architecture for Conservative Dynamical Systems: Application to Rainfall-Runoff Modeling]()
     <details>
  <summary><b>Abstract</b></summary>
  
  >The most accurate and generalizable rainfall-runoff models produced by the hydrological sciences community to-date are based on deep learning, and in particular, on
Long Short Term Memory networks (LSTMs). Although LSTMs have an explicit
state space and gates that mimic input-state-output relationships, these models are
not based on physical principles. We propose a deep learning architecture that is
based on the LSTM and obeys conservation principles. The model is benchmarked
on the mass-conservation problem of simulating streamflow.
  </details>
   
    
- [Pix2Streams: Dynamic Hydrology Maps from Satellite-LiDAR Fusion](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_39.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Where are the Earth’s streams flowing right now? Inland surface waters expand
with floods and contract with droughts, so there is no one map of our streams.
Current satellite approaches are limited to monthly observations that map only
the widest streams. These are fed by smaller tributaries that make up much of
the dendritic surface network but whose flow is unobserved. A complete map of
our daily waters can give us an early warning for where droughts are born: the
receding tips of the flowing network. Mapping them over years can give us a map
of impermanence of our waters, showing where to expect water, and where not to.
To that end, we feed the latest high-res sensor data to multiple deep learning models
in order to map these flowing networks every day, stacking the times series maps
over many years. Specifically, i) we enhance water segmentation to 50 cm/pixel
resolution, a 60× improvement over previous state-of-the-art results. Our U-Net
trained on 30-40cm WorldView3 images can detect streams as narrow as 1-3m
(30-60× over SOTA). Our multi-sensor, multi-res variant, WasserNetz, fuses a
multi-day window of 3m PlanetScope imagery with 1m LiDAR data, to detect
streams 5-7m wide. Both U-Nets produce a water probability map at the pixel-level.
ii) We integrate this water map over a DEM-derived synthetic valley network map
to produce a snapshot of flow at the stream level. iii) We apply this pipeline,
which we call Pix2Streams, to a 2-year daily PlanetScope time-series of three
watersheds in the US to produce the first high-fidelity dynamic map of stream flow
frequency. The end result is a new map that, if applied at the national scale, could
fundamentally improve how we manage our water resources around the world.
  </details>    
    
- [Efficient Reservoir Management through Deep Reinforcement Learning](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_36.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Dams impact downstream river dynamics through flow regulation and disruption
of upstream-downstream linkages. However, current dam operation is far from satisfactory due to the inability to respond the complicated and uncertain dynamics of
the upstream-downstream system and various usages of the reservoir. Even further,
the insuitable dam operation can cause floods in downstream areas. Therefore, we
leverage reinforcement learning (RL) methods to compute efficient dam operation
guidelines in this work. Specifically, we build offline simulators with real data
and different mathematical models for the upstream inflow, i.e., generalized least
square (GLS) and dynamic linear model (DLM), then use the simulator to train
the state-of-the-art RL algorithms, including DDPG, TD3 and SAC. Experiments
show that the simulator with DLM can efficiently model the inflow dynamics in
the upstream and the dam operation policies trained by RL algorithms significantly
outperform the human-generated policy.
  </details>    
    
- [Inductive Predictions of Extreme Hydrologic Events in The Wabash River Watershed](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_34.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >We present a machine learning method to predict extreme hydrologic events from
spatially and temporally varying hydrological and meteorological data. We used a
timestep reduction technique to reduce the computational and memory requirements and
trained a bidirection LSTM network to predict soil water and stream flow from time
series data observed and simulated over eighty years in the Wabash River Watershed. We
show that our simple model can be trained much faster than complex attention networks
such as GeoMAN without sacrificing accuracy. Based on the predicted values of soil
water and stream flow, we predict the occurrence and severity of extreme hydrologic
events such as droughts. We also demonstrate that extreme events can be predicted in
geographical locations separate from locations observed during the training process.
This spatially-inductive setting enables us to predict extreme events in other areas in the
US and other parts of the world using our model trained with the Wabash Basin data.
  </details>    
    

- [A Comparison of Data-Driven Models for Predicting Stream Water Temperature](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_23.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Changes to the Earth’s climate are expected to negatively impact water resources in
the future. It is important to have accurate modelling of river flow and water quality
to make optimal decisions for water management. Machine learning and deep
learning models have become promising methods for making such hydrological
predictions. Using these models, however, requires careful consideration both
of data constraints and of model complexity for a given problem. Here, we use
machine learning (ML) models to predict monthly stream water temperature records
at three monitoring locations in the Northwestern United States with long-term
datasets, using meteorological data as predictors. We fit three ML models: a
Multiple Linear Regression, a Random Forest Regression, and a Support Vector
Regression, and compare them against two baseline models: a persistence model
and historical model. We show that all three ML models are reasonably able to
predict mean monthly stream temperatures with root mean-squared errors (RMSE)
ranging from 0.63-0.91 ◦C. Of the three ML models, Support Vector Regression
performs the best with an error of 0.63-0.75 ◦C. However, all models perform
poorly on extreme values of water temperature. We identify the need for machine
learning approaches for predicting extreme values for variables such as water
temperature, since it has significant implications for stream ecosystems and biota.
  </details>    
      
- [HESS Opinions: Incubating deep-learning-powered hydrologic science advances as a community](https://hess.copernicus.org/articles/22/5639/2018/)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Recently, deep learning (DL) has emerged as a revolutionary and versatile tool transforming industry applications and generating new and improved capabilities for scientific discovery and model building. The adoption of DL in hydrology has so far been gradual, but the field is now ripe for breakthroughs. This paper suggests that DL-based methods can open up a complementary avenue toward knowledge discovery in hydrologic sciences. In the new avenue, machine-learning algorithms present competing hypotheses that are consistent with data. Interrogative methods are then invoked to interpret DL models for scientists to further evaluate. However, hydrology presents many challenges for DL methods, such as data limitations, heterogeneity and co-evolution, and the general inexperience of the hydrologic field with DL. The roadmap toward DL-powered scientific advances will require the coordinated effort from a large community involving scientists and citizens. Integrating process-based models with DL models will help alleviate data limitations. The sharing of data and baseline models will improve the efficiency of the community as a whole. Open competitions could serve as the organizing events to greatly propel growth and nurture data science education in hydrology, which demands a grassroots collaboration. The area of hydrologic DL presents numerous research opportunities that could, in turn, stimulate advances in machine learning as well.
  </details>
    
    
- [A Transdisciplinary Review of Deep Learning Research and Its Relevance for Water Resources Scientists](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018WR022643)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Deep learning (DL), a new generation of artificial neural network research, has transformed industries, daily lives, and various scientific disciplines in recent years. DL represents significant progress in the ability of neural networks to automatically engineer problem-relevant features and capture highly complex data distributions. I argue that DL can help address several major new and old challenges facing research in water sciences such as interdisciplinarity, data discoverability, hydrologic scaling, equifinality, and needs for parameter regionalization. This review paper is intended to provide water resources scientists and hydrologists in particular with a simple technical overview, transdisciplinary progress update, and a source of inspiration about the relevance of DL to water. The review reveals that various physical and geoscientific disciplines have utilized DL to address data challenges, improve efficiency, and gain scientific insights. DL is especially suited for information extraction from image-like data and sequential data. Techniques and experiences presented in other disciplines are of high relevance to water research. Meanwhile, less noticed is that DL may also serve as a scientific exploratory tool. A new area termed AI neuroscience, where scientists interpret the decision process of deep networks and derive insights, has been born. This budding subdiscipline has demonstrated methods including correlation-based analysis, inversion of network-extracted features, reduced-order approximations by interpretable models, and attribution of network decisions to inputs. Moreover, DL can also use data to condition neurons that mimic problem-specific fundamental organizing units, thus revealing emergent behaviors of these units. Vast opportunities exist for DL to propel advances in water sciences.
  </details>
    
    
- [Performance analysis of unorganized machines in streamflow forecasting of Brazilian plants](https://www.sciencedirect.com/science/article/pii/S1568494618301935?via%3Dihub)
     <details>
  <summary><b>Abstract</b></summary>
  
  >This work performs an extensive investigation about the application of unorganized machines – extreme learning machines and echo state networks – to predict monthly seasonal streamflow series, associated to three important Brazilian hydroelectric plants, for many forecasting horizons. The aforementioned models are neural network architectures which present efficient and simple training processes. Moreover, the selection of the best inputs of each model is carried out by the wrapper method, using three different evaluation criteria, and three filters, viz., those based on the partial autocorrelation function, the mutual information and the normalization of maximum relevance and minimum common redundancy method. This study also establishes a comparison between the unorganized machines and two classical models: the partial autoregressive model and the multilayer perceptron. The computational results demonstrate that the unorganized machines, especially the echo state networks, represent efficient alternatives to solve the task.
  </details> 
    
  
- [An enhanced extreme learning machine model for river flow forecasting: State-of-the-art, practical applications in water resource engineering area and future research direction](https://doi.org/10.1016/j.jhydrol.2018.11.069)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Despite the massive diversity in the modeling requirements for practical hydrological applications, there remains a need to develop more reliable and intelligent expert systems used for real-time prediction purposes. The challenge in meeting the standards of an expert system is primarily due to the influence and behavior of hydrological processes that is driven by natural fluctuations over the physical scale, and the resulting variance in the underlying model input datasets. River flow forecasting is an imperative task for water resources operation and management, water demand assessments, irrigation and agriculture, early flood warning and hydropower generations. This paper aims to investigate the viability of the enhanced version of extreme learning machine (EELM) model in river flow forecasting applied in a tropical environment. Herein, we apply the complete orthogonal decomposition (COD) learning tool to tune the output-hidden layer of the ELM model’s internal neuronal system, instead of the conventional multi-resolution tool (e.g., singular value decomposition). To demonstrate the application of EELM model, the Kelantan River, located in the Malaysian peninsular, selected as a case study. For a comparison of the EELM model, and further model evaluation, two distinct data-intelligent models are developed (i.e., the classical ELM and the support vector regression, SVR model). An exhaustive list of diagnostic indicators are used to evaluate the EELM model in respect to the benchmark algorithms, namely, SVR and ELM. The model performance indicators exhibit superior results for the EELM model relative to ELM and SVR models. In addition, the EELM model is presented as a more accurate, alternative predictive tool for modelling the tropical river flow patterns and its underlying characteristic perturbations in the physical space. Several statistical metrics defined as the coefficient of determination (r), Nash-Sutcliffe efficiency (Ens), Willmott’s Index (WI), root-mean-square error (RMSE) and mean absolute error (MAE) are computed to assess the model’s effectiveness. In quantitative terms, superiority of EELM over ELM and SVR models was exhibited by Ens = 0.7995, 0.7434 and 0.665, r = 0.894, 0.869 and 0.818 and WI = 0.9380, 0.9180 and 0.8921, respectively. Whereas, EELM model attained lower (RMSE and MAE) values by approximately (11.61–22.53%) and (8.26–8.72%) relative to ELM and SVR models, respectively. The obtained results reveal that the EELM model is a robust expert model and can be embraced practically in real-life water resources management and river sustainability decisions. As a complementary component of this paper, we also review state-of-art research works where scholars have embraced extensive implementation of the ELM model in water resource engineering problems. A comprehensive evaluation is carried out to recognize the current limitations, and also to propose potential opportunities of applying improved variants of the ELM model presented as a future research direction.
  </details>    
    
- [Comparison of stochastic and machine learning methods for multi-step ahead forecasting of hydrological processes](https://link.springer.com/article/10.1007/s00477-018-1638-6)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Research within the field of hydrology often focuses on the statistical problem of comparing stochastic to machine learning (ML) forecasting methods. The performed comparisons are based on case studies, while a study providing large-scale results on the subject is missing. Herein, we compare 11 stochastic and 9 ML methods regarding their multi-step ahead forecasting properties by conducting 12 extensive computational experiments based on simulations. Each of these experiments uses 2000 time series generated by linear stationary stochastic processes. We conduct each simulation experiment twice; the first time using time series of 100 values and the second time using time series of 300 values. Additionally, we conduct a real-world experiment using 405 mean annual river discharge time series of 100 values. We quantify the forecasting performance of the methods using 18 metrics. The results indicate that stochastic and ML methods may produce equally useful forecasts.
  </details>
    
    
- [Prolongation of SMAP to Spatiotemporally Seamless Coverage of Continental U.S. Using a Deep Learning Neural Network](https://doi.org/10.1002/2017GL075619)
     <details>
  <summary><b>Abstract</b></summary>
  
  >The Soil Moisture Active Passive (SMAP) mission has delivered valuable sensing of surface soil moisture since 2015. However, it has a short time span and irregular revisit schedules. Utilizing a state-of-the-art time series deep learning neural network, Long Short-Term Memory (LSTM), we created a system that predicts SMAP level-3 moisture product with atmospheric forcings, model-simulated moisture, and static physiographic attributes as inputs. The system removes most of the bias with model simulations and improves predicted moisture climatology, achieving small test root-mean-square errors (<0.035) and high-correlation coefficients >0.87 for over 75% of Continental United States, including the forested southeast. As the first application of LSTM in hydrology, we show the proposed network avoids overfitting and is robust for both temporal and spatial extrapolation tests. LSTM generalizes well across regions with distinct climates and environmental settings. With high fidelity to SMAP, LSTM shows great potential for hindcasting, data assimilation, and weather forecasting.
  </details>    
    
- [Estimating surface soil moisture from SMAP observations using a Neural Network technique](https://doi.org/10.1016/j.rse.2017.10.045)
     <details>
  <summary><b>Abstract</b></summary>
  
  >A Neural Network (NN) algorithm was developed to estimate global surface soil moisture for April 2015 to March 2017 with a 2–3 day repeat frequency using passive microwave observations from the Soil Moisture Active Passive (SMAP) satellite, surface soil temperatures from the NASA Goddard Earth Observing System Model version 5 (GEOS-5) land modeling system, and Moderate Resolution Imaging Spectroradiometer-based vegetation water content. The NN was trained on GEOS-5 soil moisture target data, making the NN estimates consistent with the GEOS-5 climatology, such that they may ultimately be assimilated into this model without further bias correction. Evaluated against in situ soil moisture measurements, the average unbiased root mean square error (ubRMSE), correlation and anomaly correlation of the NN retrievals were 0.037 m3m −3, 0.70 and 0.66, respectively, against SMAP core validation site measurements and 0.026 m3m −3, 0.58 and 0.48, respectively, against International Soil Moisture Network (ISMN) measurements. At the core validation sites, the NN retrievals have a significantly higher skill than the GEOS-5 model estimates and a slightly lower correlation skill than the SMAP Level-2 Passive (L2P) product. The feasibility of the NN method was reflected by a lower ubRMSE compared to the L2P retrievals as well as a higher skill when ancillary parameters in physically-based retrievals were uncertain. Against ISMN measurements, the skill of the two retrieval products was more comparable. A triple collocation analysis against Advanced Microwave Scanning Radiometer 2 (AMSR2) and Advanced Scatterometer (ASCAT) soil moisture retrievals showed that the NN and L2P retrieval errors have a similar spatial distribution, but the NN retrieval errors are generally lower in densely vegetated regions and transition zones.
  </details>    
  
- [Using Machine Learning to Analyze Physical Causes of Climate Change: A Case Study of U.S. Midwest Extreme Precipitation](https://doi.org/10.1029/2021GL093787)
     <details>
  <summary><b>Abstract</b></summary>
  
  >While global warming has generally increased the occurrence of extreme precipitation, the physical mechanisms by which climate change alters regional and local precipitation extremes remain uncertain, with debate about the role of changes in the atmospheric circulation. We use a convolutional neural network (CNN) to analyze large-scale circulation patterns associated with U.S. Midwest extreme precipitation. The CNN correctly identifies 91% of observed precipitation extremes based on daily sea level pressure and 500-hPa geopotential height anomalies. There is evidence of increasing frequency of extreme precipitation circulation patterns (EPCPs) over the past two decades, although frequency changes are insignificant over the past four decades. Additionally, we find that moisture transport and precipitation intensity during EPCPs have increased. Our approach, which uses deep learning visualization to understand how the CNN predicts EPCPs, advances machine learning as a tool for providing insight into physical causes of changing extremes, potentially reducing uncertainty in future projections.
  </details>         
         
         
- [Enabling Smart Dynamical Downscaling of Extreme Precipitation Events With Machine Learning](https://doi.org/10.1029/2020GL090309)
     <details>
  <summary><b>Abstract</b></summary>
  
  >The projection of extreme convective precipitation by global climate models (GCM) exhibits significant uncertainty due to coarse resolutions. Direct dynamical downscaling (DDD) of regional climate at kilometer-scale resolutions provides valuable insight into extreme precipitation changes, but its computational expense is formidable. Here we document the effectiveness of machine learning to enable smart dynamical downscaling (SDD), which selects a small subset of GCM data to conduct downscaling. Trained with data for three subtropical/tropical regions, convolutional neural networks (CNNs) retained 92% to 98% of extreme precipitation events (rain intensity higher than the 99th percentile) while filtering out 88% to 95% of circulation data. When applied to reanalysis data sets differing from training data, the CNNs' skill in retaining extremes decreases modestly in subtropical regions but sharply in the deep tropics. Nonetheless, one of the CNNs can still retain 62% of all extreme events in the deep tropical region in the worst case.
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
 
- [Crop type mapping without field-level labels: Random forest transfer and unsupervised clustering techniques](https://doi.org/10.1016/j.rse.2018.12.026)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Crop type mapping at the field level is necessary for a variety of applications in agricultural monitoring and food security. As remote sensing imagery continues to increase in spatial and temporal resolution, it is becoming an increasingly powerful raw input from which to create crop type maps. Still, automated crop type mapping remains constrained by a lack of field-level crop labels for training supervised classification models. In this study, we explore the use of random forests transferred across geographic distance and time and unsupervised methods in conjunction with aggregate crop statistics for crop type mapping in the US Midwest, where we simulated the label-poor setting by depriving the models of labels in various states and years. We validated our methodology using available 30 m spatial resolution crop type labels from the US Department of Agriculture's Cropland Data Layer (CDL). Using Google Earth Engine, we computed Fourier transforms (or harmonic regressions) on the time series of Landsat Surface Reflectance and derived vegetation indices, and extracted the coefficients as features for machine learning models. We found that random forests trained on regions and years similar in growing degree days (GDD) transfer to the target region with accuracies consistently exceeding 80%. Accuracies decrease as differences in GDD expand. Unsupervised Gaussian mixture models (GMM) with class labels derived using county-level crop statistics classify crops less consistently but require no field-level labels for training. GMM achieves over 85% accuracy in states with low crop diversity (Illinois, Iowa, Indiana, Nebraska), but performs sometimes no better than random when high crop diversity interferes with clustering (North Dakota, South Dakota, Wisconsin, Michigan). Under the appropriate conditions, these methods offer options for field-resolution crop type mapping in regions around the world with few or no ground labels.
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

- [Identifying Opportunities for Skillful Weather Prediction with Interpretable Neural Networks](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_11.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >The atmosphere is chaotic. This fundamental property of the climate system makes
forecasting weather incredibly challenging: it’s impossible to expect weather models to ever provide perfect predictions of the Earth system beyond timescales of
approximately 2 weeks. Instead, atmospheric scientists look for specific states of
the climate system that lead to more predictable behaviour than others. Here, we
demonstrate how neural networks can be used, not only to leverage these states to
make skillful predictions, but moreover to identify the climatic conditions that lead
to enhanced predictability. Furthermore, we employ a neural network interpretability method called “layer-wise relevance propagation” to create heatmaps of the
regions in the input most relevant for a network’s output. For Earth scientists, these
relevant regions for the neural network’s prediction are by far the most important
product of our study: they provide scientific insight into the physical mechanisms
that lead to enhanced weather predictability. While we demonstrate our approach
for the atmospheric science domain, this methodology is applicable to a large range
of geoscientific problems.
  </details>
 
- [Spatio-temporal segmentation and tracking of weather patterns with light-weight Neural Networks](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_55.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >The reliable detection and tracking of weather patterns is a necessary first step
towards characterizing extreme weather events in a warming world. Recent work
[Prabhat et al.(2020)] has shown that weather pattern recognition by deep neural
networks can work remarkably better than feature engineering, such as hand-crafted
heuristics, used traditionally in climate science. As an extension of this work, we
perform Deep Learning - based semantic segmentation of atmospheric rivers and
tropical cyclones on the expert-annotated ClimateNet data set, and track individual
events using a spatio-temporal overlapping approach. Our approach is fast and
scalable to more data modalities and event types, motivating expansion of the
ClimateNet dataset and development of novel deep learning architectures. Furthermore, we show that the spatio-temporal tracking capability enables investigating a
host of important climate science research questions pertaining to the behavior of
extreme weather events in a warming world.
  </details> 
 
- [Leveraging Lightning with Convolutional Recurrent AutoEncoder and ROCKET for Severe Weather Detection](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_16.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Previous studies have shown that increases in flash rates detected in ground-based
lightning data can be a precursor to severe weather hazards. Lightning data from
the Geostationary Lightning Mapper (GLM) aboard the GOES-R satellite is not
part of an operational model used by forecasters and is underutilized in severe
storm research. The Advanced Baseline Imager’s (ABI) visible imagery also shows
cloud features, such as overshooting tops and above-anvil cirrus plumes, which
have been associated with severe weather hazards. We introduce a generative video
frame prediction methodology using a convolutional recurrent autoencoder, to
leverage these spatio-temporal patterns in GLM and ABI, along with ground-based
severe weather data. An initial case study is presented and contrasted with a time
series classification of GLM data. Through this study, we seek to highlight the
value of GLM data to assist meteorologists in time-constrained nowcasting (15-30
minute lead time) of severe hazards.
  </details>    
    
- [Towards Data-Driven Physics-Informed Global Precipitation Forecasting from Satellite Imagery](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_20.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Under the effects of global warming, extreme events such as floods and droughts
are increasing in frequency and intensity. This trend directly affects communities
and make all the more urgent widening the access to accurate precipitation forecasting systems for disaster preparedness. Nowadays, weather forecasting relies on
numerical models necessitating massive computing resources that most developing
countries cannot afford. Machine learning approaches are still in their infancy but
already show the promise for democratizing weather predictions, by leveraging any
data source and requiring less compute. In this work, we propose a methodology
for data-driven and physics-aware global precipitation forecasting from satellite
imagery. To fully take advantage of the available data, we design the system as
three elements: 1. The atmospheric state is estimated from recent satellite data. 2. The atmospheric state is propagated forward in time. 3. The atmospheric state is
used to derive the precipitation intensity within a nearby time interval. In particular,
our use of stochastic methods for forecasting the atmospheric state represents a
novel application in this domain.
  </details>
    
- [Temporally Weighting Machine Learning Models for High-Impact Severe Hail Prediction](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_35.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >We explore a new method to improve machine-learning (ML) based severe hail predictions. A temporal weighting scheme allows the random forest models to increase importance of relevant feature data while maintaining general information about the problem domain from other feature data. We show that the weighting scheme improves forecast skill and forecaster rust. With a flexible design, this method can produce localized forecasts under multiple different scenarios without increasing computational expense.
  </details>   
 
         
         
         
    
 <br>
    
## Climate Science
 
- [Bias correction of global climate model using machine learning algorithms to determine meteorological variables in different tropical climates of Indonesia](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_10.pdf)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Accurate and localized forecasting of climate variables are important especially
in the face of uncertainty imposed by climate change. However, the data used
for prediction are either incomplete at the local level or inaccurate because the
simulation models do not explicitly consider local contexts and extreme events.
This paper, therefore, attempts to bridge this gap by applying tree-based machine
learning algorithms to correct biases inherent in simulated, reanalysed climate
model against local climate observations in differing tropical climate subsystems
of Indonesia. The new observation datasets were compiled from various weather
stations and agencies across the country. Our results show that regions of tropical
savanna experience greatest bias corrections, followed by the tropical monsoon
and tropical forest. Finally, to account for extreme events, we embed regional
large-scale climate events into these models. In particular, we incorporate ENSO to
account for the residual error of extreme rainfall observations, and have achieved
an improved bias-correction of 36.67%.
  </details>
  
- [Deep learning to represent subgrid processes in climate models](https://www.pnas.org/doi/full/10.1073/pnas.1810286115)
     <details>
  <summary><b>Abstract</b></summary>
  
  >The representation of nonlinear subgrid processes, especially clouds, has been a major source of uncertainty in climate models for decades. Cloud-resolving models better represent many of these processes and can now be run globally but only for short-term simulations of at most a few years because of computational limitations. Here we demonstrate that deep learning can be used to capture many advantages of cloud-resolving modeling at a fraction of the computational cost. We train a deep neural network to represent all atmospheric subgrid processes in a climate model by learning from a multiscale model in which convection is treated explicitly. The trained neural network then replaces the traditional subgrid parameterizations in a global general circulation model in which it freely interacts with the resolved dynamics and the surface-flux scheme. The prognostic multiyear simulations are stable and closely reproduce not only the mean climate of the cloud-resolving simulation but also key aspects of variability, including precipitation extremes and the equatorial wave spectrum. Furthermore, the neural network approximately conserves energy despite not being explicitly instructed to. Finally, we show that the neural network parameterization generalizes to new surface forcing patterns but struggles to cope with temperatures far outside its training manifold. Our results show the feasibility of using deep learning for climate model parameterization. In a broader context, we anticipate that data-driven Earth system model development could play a key role in reducing climate prediction uncertainty in the coming decade.
  </details>         
         
         
- [Using Machine Learning to Parameterize Moist Convection: Potential for Modeling of Climate, Climate Change, and Extreme Events](https://doi.org/10.1029/2018MS001351)
     <details>
  <summary><b>Abstract</b></summary>
  
  >The parameterization of moist convection contributes to uncertainty in climate modeling and numerical weather prediction. Machine learning (ML) can be used to learn new parameterizations directly from high-resolution model output, but it remains poorly understood how such parameterizations behave when fully coupled in a general circulation model (GCM) and whether they are useful for simulations of climate change or extreme events. Here we focus on these issues using idealized tests in which an ML-based parameterization is trained on output from a conventional parameterization and its performance is assessed in simulations with a GCM. We use an ensemble of decision trees (random forest) as the ML algorithm, and this has the advantage that it automatically ensures conservation of energy and nonnegativity of surface precipitation. The GCM with the ML convective parameterization runs stably and accurately captures important climate statistics including precipitation extremes without the need for special training on extremes. Climate change between a control climate and a warm climate is not captured if the ML parameterization is only trained on the control climate, but it is captured if the training includes samples from both climates. Remarkably, climate change is also captured when training only on the warm climate, and this is because the extratropics of the warm climate provides training samples for the tropics of the control climate. In addition to being potentially useful for the simulation of climate, we show that ML parameterizations can be interrogated to provide diagnostics of the interaction between convection and the large-scale environment.
  </details>         
         

- [A Machine Learning Assisted Development of a Model for the Populations of Convective and Stratiform Clouds](https://doi.org/10.1029/2019MS001798)
     <details>
  <summary><b>Abstract</b></summary>
  
  >Traditional parameterizations of the interaction between convection and the environment have relied on an assumption that the slowly varying large-scale environment is in statistical equilibrium with a large number of small and short-lived convective clouds. They fail to capture nonequilibrium transitions such as the diurnal cycle and the formation of mesoscale convective systems as well as observed precipitation statistics and extremes. Informed by analysis of radar observations, cloud-permitting model simulation, theory, and machine learning, this work presents a new stochastic cloud population dynamics model for characterizing the interactions between convective and stratiform clouds, with the goal of informing the representation of these interactions in global climate models. Fifteen wet seasons of precipitating cloud observations by a C-band radar at Darwin, Australia are fed into a machine learning algorithm to obtain transition functions that close a set of coupled equations relating large-scale forcing, mass flux, the convective cell size distribution, and the stratiform area. Under realistic large-scale forcing, the derived transition functions show that, on the one hand, interactions with stratiform clouds act to dampen the variability in the size and number of convective cells and therefore in the convective mass flux. On the other, for a given convective area fraction, a larger number of smaller cells is more favorable for the growth of stratiform area than a smaller number of larger cells. The combination of these two factors gives rise to solutions with a few convective cells embedded in a large stratiform area, reminiscent of mesoscale convective systems.
  </details>  
         
         
         
         
         
<br>
  
## How To Read a Paper

_(Taken from [papers we love](https://github.com/papers-we-love/papers-we-love))_

Reading a paper is not the same as reading a blogpost or a novel. Here are a few handy resources to help you get started.

* [How to read an academic article](http://organizationsandmarkets.com/2010/08/31/how-to-read-an-academic-article/)
* [Advice on reading academic papers](https://userpages.umbc.edu/~akmassey/posts/2012-02-15-advice-on-reading-academic-papers.html)
* [How to read and understand a scientific paper](http://violentmetaphors.com/2013/08/25/how-to-read-and-understand-a-scientific-paper-2/)
* [Should I Read Papers?](http://michaelrbernste.in/2014/10/21/should-i-read-papers.html)
* [The Refreshingly Rewarding Realm of Research Papers](https://www.youtube.com/watch?v=8eRx5Wo3xYA)
