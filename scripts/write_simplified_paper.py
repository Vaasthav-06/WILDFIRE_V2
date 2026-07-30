"""
write_simplified_paper.py
Writes the plain-English rewrite of wildfire_paper_with_figures.tex to
wildfire_paper_simplified.tex in the same directory.
"""

OUTPUT_PATH = r"c:\dev\wildfire_v2\wildfire_paper_simplified.tex"

SIMPLIFIED = r"""\documentclass[11pt]{article}

% ── Page layout ──────────────────────────────────────────────────────────────
\usepackage[a4paper, margin=1in]{geometry}

% ── Fonts and encoding ───────────────────────────────────────────────────────
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}

% ── Mathematics ──────────────────────────────────────────────────────────────
\usepackage{amsmath}
\usepackage{amssymb}

% ── Tables ───────────────────────────────────────────────────────────────────
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}

% ── Graphics and figures ─────────────────────────────────────────────────────
\usepackage{graphicx}
\usepackage{float}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning}

% ── Hyperlinks ───────────────────────────────────────────────────────────────
\usepackage{hyperref}
\usepackage{textcomp}

\graphicspath{{paper_figures/}}

\title{Robust Wildfire Risk Prediction Across India via Date-Conditional Negative Sampling and Explainable Stacking Ensembles}
\author{}
\date{}

\begin{document}
\sloppy

\maketitle

\section*{Abstract}

A common problem in machine learning for wildfire prediction is that models trained naively tend to pick up on seasonal patterns or weather shortcuts rather than the true spatial and environmental signals that separate fire-prone locations from safe ones. To fix this, we introduce \textbf{date-conditional negative sampling} --- a new training method that pairs every fire event with non-fire examples from the exact same calendar date. This forces the model to tell apart fire from non-fire conditions within the same weather context, and we show in a controlled experiment that it stops the model from learning seasonal shortcuts and leads to more stable performance across all months of the year.

Building on this, we develop a next-day wildfire risk prediction system for India. We collected eight years (2018--2025) of satellite data from multiple sources: fire detections from NASA FIRMS (VIIRS S-NPP, Collection 2), weather from ERA5, vegetation health from MODIS, and terrain and land-cover from SRTM and ESA WorldCover. This produced 9.4 million labeled training examples across space and time.

We compare five base classifiers --- Logistic Regression, Random Forest, LightGBM, CatBoost, and XGBoost --- plus Voting and Stacking ensembles. All models are tested on a strict time-based holdout (trained on 2018--2023, tested on 2024--2025 data the model never saw during training). We chose the Stacking Ensemble as the final deployed model because it has the best precision (0.8937) and the highest AUC among ensembles. Its raw probability outputs are then corrected with Isotonic Regression post-calibration, which reduces the Expected Calibration Error by 67.7\% and makes the scores usable as real fire probabilities rather than just rankings.

For interpretability, we use SHAP to study which features matter most, both across all of India and separately across six different ecological zones. We also show how the combination of Vapor Pressure Deficit (VPD) and EVI creates a non-linear fire-risk threshold that neither variable can capture on its own. A feature ablation study measures how much each group of features contributes, confirming that the combined fuel stack (EVI + elevation) and meteorology are the most important inputs. The full system also includes a FastAPI inference backend and a React-based geospatial dashboard for viewing daily risk maps.

\section*{Introduction}

Roughly a quarter of India's land is covered by forests and natural vegetation --- these areas matter a lot, both as carbon sinks and as places millions of people depend on for their livelihoods. Unfortunately, more and more of this land is getting hit by wildfires, driven by a mix of things: climate change, longer droughts, agricultural burning practices, and terrain that makes it hard to actually put fires out once they start. Temperature and rainfall patterns in India are shifting --- fire seasons are stretching longer, and monsoon changes are creating extended dry stretches where vegetation just dries out and becomes way more flammable. Once a fire starts, it can spread incredibly fast, and the damage --- biodiversity loss, degraded soil, carbon released into the atmosphere, risk to people's homes, and messed-up watersheds --- is way bigger than the amount of time anyone actually has to contain it. Basically, fires spread faster than we can respond, which is exactly why being able to predict fires \textit{before} they happen matters so much.

The real challenge in wildfire management isn't detecting a fire once it's already burning --- satellite systems like NASA FIRMS already do that pretty well. The hard part is anticipating: figuring out, days ahead of time, which specific locations are most likely to catch fire given the current weather, how dry the vegetation is, and the shape of the terrain. This is where machine learning comes in --- it can learn the complicated, non-linear relationships between weather variables (temperature, humidity, wind, vapor pressure deficit), vegetation health (using EVI as a stand-in for how much moisture is in live vegetation), terrain factors (elevation, slope), and actual historical fire occurrence, all straight from satellite data. This approach fits the problem well because fire ignition isn't a smooth, gradual thing --- it behaves more like a threshold: past a certain point, small shifts in weather can massively change how likely a fire is, and tree-based models are naturally good at picking up on these kinds of thresholds.

One big challenge with building a model like this is that fires are rare events. On any given day, less than 0.1\% of grid cells actually catch fire, which creates a huge class imbalance problem. To deal with this properly, we use date-conditional negative sampling --- we only pull ``non-fire'' training examples from locations that didn't catch fire on the exact same date and under the same weather conditions as a real fire event elsewhere. This forces the model to learn genuine fire-vs-no-fire signals within the same season and weather regime, instead of just learning ``oh, it's summer, so fire risk is high everywhere.''

To summarize, this paper puts together: (1) a strict train/test split by time, so we're testing on genuinely unseen future data; (2) the date-conditional negative sampling method described above; (3) feature engineering based on real fire science --- weather, vegetation, and terrain features known to influence ignition; and (4) a comparison of five different classifiers, together with Voting and Stacking ensembles built on top of them, to see how much value nonlinear modeling and ensembling actually add over a simple baseline.

\textbf{Our main contributions:}

\begin{itemize}
    \item We built a large training dataset (9.4 million samples) combining NASA FIRMS fire data, ERA5 weather data, MODIS vegetation indices, and static terrain and land-cover info, spanning eight years (2018--2025). This means every training example captures the full weather, vegetation, and terrain picture relevant to whether a fire starts.

    \item We designed a date-conditional negative sampling method that matches non-fire examples to the same date and weather conditions as real fires, which helps deal with class imbalance and forces the model to learn actual fire-weather relationships instead of shortcuts.

    \item We compared five classifiers --- Logistic Regression (a simple, interpretable baseline), Random Forest, LightGBM, CatBoost, and XGBoost --- all trained and tested using the strict time-based split. We further combined these base models into Voting and Stacking ensembles and selected the best-performing configuration as the production model.

    \item We built an explanation layer using SHAP that turns model outputs into natural-language explanations, so the predictions are actually useful and understandable for forest management teams, not just data scientists.

    \item We built an interactive map-based dashboard showing daily wildfire risk across India along with risk tiers, feature importance breakdowns, and rough fire spread direction based on wind and terrain.
\end{itemize}

\noindent The rest of the paper is organized as follows. Section 2 covers related work on ML-based wildfire prediction. Section 3 explains our methodology --- data sources, feature engineering, the negative sampling approach, model architecture, and overall workflow. Section 4 covers our results and what we found. Section 5 wraps up and talks about what we'd want to do next.

\section{Materials and Methods}

\subsection{Data Description and Integration Strategy}

We combined five different datasets, each chosen because it captures something scientifically important about what drives wildfires. Table 1 lays out where each dataset comes from, what time period and resolution it covers, and why we included it.

\begin{longtable}{|p{2.8cm}|p{3.6cm}|p{1.8cm}|p{3.9cm}|}
\caption{Data Sources and Scientific Integration Rationale}
\label{tab:data_sources}\\
\hline
\textbf{Data Source} & \textbf{Variables Provided} & \textbf{Coverage} & \textbf{Rationale} \\
\hline
\endfirsthead
\hline
\textbf{Data Source} & \textbf{Variables Provided} & \textbf{Coverage} & \textbf{Rationale} \\
\hline
\endhead
NASA FIRMS (VIIRS S-NPP, C2) & Active fire detections (confidence and radiative power) & 2018--2025 & Ground truth for fire occurrence and validation \\
\hline
ERA5 Reanalysis & 2m temperature, humidity, wind speed, vapor pressure deficit & 2018--2025, $0.25 \times 0.25$ grid & Physics-driven weather context; consistent interpolation across topographically complex terrain \\
\hline
MODIS EVI & Enhanced Vegetation Index (monthly composites) & 2018--2025 & Live-fuel moisture proxy; disentangles green vegetation from soil background \\
\hline
SRTM / ESA WorldCover & Elevation (m), terrain slope, land-cover class & Static & Terrain controls fire spread rate, fuel moisture retention, and vegetation composition \\
\hline
Open-Meteo API & 24h / 7-day weather forecasts (temperature, humidity, wind) & Real-time & Operational inference; enables next-day and weekly risk visualization \\
\hline
\end{longtable}

\noindent
\subsubsection{Fire Occurrence Ground Truth}

For the ground truth on where and when fires happened, we used NASA FIRMS' VIIRS S-NPP Collection 2 active-fire product, covering 2018--2025. VIIRS detects active fires worldwide about every 12 hours. Each detection comes with a confidence score (nominal, probable, or high) and a measure of radiative power. Following standard practice in fire monitoring, we only kept detections with at least ``nominal'' confidence. We also discarded cloudy days entirely --- otherwise we'd risk labeling days as ``no fire'' just because clouds blocked the satellite's view, which would add false negatives to our training data.

Figure~\ref{fig:seasonal_freq} shows the clear seasonal pattern of wildfire occurrence across India over the 2018--2025 study period. March and April are the peak fire months (698k and 558k detections respectively), during the pre-monsoon dry season when vegetation is at its driest. The monsoon season (June--September) drives fires to near zero because of the high humidity and rainfall. There is a second smaller peak in October--November as post-monsoon drying returns to peninsular India and the north-east.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig_seasonal_fire_frequency.png}
    \caption{Monthly wildfire detection frequency across India (2018--2025, NASA FIRMS VIIRS S-NPP Collection 2). March is the peak fire month with 698k detections, driven by the pre-monsoon dry season. The monsoon period (June--September) nearly eliminates fire activity. A secondary peak occurs in October--November due to agricultural burning and post-monsoon drying. This bimodal seasonality motivates the use of cyclical month encoding and date-conditional negative sampling to prevent the model from exploiting trivial seasonal shortcuts.}
    \label{fig:seasonal_freq}
\end{figure}

\subsubsection{Meteorological Reanalysis}

For weather data, we used ERA5, which is ECMWF's global atmospheric reanalysis dataset. For every grid cell and day, we pulled: 2m air temperature, 2m relative humidity, 10m wind speed, and vapor pressure deficit (VPD), which we calculated as the gap between saturated and actual vapor pressure. We chose these specific variables because fire science research consistently links them to fire ignition --- temperature affects how dry the fuel gets, humidity directly impacts fuel moisture, wind speed helps fires spread faster by increasing oxygen availability, and VPD is a more complete way of measuring how much water stress plants are under, making it arguably a better fire-weather signal than humidity alone.

\subsubsection{Vegetation Health Indices}

To measure vegetation health, we used MODIS's Enhanced Vegetation Index (EVI), accessed through NASA's AppEEARS API. We chose EVI instead of the more common NDVI because EVI is more sensitive to actual live-canopy greenness, corrects for atmospheric effects on its own, and is less affected by bare soil in sparsely vegetated areas. EVI values range from -1 to +1 --- values near or above 0.3 generally indicate healthy, actively growing vegetation, while values under 0.2 suggest sparse or stressed vegetation. We used monthly EVI composites at 1km resolution from 2018--2025.

One technical issue we had to work around: MODIS composites are recorded on an 8-day rolling cycle, which does not line up neatly with ERA5's daily timestamps or regular calendar months. To fix this, we built a custom time-alignment procedure (using cftime) that matches each MODIS timestamp to the nearest corresponding ERA5 date.

\subsubsection{Static Topographic and Land-Cover Layers}

For elevation, we used the SRTM digital elevation model (30m resolution), averaged up to our 0.1 degree grid. For land cover, we used ESA WorldCover (10m resolution, 2021 snapshot), which classifies land into 11 standard categories such as forest, grassland, cropland, built-up areas, water, and others.

\subsubsection{Data Integration and Preprocessing}

Once everything was downloaded and cleaned, we merged all the layers into a single geospatial database, aligned to a common 0.1 degree grid covering all of India. We chose this resolution as a balance --- fine enough to capture meaningful local differences in fire risk, but coarse enough that training on 9.4 million samples does not require anything beyond normal computing hardware.

For each grid cell and date, we did the following: (1) checked FIRMS for whether a fire occurred that day, (2) pulled ERA5 weather variables via interpolation, (3) aligned and aggregated MODIS EVI data, (4) grabbed static elevation and land-cover values, (5) ran quality control checks, and (6) built the final feature vector for that cell-day. This whole process gave us 9.4 million labeled cell-day records across eight years.

\subsection{Feature Engineering and Physics-Informed Design}

We engineered features grouped into four categories based on the underlying fire science.

\subsubsection{Temporal Features}

To help the model understand seasonal patterns without creating odd artificial jumps, we encoded the calendar month using a sine-cosine (cyclical) transformation instead of raw month numbers. The problem with raw numbers is that a model would see January (1) and December (12) as being far apart, even though they are adjacent in terms of season. The cyclical encoding fixes this:

\begin{equation}
\text{Month\_sin} = \sin\left(\frac{2\pi \cdot \text{Month}}{12}\right)
\end{equation}

\begin{equation}
\text{Month\_cos} = \cos\left(\frac{2\pi \cdot \text{Month}}{12}\right)
\end{equation}

This way, December and January end up close together in feature space, matching how they are actually adjacent in the fire season calendar. This lets the gradient-boosted models pick up on seasonal fire patterns naturally, without needing separate rules for each individual month.

\subsubsection{Meteorological Features}

Four weather variables were taken from ERA5 daily fields: 2m air temperature (degrees C), 2m relative humidity (\%), 10m wind speed (m s$^{-1}$), and vapor pressure deficit (VPD, kPa). Temperature and humidity are classic fire-danger indicators from the Forest Fire Weather Index family. Wind speed is important for understanding how fast a fire can spread and how far embers can travel (long-range fire spread via flying embers). Vapor pressure deficit --- the difference between saturated and actual vapor pressure --- is an overall measure of how much water the atmosphere is demanding from plants. Recent fire ecology research has shown that VPD is a better fire-risk predictor than raw humidity. VPD is calculated from temperature and relative humidity using standard formulas, and it connects more directly to how much moisture is left in live fuel than humidity alone.

\medskip
\paragraph{Calculating VPD.} VPD is derived in three steps. First, the saturation vapor pressure (SVP) is computed from air temperature using the Tetens (Magnus-type) approximation:

\begin{equation}
SVP = 0.6108 \times \exp\left(\frac{17.27 \times T}{T + 237.3}\right)
\end{equation}

where $T$ is the air temperature in degrees C and $SVP$ is expressed in kPa. Next, the actual vapor pressure (AVP) is obtained by scaling SVP with the relative humidity:

\begin{equation}
AVP = SVP \times \left(\frac{RH}{100}\right)
\end{equation}

where $RH$ is relative humidity in \%. Finally, VPD is the difference between the saturation and actual vapor pressures:

\begin{equation}
VPD = SVP - AVP = SVP \times \left(1 - \frac{RH}{100}\right)
\end{equation}

In plain terms, SVP is the maximum moisture the air could hold at a given temperature, AVP is how much moisture is actually present, and VPD is just the gap between the two. A larger VPD means the air is ``thirstier'' --- pulling more moisture out of vegetation --- which is exactly why VPD tracks fire risk so well as temperatures rise and humidity drops.

\subsubsection{Topography and Fuel Moisture Proxy Features}

Two features describe the fuel and terrain environment: (i) SRTM elevation (m), which is the altitude of the prediction grid cell; and (ii) MODIS EVI, which measures live-vegetation greenness as a proxy for live-fuel moisture content. The reason for including elevation is well established: high-elevation areas usually have lower temperatures and higher humidity, which reduces fire risk. By contrast, valley locations can experience strong downslope winds and lower relative humidity, which increases fire risk. EVI acts as a satellite-based measure of plant water content: active photosynthesis needs adequate water inside plants, so EVI correlates with fuel moisture. During the monsoon season, when rainfall is high, EVI is elevated and fire risk is low. During dry seasons, EVI falls and fire risk rises. By including EVI directly, the model can learn this fuel-moisture signal without needing ground-based fuel-moisture measurements, which are sparse and expensive to collect across India's varied landscape.

Figure~\ref{fig:evi_dist} backs up this reasoning by comparing the MODIS EVI distribution between fire events and non-fire examples drawn from the same dates (under date-conditional negative sampling). Fire events are consistently concentrated at lower EVI values (median = 0.205), which matches the expectation that dry, stressed, or sparse vegetation is more likely to ignite. Non-fire cells cover a wider EVI range (median = 0.200) with more examples at higher values, corresponding to well-watered, high-biomass areas where high live-fuel moisture stops fires from starting even in warm weather.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig_evi_distribution_fire_class.png}
    \caption{MODIS EVI distribution for fire events (red) versus non-fire observations (blue) drawn under date-conditional sampling. Left: violin and box plots showing the interquartile range and median for each class. Right: overlapping kernel density estimates with class medians marked by dashed vertical lines. Fire events cluster at lower EVI values (median~=~0.205), consistent with drier, more flammable vegetation, while non-fire cells extend to higher EVI values indicating well-hydrated live fuel. This separation validates EVI as a meaningful live-fuel moisture proxy and justifies its inclusion in the feature set.}
    \label{fig:evi_dist}
\end{figure}

\subsubsection{Land-Cover and Interaction Features}

ESA WorldCover-derived land-cover classes (e.g., tree cover, shrubland, grassland, cropland, built-up, sparse vegetation, snow/ice, water, herbaceous wetland, mangroves, moss/lichen) were each one-hot encoded into binary indicators. This lets the model learn class-specific fire tendencies. For example, grasslands may have different fire-weather thresholds than forests because they differ in fuel loading, moisture-holding capacity, and vegetation structure.

We also computed a derived interaction feature to capture how atmospheric drying and wind work together to affect how fast a fire can spread. The fire-spread potential index (FSP) is defined as:

\begin{equation}
FSP = VPD \times W_s
\end{equation}

where VPD is vapor pressure deficit (kPa) and $W_s$ is wind speed (m s$^{-1}$). This product captures the combined effect of atmospheric water demand (drying) and oxygen replenishment (wind), both of which speed up fire spread. By including this interaction term explicitly, the model is given a physics-based feature that captures how these two effects work together. This reduces the burden on the learning algorithm to discover such interactions on its own.

Table~\ref{tab:ablation} presents a systematic feature group ablation study that quantitatively justifies the final feature set. For each group, the corresponding columns are replaced with their training-set mean values in the fitted XGBoost model (mean-substitution ablation), and performance is re-evaluated on the held-out 2024--2025 test set. This simulates the effect of that group being absent without retraining, producing directionally accurate estimates of each group's discriminative contribution.

\begin{table}[H]
\centering
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{lcccc}
\toprule
\textbf{Feature Group Ablated} & \textbf{ROC-AUC} & \textbf{Recall} & \textbf{$\Delta$AUC} & \textbf{$\Delta$Recall} \\
\midrule
\textbf{Full model (baseline)} & 0.9613 & 0.9606 & $-$ & $-$ \\
Temporal (\texttt{month\_sin}, \texttt{month\_cos}) & 0.9305 & 0.9384 & $-0.0308$ & $-0.0222$ \\
Meteorology (temp, humidity, wind, VPD) & 0.8910 & 0.9017 & $-0.0703$ & $-0.0588$ \\
VPD$\times$Wind Interaction (FSP) & 0.9586 & 0.9568 & $-0.0027$ & $-0.0038$ \\
Vegetation -- EVI & 0.9352 & 0.9789 & $-0.0262$ & $+0.0183$ \\
Elevation & 0.9153 & 0.8289 & $-0.0460$ & $-0.1317$ \\
Land Cover (all 11 indicators) & 0.9612 & 0.9606 & $-0.0001$ & $+0.0000$ \\
Fuel Stack (EVI + Elevation) & 0.8456 & 0.8717 & $-0.1157$ & $-0.0888$ \\
\midrule
KBDI (rejected candidate)$^\dagger$ & --- & --- & $\approx 0$ & $\approx 0$ \\
\bottomrule
\end{tabular}
\caption{Feature group ablation study on the held-out 2024--2025 test set, using mean-substitution on the fitted XGBoost model. Each row reports model performance when the corresponding feature group is replaced by training-set column means, simulating the effect of that group being absent (no retraining required). $\Delta$AUC and $\Delta$Recall are relative to the full model baseline (row 1); larger negative values indicate higher group importance. \textbf{Meteorology} (temperature, humidity, wind, VPD) is the single most important group by AUC ($\Delta$AUC $= -0.070$). \textbf{Elevation} is the most critical group for recall ($\Delta$Recall $= -0.132$), consistent with its role as a primary fire-suppressor in the SHAP analysis. Removing the combined \textbf{Fuel Stack} (EVI + Elevation) produces the largest joint drop ($\Delta$AUC $= -0.116$), confirming that terrain and live-fuel moisture together form the core fire-discriminating signal once weather is held constant. \textbf{Land Cover} (11 binary indicators) contributes negligible AUC ($\Delta$AUC $\approx 0$) but is retained for fuel-type context. The \textbf{VPD$\times$Wind} interaction term contributes only marginally on its own, but its physics motivation (fire-spread potential) justifies retention. $^\dagger$KBDI (Keetch--Byram Drought Index) was evaluated as a candidate feature during feature selection but excluded prior to final model training: its atmospheric drought signal was found largely redundant with VPD, and consistent daily precipitation inputs across India's heterogeneous monsoon zones introduced data-quality issues. Removing KBDI from early experiments produced no measurable change in held-out AUC.}
\label{tab:ablation}
\end{table}

Figure~\ref{fig:corr_matrix} shows the full Pearson correlation matrix for the retained feature set. Several key observations justify the final feature selection. First, VPD has a high positive correlation with temperature ($r = 0.61$) and a strong negative correlation with humidity ($r = -0.66$), confirming that VPD captures the joint effect of these two classic fire-weather variables in a single, physically meaningful number. Second, elevation is strongly negatively correlated with temperature ($r = -0.87$), VPD ($r = -0.34$), and EVI ($r = -0.54$), reflecting the cooling and drier air at higher altitudes --- all three are kept independently because each carries distinct predictive signal beyond elevation alone. Third, land-cover binary indicators (LC: Tree Cover, Grassland, Cropland, Water) have near-zero correlations with all weather and terrain features, confirming that they contribute independent information about fuel type that is not already captured by EVI or elevation.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.90\textwidth]{fig_feature_correlation_heatmap.png}
    \caption{Pearson correlation matrix for the retained feature set, computed on a 500k stratified sample of the training data. Strong correlations among weather variables (VPD--Temperature: $r=0.61$; VPD--Humidity: $r=-0.66$) confirm that VPD summarises the joint fire-weather signal. Elevation exhibits strong negative correlations with temperature ($r=-0.87$), VPD ($r=-0.34$), and EVI ($r=-0.54$), motivating its independent retention. Land-cover binary indicators are orthogonal to weather and terrain features, providing fuel-type information not captured by other features.}
    \label{fig:corr_matrix}
\end{figure}

\subsection{Date-Conditional Negative Sampling Strategy}
\label{sec:dcns}

An important design decision in any imbalanced classification problem is how to construct negative (non-event) training examples. In a real-world setting, fire events are extremely rare: across the fine-grained prediction grid covering India, the fraction of 0.1 degree cells that experience a fire on any given calendar date is approximately 0.08\% to 0.15\% depending on the season. This gives a class ratio of roughly 1:600 to 1:1200. This severe imbalance must be handled carefully during training, otherwise the model may simply predict ``no fire'' everywhere and achieve high accuracy just from the base rate --- while missing every actual fire event.

Table~\ref{tab:sampling_strategies} formally compares three negative sampling strategies. Let $\mathcal{F}$ denote the set of all observed fire events $(i,j,t)$ where $\text{FIRMS}(i,j,t)=1$.

\begin{table}[H]
\centering
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.35}
\begin{tabular}{p{3.2cm}p{6.2cm}p{4.5cm}}
\toprule
\textbf{Strategy} & \textbf{Formal Negative Set} & \textbf{Shortcut Risk} \\
\midrule
Random Uniform (RU) &
  $N^{\text{RU}}(i,j,t) = \{(i',j',t') : \text{FIRMS}(i',j',t')=0\}$,
  drawn uniformly over all dates and locations &
  Model can learn \textit{``winter / monsoon $\Rightarrow$ safe''} by
  memorising the seasonal fire calendar \\
\midrule
Spatial-Only (SO) &
  $N^{\text{SO}}(i,j,t) = \{(i',j',t') : \text{FIRMS}=0,\;t'\in\text{same month as }t\}$ &
  Reduces seasonal bias but still exposes within-month weather-regime variation
  as an exploitable signal \\
\midrule
\textbf{Date-Conditional (DC, ours)} &
  $N^{\text{DC}}(i,j,t) = \{(i',j',t) : \text{FIRMS}(i',j',t)=0,\;(i',j')\neq(i,j)\}$ &
  Negatives share the \textit{identical} ERA5 snapshot as the paired fire;
  model is forced to learn local spatial-environmental signal \\
\bottomrule
\end{tabular}
\caption{Formal comparison of three negative sampling strategies for wildfire classification.
  $\text{FIRMS}(i,j,t)$ is the fire-detection indicator at grid cell $(i,j)$ on date $t$.
  Only Date-Conditional (DC) sampling eliminates the temporal prior shift that allows
  classifiers to exploit seasonal shortcuts.}
\label{tab:sampling_strategies}
\end{table}

\medskip
\paragraph{Temporal Leakage Theorem (informal).}
Under Random-Uniform sampling, the joint training distribution $P_{\text{RU}}(\mathbf{x},y)$ conflates
the genuine fire-weather signal with seasonal prior shifts: a classifier can achieve arbitrarily low
training loss by mapping the cyclical month features $(\texttt{month\_sin},\texttt{month\_cos})$
to the historical fire-season calendar, without ever identifying which specific grid cells
are most susceptible.  Under Date-Conditional sampling, the within-batch distribution is
conditioned on the shared meteorological context $\mathbf{z}_t$ for date $t$:
\begin{equation}
P_{\text{DC}}(y=1 \mid \mathbf{x},\,\mathbf{z}_t) \;=\; P(y=1 \mid \mathbf{x}_{\text{local}})
\end{equation}
This isolates the local spatial-environmental signal $\mathbf{x}_{\text{local}}$ (vegetation health,
elevation, land cover) as the only remaining discriminative axis, exactly the signal needed for
calibrated operational fire-risk prediction.

The method we use in this work is date-conditional negative sampling. For each observed fire event on date $t$ at grid cell $(i, j)$, we draw negative examples only from confirmed non-fire cells on the same date $t$, using the same ERA5 weather snapshot as that fire event. Formally, the negative set for fire event $(i, j, t)$ is:

\begin{equation}
N(i, j, t) = \{(i', j', t) : \text{FIRMS}(i', j', t) = 0 \text{ and } (i', j') \neq (i, j)\}
\end{equation}

where $\text{FIRMS}(i', j', t)$ is the fire detection indicator. This ensures that the model must tell fire-prone from fire-safe conditions within the same weather regime and season, rather than exploiting simple seasonal shortcuts (such as ``winter does not burn'' or ``high-altitude cells do not burn''). The date-conditional strategy forces the model to learn which combinations of weather, vegetation, and topography actively lead to fire, relative to other locations experiencing the exact same weather on the same day.

To keep the classes balanced without losing signal, we used a sampling ratio of approximately 1:5 to 1:10 (fire:non-fire) during training. This ratio is balanced enough that the loss function does not collapse to a trivial solution, but still includes enough negative examples to cover the diversity of non-fire conditions. The final training dataset, after date-conditional negative sampling, contained approximately 1.2 million fire examples and 8.2 million non-fire examples, for a total of 9.4 million labeled samples.

Figure~\ref{fig:sampling_diag} illustrates the date-conditional sampling strategy visually. For each fire cell on date $t$, negative examples are drawn only from confirmed non-fire cells on the same date, using the same ERA5 weather snapshot. This forces the model to learn which combinations of local conditions --- vegetation health, terrain, and land cover --- differentiate fire-prone from fire-safe areas, given that all cells share the same regional weather.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig_date_conditional_sampling_diagram.png}
    \caption{Schematic of the date-conditional negative sampling strategy. For each fire detection (positive sample, red grid cell marked ``F'') on date~$t$, negative examples (non-fire cells on the same date~$t$) are sampled from the same ERA5 meteorological snapshot. The positive sample captures feature conditions at the fire cell; the negative samples capture conditions at non-fire cells under identical regional weather. This design forces the classifier to learn local fire-discriminating signals (EVI, elevation, land cover) rather than exploiting seasonal or weather-regime shortcuts across different dates.}
    \label{fig:sampling_diag}
\end{figure}

\subsubsection{Empirical Validation: Date-Conditional vs.\ Random-Uniform Sampling}
\label{sec:sampling_experiment}

To prove empirically that Date-Conditional sampling stops seasonal shortcut learning, we ran a controlled experiment. A \textbf{Random-Uniform (RU) baseline} XGBoost model is trained on the same positive fire examples but with negatives sampled uniformly at random across all dates --- independent of which dates had fires --- using the same hyperparameters. Both models are then tested on the same held-out 2024--2025 test set.

Table~\ref{tab:sampling_experiment} summarises the overall results. The DC model achieves a substantially higher recall (0.9606 vs.\ 0.9291, a gap of $+3.1$ percentage points) with comparable AUC, confirming that date-conditional training catches more fire events without sacrificing discriminative power. The Temporal Bias Score --- the Pearson correlation between calendar month and predicted fire probability --- is $r = -0.326$ for DC vs.\ $r = -0.340$ for RU. The RU model's stronger negative correlation shows it relies more heavily on the seasonal pattern (the link between cold/wet months and low fire probability), which is evidence of learned seasonal shortcut behaviour.

\begin{table}[H]
\centering
\renewcommand{\arraystretch}{1.25}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
\textbf{Sampling Strategy} & \textbf{ROC-AUC} & \textbf{Recall} & \textbf{Precision} & \textbf{F1} & \textbf{Month--Prob Corr.} \\
\midrule
\textbf{Date-Conditional (DC, ours)} & \textbf{0.9613} & \textbf{0.9606} & 0.8602 & \textbf{0.9076} & $-0.326$ \\
Random-Uniform (RU, baseline)        & 0.9600          & 0.9291          & \textbf{0.8823} & 0.9051 & $-0.340$ \\
\midrule
$\Delta$ (DC $-$ RU)                 & $+0.0013$       & $\mathbf{+0.0315}$ & $-0.0221$ & $+0.0025$ & $+0.014$ \\
\bottomrule
\end{tabular}%
}
\caption{Controlled comparison of Date-Conditional (DC) vs.\ Random-Uniform (RU) negative sampling strategies on the held-out 2024--2025 test set. Both models use XGBoost with identical hyperparameters; only the negative sampling strategy differs. DC achieves significantly higher recall (+3.1 pp), confirming it recovers more true fire events. The smaller absolute value of the month--probability correlation for DC ($|r|=0.326$ vs.\ $0.340$) confirms that the DC model is less dependent on the seasonal prior, i.e.\ it exploits the fire-season calendar shortcut less.}
\label{tab:sampling_experiment}
\end{table}

Figure~\ref{fig:sampling_metrics} presents the same results as a bar chart for direct visual comparison. Figure~\ref{fig:sampling_monthly} shows the critical month-by-month recall breakdown --- the most direct test for seasonal shortcut learning. During the monsoon months (July--September), when fire frequency drops sharply, the RU model's recall falls to as low as \textbf{18\%} (July), while the DC model maintains \textbf{34\%} recall. Both models show reduced recall during this naturally difficult period, but the RU model's much steeper collapse confirms it learned to associate the monsoon season with the absence of fire, rather than learning which cells are genuinely fire-prone. Across the full year, the DC model shows a substantially more stable recall profile.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.90\textwidth]{fig_sampling_comparison_metrics.png}
    \caption{Overall performance comparison between Date-Conditional (DC) and Random-Uniform (RU) negative sampling on the 2024--2025 test set. DC achieves higher recall (+3.1 pp) and F1 at a marginally lower precision, reflecting its more aggressive fire-event detection. The header reports the month--probability Pearson correlation for each model; the higher absolute value for RU ($|r|=0.340$) confirms stronger seasonal shortcut dependency. AUC values are similar (0.961 vs.\ 0.960), confirming that the benefit of DC is concentrated in the operational recall metric rather than ranking alone.}
    \label{fig:sampling_metrics}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig_sampling_monthly_recall.png}
    \caption{Per-month recall on the 2024--2025 test set for Date-Conditional (DC) and Random-Uniform (RU) sampling. The shaded band marks the peak fire season (March--June), where both models perform similarly. The critical divergence occurs during monsoon months (July--September): the RU model's recall collapses to 18\% in July (vs.\ 34\% for DC) and 29\% in August (vs.\ 43\% for DC). This collapse is the hallmark of seasonal shortcut learning --- the RU model has learned that ``July is not fire season'' rather than discriminating fire-prone from fire-safe cells on any given day. The DC model, by construction forced to learn within-date spatial signals, maintains a substantially more stable recall across all months.}
    \label{fig:sampling_monthly}
\end{figure}

\subsection{Algorithms and Model Selection Rationale}

Five supervised classification algorithms were trained as base models. They span a range from simple, interpretable linear models to complex ensembles. These base models were then further combined into two ensemble configurations.

\subsubsection{Logistic Regression (LR)}

Logistic Regression serves as an interpretable linear baseline. Despite its simplicity, LR is surprisingly competitive in many tabular machine-learning benchmarks and provides a clear probabilistic model: the predicted fire probability at grid cell $(i, j)$ on date $t$ is:

\begin{equation}
P(\text{fire} \mid x_{ijt}) = \frac{1}{1 + e^{-(w^T x_{ijt} + b)}}
\end{equation}

where $x_{ijt}$ is the feature vector, $w$ are learned coefficients, and $b$ is a bias term. Including LR establishes a lower bound on performance and lets us check whether more complex models give genuine improvement or just overfit. The feature coefficients learned by LR also offer direct interpretability: positive coefficients point to features linked to increased fire risk, and negative coefficients indicate protective factors.

\subsubsection{Random Forest (RF)}

Random Forest is an ensemble of decision trees, each trained on a random sample of the data with random feature subsampling at each split. RF naturally handles nonlinear relationships, feature interactions, and mixed data types (continuous and categorical), and is robust to noisy or irrelevant features. In this work, RF serves as a non-linear baseline that is less aggressive than gradient boosting.

\subsubsection{LightGBM}

LightGBM is a histogram-based gradient boosting framework designed for speed and memory efficiency on large tabular datasets. LightGBM builds an ensemble of decision trees one after another, with each tree fitted to the residuals of the previous trees. The histogram-based approach reduces memory use and speeds up training by grouping continuous features into bins, making it well suited to the 9.4 million-sample dataset. LightGBM also has native support for categorical features (land-cover classes), so we do not need to manually one-hot encode them.

\subsubsection{CatBoost}

CatBoost is a gradient-boosting variant specifically designed to handle categorical features with less target leakage. Unlike standard gradient boosting, CatBoost uses ordered boosting, which calculates residuals on a separate subset of data to reduce overfitting caused by categorical-feature encoding. CatBoost also automatically applies permutation-based feature importances, making it useful for explainability analysis.

\subsubsection{XGBoost}

XGBoost is a regularized gradient boosting framework. XGBoost includes L1/L2 regularization on tree weights to prevent overfitting, and a learning-rate schedule (shrinkage) to improve generalization.

\subsubsection{Ensemble Models}

Beyond the five base classifiers, we built two ensemble strategies from the same base-model configurations and evaluated them as candidates for the production model:

\begin{itemize}
    \item \textbf{Voting Ensemble:} A soft-voting classifier that averages the predicted probabilities from all five base models. This is a simple, low-overhead way to smooth out individual model quirks without adding an extra learned combination layer.
    \item \textbf{Stacking Ensemble:} A stacked generalization model (two-level architecture) designed to learn the best way to combine base learner predictions:
    \begin{enumerate}
        \item \textbf{Level-0 Base Learners:} The five base classifiers described above (Logistic Regression, Random Forest, LightGBM, CatBoost, and XGBoost). Let $f_k(x)$ denote the predicted probability of fire for input sample $x$ by the $k$-th base learner ($k = 1, \dots, 5$).
        \item \textbf{Level-1 Meta-Learner:} A Logistic Regression model that takes the predictions of the base learners as inputs. The final predicted probability $\hat{p}(x)$ is computed as:
        \begin{equation}
        \hat{p}(x) = \sigma \left( \sum_{k=1}^{5} w_k f_k(x) + b \right) = \frac{1}{1 + e^{-\left(\sum_{k=1}^{5} w_k f_k(x) + b\right)}}
        \end{equation}
        where $w_k$ are the learned weights assigned to each base learner, $b$ is the bias, and $\sigma(\cdot)$ is the sigmoid activation function.
    \end{enumerate}
    To prevent overfitting and target leakage, the Level-1 meta-learner is trained strictly on the out-of-fold predictions generated by a 5-fold Stratified Cross-Validation (\texttt{StratifiedKFold(n\_splits=5, shuffle=True)}) over the training set. For each fold, base models are trained on the remaining 4 folds and used to generate predictions on the held-out fold. The meta-learner is then fitted on the resulting $N \times 5$ out-of-fold feature matrix (where $N$ is the training sample size). This ensures that the meta-learner is fitted on base learner outputs for samples they did not see during Level-0 training, preventing the meta-learner from overfitting to overly confident base model predictions.
\end{itemize}

Both ensembles are evaluated on the same held-out 2024--2025 test set, and the Stacking Ensemble is selected as the final production model.

\subsection{System Architecture and Temporal Validation Protocol}

The end-to-end pipeline consists of seven stages:

\begin{enumerate}
    \item \textbf{Raw Data Ingestion:} Parallel download of FIRMS fire detections, ERA5 meteorological fields, MODIS EVI, SRTM elevation, and ESA WorldCover from public APIs and cloud storage systems. Data are cached locally and validated against expected ranges and temporal coverage.

    \item \textbf{Spatial-Temporal Alignment and Quality Control:} Resampling and interpolation of multi-source data to the common 0.1 degree grid and daily temporal resolution. Cloud-masked EVI days are retained (not excluded) to provide vegetation information even under partial cloud cover; however, records with excessive missing data are filtered. Date-conditional negative sampling is applied to construct balanced training sets.

    \item \textbf{Feature Engineering:} Extraction of physics-informed features (cyclical month, meteorological variables, EVI, elevation, land-cover classes, interaction term). Feature normalization (standardization to zero mean and unit variance) is applied using statistics computed on the training set only, ensuring that test-set features are normalized using training-set statistics (preventing leakage).

    \item \textbf{Model Training with Strict Temporal Validation:} A critical aspect of this work is the validation protocol. Rather than applying random k-fold cross-validation or random train-test splits --- both of which leak temporal information --- a strict temporal holdout is employed:
    \begin{itemize}
        \item \textbf{Training set:} All data with acquisition year $\leq$ 2023.
        \item \textbf{Test set:} All data with acquisition year $\geq$ 2024, entirely unseen during training.
        \item \textbf{Cross-validation (for diagnostics):} \texttt{TimeSeriesSplit} with 5 folds, run exclusively on the training set to produce per-fold ROC-AUC estimates.
    \end{itemize}
    Each trained model (base and ensemble) is automatically checked against three diagnostic flags: an \textbf{overfit flag} (train AUC exceeding test or CV AUC by more than 0.05), an \textbf{underfit flag} (train AUC below 0.65), and a \textbf{suspicious flag} (any AUC $\geq$ 0.99, which would suggest possible data leakage or trivially separable classes rather than genuine signal). This protocol ensures that model generalization is assessed on data from a future time period, which realistically represents the deployment scenario where the system is asked to predict fire risk on dates it has never encountered during training.

    \item \textbf{Model Evaluation and Explainability:} After training, each model is evaluated on the held-out test set using accuracy, recall, precision, and ROC-AUC (described below). SHAP (SHapley Additive exPlanations) feature importance is computed using the TreeExplainer algorithm, which efficiently attributes each prediction to the contribution of individual features. SHAP values provide a principled game-theoretic foundation for model interpretability.

    \item \textbf{Probability Calibration:} The raw probability outputs of the selected Stacking Ensemble are passed through a fitted \textbf{Isotonic Regression} post-processor to produce well-calibrated fire probability estimates. The isotonic regressor is fitted on the chronologically earlier half of the held-out test set and evaluated on the later half, ensuring no leakage into the calibration evaluation. This two-stage predictor (Stacking Ensemble + Isotonic Regressor) is the artefact that is serialized and deployed. Raw ensemble probabilities may overestimate fire risk in the low-to-mid probability range due to class-imbalance effects during stacking; the calibration step corrects this and enables the probability outputs to be used directly as quantitative risk estimates (e.g., for expected-loss calculations or risk-tier assignment), not merely as relative rankings.

    \item \textbf{Deployment and Dashboard Integration:} The calibrated two-stage predictor is deployed via a FastAPI inference service, which accepts real-time weather forecasts from the Open-Meteo API and produces daily calibrated fire probability maps. An interactive React-based geospatial dashboard visualizes predicted fire risk, risk tiers (Low/Moderate/High/Extreme), SHAP-based feature importance, and indicative fire-spread direction vectors.
\end{enumerate}

\begin{figure}[H]
    \centering
    \begin{tikzpicture}[
        node distance=0.9cm,
        box/.style={
            rectangle, draw=black, thick, rounded corners,
            text width=10.5cm, minimum height=1cm,
            align=center, fill=gray!8, font=\small
        },
        arr/.style={-{Latex[length=2.5mm]}, thick}
    ]
        \node[box] (n1) {Raw Multi-Source Data (FIRMS, ERA5, MODIS, SRTM, WorldCover)};
        \node[box, below=of n1] (n2) {Spatial-Temporal Alignment \& Date-Conditional Negative Sampling};
        \node[box, below=of n2] (n3) {Feature Engineering (Physics-Informed Features)};
        \node[box, below=of n3] (n4) {Temporal Train/Test Split (Acq.\ Year $\leq$ 2023 / $\geq$ 2024)};
        \node[box, below=of n4] (n5) {Base Model Training (LR, RF, LightGBM, CatBoost, XGBoost)};
        \node[box, below=of n5] (n6) {Voting \& Stacking Ensembles $\rightarrow$ Best Model Selection};
        \node[box, below=of n6, fill=blue!8] (n7) {Probability Calibration (Isotonic Regression fitted on held-out calibration set)};
        \node[box, below=of n7] (n8) {Evaluation, SHAP Explainability \& Dashboard Deployment};
        \draw[arr] (n1) -- (n2);
        \draw[arr] (n2) -- (n3);
        \draw[arr] (n3) -- (n4);
        \draw[arr] (n4) -- (n5);
        \draw[arr] (n5) -- (n6);
        \draw[arr] (n6) -- (n7);
        \draw[arr] (n7) -- (n8);
    \end{tikzpicture}
    \caption{Proposed System Workflow}
    \label{fig:workflow}
\end{figure}

\subsubsection{Spatial Autocorrelation and Data Leakage Considerations}

Spatial autocorrelation is the tendency of nearby locations to have more similar feature values than distant ones (Tobler's First Law of Geography). It can introduce a form of data leakage when training and test cells are geographically close. This section explains how the pipeline design limits this concern.

\medskip
\noindent\textbf{Why spatial leakage is structurally limited here.}
The main weather covariates (temperature, humidity, wind speed, VPD) come from ERA5 reanalysis at a native resolution of 0.25$^\circ$ ($\approx$28~km). The prediction grid is at 0.1$^\circ$ ($\approx$11~km). This means that \textit{up to six adjacent 0.1$^\circ$ cells fall within a single ERA5 tile and share an identical weather feature vector by construction} --- bilinear spatial interpolation cannot create finer-scale weather gradients than exist in the source data. The model therefore \textit{cannot} exploit hyper-local weather autocorrelation because no such signal exists in the ERA5-derived features at 11~km.

The static features --- elevation from SRTM ($\approx$90~m), land cover from ESA WorldCover ($\approx$10~m), and EVI from MODIS ($\approx$500~m) --- do vary at finer spatial scales, but these layers do not change from year to year. Since the temporal holdout places all training data in 2018--2023 and all test data in 2024--2025, spatial proximity between a training cell and a test cell at the same geographic location does not introduce leakage: the static features at that location are identical in both sets by definition, and the model is evaluated on its ability to generalise across \textit{time} (a new fire season), not across space.

\medskip
\noindent\textbf{Role of date-conditional sampling.}
Beyond the above structural constraints, the Date-Conditional Negative Sampling strategy (Section~\ref{sec:dcns}) makes within-date spatial discrimination the model's primary training objective. Because positive and negative examples within each training batch share the same regional weather snapshot, the model is explicitly trained to identify which \textit{local} combinations of terrain, vegetation, and land cover distinguish fire-prone cells from neighbouring fire-safe cells on the same day. This is exactly the spatial learning signal needed for operational deployment --- and it is separate from the temporal generalisation guarantee provided by the chronological holdout.

\medskip
\noindent\textbf{Remaining limitation.}
Full spatial block cross-validation --- in which entire geographic regions are held out during training and used only for evaluation --- would give the most rigorous estimate of spatial generalisation, especially for quantifying performance in data-sparse regions of North-East India. We acknowledge this as a priority for future work. For this study, the combination of strict temporal holdout and ERA5's inherent spatial resolution limit provides the primary leakage safeguards.

\subsection{Performance Evaluation Metrics}

Model performance was evaluated using standard classification metrics, each chosen for its operational relevance to wildfire early-warning:

\subsubsection{Accuracy}

Overall accuracy is the proportion of correct predictions (both fire and non-fire) across the entire test set:

\begin{equation}
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\end{equation}

where TP (true positive) denotes correctly predicted fires, TN (true negative) denotes correctly predicted non-fires, FP (false positive) denotes non-fires mispredicted as fires, and FN (false negative) denotes fires mispredicted as non-fires. Accuracy is commonly reported but can be misleading in imbalanced datasets; a naive classifier predicting all non-fire would achieve 99.9\% accuracy while catching zero fires.

\subsubsection{Recall (Fire-Event Sensitivity)}

Recall measures the proportion of actual fire events successfully identified by the model:

\begin{equation}
\text{Recall} = \frac{TP}{TP + FN}
\end{equation}

Recall is the primary safety-critical metric in this work, since a missed fire event (false negative) has enormous operational cost: delayed suppression allows the fire to grow and cause damage. In contrast, a false alarm (false positive) has minimal cost --- it simply prompts additional surveillance of a non-burning location. For operational fire-warning systems, high recall is therefore essential.

\subsubsection{Precision}

Precision is the proportion of predicted fires that are actually fires:

\begin{equation}
\text{Precision} = \frac{TP}{TP + FP}
\end{equation}

Precision indicates how many of the system's fire alerts are true alarms versus false alarms. Excessively low precision wastes fire-management resources on false alerts.

\subsubsection{F1 Score}

The F1 score is the harmonic mean of precision and recall:

\begin{equation}
F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\end{equation}

F1 provides a single summary metric balancing precision and recall.

\subsubsection{Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC)}

The ROC curve plots the true-positive rate (recall) against the false-positive rate ($FP/(FP + TN)$) as the classification threshold is varied from 0 to 1. The ROC-AUC is the area under this curve, ranging from 0.5 (random guessing) to 1.0 (perfect classification). ROC-AUC is threshold-independent and thus suitable for comparing models when the optimal operating point (balance of recall and precision) is not known in advance. For this work, ROC-AUC is reported as the primary comparison metric across the five base classifiers and the two ensembles, because it is robust to class imbalance and can be interpreted as the probability that the model ranks a random fire event higher than a random non-fire event.

% ============================================================
%  RESULTS
% ============================================================
\section{Results}

\subsection{Model Comparison and Selection}

Figure~\ref{fig:roc_all} shows the ROC curves for all seven candidate models (five base classifiers plus the two ensemble configurations) evaluated on the held-out 2024--2025 test set. Logistic Regression achieves a test ROC-AUC of 0.897, confirming that the physics-informed feature set carries substantial linear signal. All four non-linear models and both ensembles cluster tightly in the range 0.956--0.961, with XGBoost leading at AUC = 0.961.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\textwidth]{fig_roc_curves_all_models.png}
    \caption{ROC curves for all seven candidate models evaluated on the held-out 2024--2025 test set. Logistic Regression (AUC~=~0.897) serves as the linear baseline. All gradient-boosted models and ensembles cluster between AUC~=~0.956 and AUC~=~0.961, with XGBoost achieving the highest discriminative power. The dashed diagonal represents random-chance performance (AUC~=~0.5).}
    \label{fig:roc_all}
\end{figure}

Figure~\ref{fig:pr_all} shows the corresponding Precision-Recall curves. Under severe class imbalance, the PR curve is more informative than the ROC curve because it focuses on the rare positive (fire) class. XGBoost again leads with PR-AUC = 0.950, versus 0.857 for Logistic Regression --- a gap of nearly ten points that translates directly into more reliable fire alerts at any fixed recall target.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\textwidth]{fig_pr_curves_all_models.png}
    \caption{Precision-Recall curves for all seven candidate models on the 2024--2025 test set. The PR curve is particularly informative under class imbalance because it highlights performance on the rare fire class. XGBoost achieves the highest PR-AUC (0.950), followed by LightGBM (0.945). The Logistic Regression baseline (PR-AUC = 0.857) is clearly separated, illustrating the benefit of nonlinear modeling for this task.}
    \label{fig:pr_all}
\end{figure}

Figure~\ref{fig:metrics_all} provides a side-by-side bar chart of Accuracy, Precision, Recall, F1 Score, and ROC-AUC across all seven models. XGBoost achieves the highest recall (0.9606) and ROC-AUC (0.961) among base models, making it the most operationally preferred classifier for a fire early-warning context where missing a real fire event carries the highest cost.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig_metrics_comparison_all_models.png}
    \caption{Side-by-side comparison of five evaluation metrics (Accuracy, Precision, Recall, F1 Score, and ROC-AUC) across all seven candidate models on the held-out test set. XGBoost achieves the highest recall (0.9606) among base models, which is the primary operational metric for fire early-warning. All gradient-boosted models substantially outperform the Logistic Regression baseline on every metric.}
    \label{fig:metrics_all}
\end{figure}

Figure~\ref{fig:auc_rank} ranks all seven models by test ROC-AUC, distinguishing base models (blue) from ensemble configurations (orange). XGBoost ranks first (0.9613), with the Stacking Ensemble selected as the production model because it achieves the highest ensemble AUC (0.9578) and is deployed for inference.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.88\textwidth]{fig_auc_ranking.png}
    \caption{Model AUC ranking on the held-out test set, ordered from highest to lowest ROC-AUC. Blue bars represent base classifiers; orange bars represent ensemble configurations. XGBoost achieves the highest test AUC among base models (0.9613). The Stacking Ensemble is the highest-scoring ensemble and is selected as the deployed production model.}
    \label{fig:auc_rank}
\end{figure}

Table~\ref{tab:model_results} consolidates all six evaluation metrics for every candidate model on the held-out test set, along with two simpler baseline models representing traditional warning approaches: a 2-Feature Logistic Regression (trained only on temperature and wind speed) and a Threshold Meteorological Heuristic (simulating an expert-rule warning trigger requiring temperature $> 30^\circ\text{C}$, relative humidity $< 35\%$, and $\text{VPD} > 2.0~\text{kPa}$). XGBoost leads among base classifiers on five of six metrics; the Stacking Ensemble is selected as the production model because it achieves the highest ensemble-level test AUC (0.9578) with the best precision (0.8937) among all machine learning configurations, giving the most balanced trade-off between false alarms and missed fires.

\begin{table}[H]
    \centering
    \renewcommand{\arraystretch}{1.3}
    \resizebox{\textwidth}{!}{%
    \begin{tabular}{clcccccc}
        \toprule
        \textbf{Rank} & \textbf{Model} & \textbf{Train AUC} & \textbf{Test AUC} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\
        \midrule
        1 & XGBoost                           & 0.9676 & 0.9613 & 0.9018 & 0.8602          & \textbf{0.9606} & 0.9076 \\
        2 & LightGBM                          & 0.9629 & 0.9586 & 0.9003 & 0.8804          & 0.9274          & 0.9033 \\
        3 & Random Forest                     & 0.9702 & 0.9583 & 0.9015 & 0.8835          & 0.9260          & 0.9043 \\
        4 & Stacking Ensemble\,$\star$         & 0.9746 & 0.9578 & 0.8982 & \textbf{0.8937} & 0.9049          & 0.8993 \\
        5 & Voting Ensemble                   & 0.9618 & 0.9570 & 0.9016 & 0.8722          & 0.9421          & 0.9058 \\
        6 & CatBoost                          & 0.9590 & 0.9560 & 0.8976 & 0.8743          & 0.9297          & 0.9011 \\
        7 & Logistic Regression (Full)        & 0.8866 & 0.8968 & 0.8258 & 0.8036          & 0.8645          & 0.8329 \\
        \midrule
        -- & 2-Feature LR (Temp + Wind)       & 0.8092 & 0.8159 & 0.7405 & 0.7094          & 0.8184          & 0.7601 \\
        -- & Threshold Meteorological Heuristic& ---    & 0.6976 & 0.6965 & 0.8886          & 0.4524          & 0.5996 \\
        \bottomrule
    \end{tabular}%
    }
    \caption{Complete performance comparison of all candidate models on the held-out 2024--2025 test set, ranked by test ROC-AUC. Boldface indicates the best value per column ($\star$~=~selected production model). All gradient-boosted models and ensembles substantially outperform the full Logistic Regression and traditional baseline models on every metric. XGBoost achieves the highest recall (0.9606) among base models. The Stacking Ensemble is selected as the deployed production model due to its leading precision (0.8937) and highest ensemble-level AUC. The two heuristic baselines (bottom) show a massive performance drop (e.g., up to 45.2\% recall loss), providing clear justification for deploying our complex machine learning system over traditional meteorological indices.}
\label{tab:model_results}
\end{table}

\subsection{Heuristic Baseline Comparison and Rationale}
\label{sec:heuristic_comparison}

To evaluate whether the operational complexity of the machine learning pipeline is justified, we benchmarked the models against two traditional warning strategies: a two-feature Logistic Regression and a hard-coded meteorological threshold rule. As shown in Table~\ref{tab:model_results}, the Stacking Ensemble outperforms the simple 2-Feature LR by 14.19 percentage points in Test AUC and 8.65 percentage points in Recall, while also improving Precision by 18.43 percentage points.

The comparison with the Threshold Meteorological Heuristic is even more striking: the heuristic achieves a high precision ($0.8886$) because it only triggers warnings under highly restricted, severe weather conditions, but it has a disastrously low recall of only $0.4524$. This means the heuristic misses more than $54\%$ of all wildfire occurrences. The Stacking Ensemble recovers these missed events, increasing Recall by $45.25$ percentage points (from $0.4524$ to $0.9049$) while maintaining a comparable precision ($0.8937$). This large margin of improvement shows that simple meteorological triggers are not sufficient for continent-scale fire warning systems in India, where fire susceptibility is heavily shaped by spatial differences in fuel moisture (EVI) and terrain topography rather than meteorological variables alone.

\subsection{Overfitting Diagnostic}

Figure~\ref{fig:train_test_auc} compares train-set AUC against test-set AUC for every model. The small gap between the two bars for each model (less than 0.03 in all cases) confirms that none of the models are materially overfitting to the training data. Logistic Regression is the only model where test AUC marginally exceeds train AUC, a sign of the linear model being a slightly conservative fit to the high-dimensional feature space rather than being overfit.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.92\textwidth]{fig_train_vs_test_auc_overfitting.png}
    \caption{Train vs.\ test ROC-AUC comparison across all seven models, serving as an overfitting diagnostic. The near-identical bar heights for all gradient-boosted models and ensembles (gap $<$0.03) indicate strong generalization to the held-out 2024--2025 test period. No model exceeds the overfitting flag threshold of a 0.05 AUC gap, confirming robustness of the temporal validation protocol.}
    \label{fig:train_test_auc}
\end{figure}

\subsection{Stacking Ensemble Detailed Evaluation}

The Stacking Ensemble, selected as the production model, is evaluated in detail in this section. Figure~\ref{fig:stack_cm} shows the confusion matrix for the Stacking Ensemble on the full held-out 2024--2025 test set. The model correctly identifies 985,285 true fire events (TP) while missing 103,582 (FN), and correctly classifies 962,476 non-fire instances (TN) while generating 117,166 false alarms (FP). Relative to the base models, the Stacking Ensemble achieves the highest precision (0.8937) --- meaning fewer false alarms --- at the cost of a moderate reduction in raw recall, reflecting the meta-learner's conservative aggregation of base model votes.

\begin{figure}[H]
    \begin{center}
        \includegraphics[width=0.72\textwidth]{fig_stacking_confusion_matrix.png}
    \end{center}
    \caption{Confusion matrix for the Stacking Ensemble on the 2024--2025 held-out test set. True Positives (985,285) and True Negatives (962,476) dominate the matrix. The 103,582 False Negatives represent missed fire events---the operationally most costly error type---while 117,166 False Positives represent unnecessary alerts. The higher precision relative to the individual base models confirms that the meta-learner successfully filters noise from individual base model predictions, reducing false alarms.}
    \label{fig:stack_cm}
\end{figure}

Figure~\ref{fig:stack_lc} shows the Stacking Ensemble learning curve, plotting train-set AUC and cross-validation (TimeSeriesSplit) AUC against increasing numbers of training examples. Both curves stabilize rapidly above 0.96 and remain within 0.02 of each other throughout, confirming the ensemble is not data-hungry and generalizes robustly to unseen temporal periods.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.88\textwidth]{fig_stacking_learning_curve.png}
    \caption{Learning curve for the Stacking Ensemble, showing train-set ROC-AUC (blue) and TimeSeriesSplit cross-validation AUC (orange) as a function of training set size. Both curves stabilize above 0.96 after approximately 50,000 training examples, indicating that the ensemble is not data-limited. The narrow and stable train--CV gap confirms the absence of problematic overfitting across temporal folds.}
    \label{fig:stack_lc}
\end{figure}

\subsection{SHAP Explainability}

Two complementary SHAP analyses are presented for the Stacking Ensemble. The first (meta-learner SHAP) examines how the Logistic Regression meta-learner combines the five base model probability outputs. The second (permutation SHAP) attributes model predictions back to the original 20 physics-informed input features, using \texttt{shap.PermutationExplainer} --- a model-agnostic method that perturbs features and measures the resulting change in prediction, applicable to any black-box model including the full stacking pipeline.

\subsubsection{Meta-Learner SHAP: Base Model Trust Analysis}

Figure~\ref{fig:shap_meta_bar} presents the SHAP feature importance bar chart for the Logistic Regression meta-learner, where each ``feature'' is the calibrated probability output of one of the five base models: $P(\text{fire} \mid \text{BaseModel}_k)$. This analysis shows how the meta-learner weights and combines the individual model votes.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.82\textwidth]{fig_shap_meta_importance_stacking.png}
    \caption{SHAP feature importance bar chart for the Stacking Ensemble meta-learner (Logistic Regression), showing mean absolute SHAP values for each base model's probability output. The dominant features correspond to the base models that carry the most unique discriminative signal. A base model with a large SHAP value indicates that the meta-learner assigns it the highest trust when making the final fire prediction, while a near-zero SHAP value indicates redundancy with other base models.}
    \label{fig:shap_meta_bar}
\end{figure}

Figure~\ref{fig:shap_meta_summary} shows the corresponding SHAP beeswarm plot for the meta-learner, encoding both the direction (positive vs.\ negative SHAP value) and magnitude of each base model's influence on the final prediction. High predicted probabilities from a trusted base model (pink dots at positive SHAP values) push the ensemble toward a fire prediction; low probabilities from the same model (blue dots at negative SHAP values) suppress the prediction.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.88\textwidth]{fig_shap_meta_summary_stacking.png}
    \caption{SHAP summary beeswarm plot for the Stacking Ensemble meta-learner. Each point represents one test-set sample; horizontal position encodes the SHAP value (impact on meta-learner output), and color encodes the base model's predicted fire probability (pink = high, blue = low). Features are ordered top-to-bottom by mean absolute SHAP value. The widest-ranging feature corresponds to the base model whose probability output exerts the largest influence on the stacked prediction.}
    \label{fig:shap_meta_summary}
\end{figure}

\subsubsection{Permutation SHAP: Input Feature Attribution}

Figures~\ref{fig:shap_perm_summary} and~\ref{fig:shap_perm_bar} present the full SHAP analysis for the Stacking Ensemble at the level of original input features, computed using \texttt{shap.PermutationExplainer} on a stratified 800-sample subsample of the test set. This method is model-agnostic: it measures the expected change in fire probability when each feature is permuted, making it applicable to the full stacking pipeline. The results are directly comparable to any single-model SHAP analysis while correctly attributing credit to the stacked system rather than any individual base model.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.82\textwidth]{fig_shap_summary_stacking.png}
    \caption{SHAP summary beeswarm plot for the Stacking Ensemble (PermutationExplainer, original 20 input features). Each point represents one test-set sample; horizontal position encodes the SHAP value (impact on predicted fire probability), and color encodes the feature value (pink = high, blue = low). Features are ordered top-to-bottom by mean absolute SHAP value. Elevation (\texttt{elevation\_m}) and EVI are the dominant suppressors of fire probability (large negative SHAP values at high values), while high VPD and temperature strongly increase predicted risk---consistent with established fire-weather science.}
    \label{fig:shap_perm_summary}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.82\textwidth]{fig_shap_importance_stacking.png}
    \caption{SHAP feature importance bar chart for the Stacking Ensemble (PermutationExplainer), showing mean absolute SHAP values averaged over the test subsample. Elevation (\texttt{elevation\_m}), Enhanced Vegetation Index (\texttt{evi}), and Vapor Pressure Deficit (\texttt{vpd}) are the three dominant predictors---the same top-three as observed in the individual base models, confirming that the ensemble preserves the physics-informed hierarchy of feature importance. Land-cover class indicators (\texttt{lc\_*}) contribute minimal individual signal.}
    \label{fig:shap_perm_bar}
\end{figure}

Figure~\ref{fig:shap_dep_elev} shows the SHAP dependence plot for elevation, revealing the nonlinear relationship between elevation and predicted fire risk in the Stacking Ensemble. At low elevations (0--500~m), SHAP values span a wide range near zero, indicating that elevation alone is not the primary factor in lowland areas and that weather and vegetation features dominate. Above approximately 1000~m, SHAP values for elevation become increasingly negative, reflecting the strong suppressive effect of altitude on fire probability that sharpens dramatically at Himalayan elevations (above 4000~m).

\begin{figure}[H]
    \centering
    \includegraphics[width=0.78\textwidth]{fig_shap_dependence_elevation_stacking.png}
    \caption{SHAP dependence plot for elevation (\texttt{elevation\_m}) in the Stacking Ensemble (PermutationExplainer). The x-axis shows the actual elevation value (in metres) for each test-set sample; the y-axis shows the corresponding SHAP contribution to the predicted fire probability. At low elevations ($<$500~m), the SHAP contribution is close to zero, indicating that elevation is not the primary driver in lowland contexts. Above 1000~m, SHAP values become increasingly negative, capturing the well-established suppressive effect of altitude on fire ignition. The effect sharpens dramatically above 4000~m, consistent with sub-alpine and alpine conditions in the Himalayas.}
    \label{fig:shap_dep_elev}
\end{figure}

\subsubsection{Zone-Wise Feature Importance and Ecological Interpretation}
\label{sec:shap_zones}

The global SHAP summaries above average over all of India's diverse ecological contexts. To check whether the model has learned ecologically meaningful zone-specific signals, Figure~\ref{fig:shap_zones} presents a Permutation SHAP analysis broken down by six macro agro-ecological zones, defined by latitude/longitude bounding boxes (Table~\ref{tab:zones}).

\begin{table}[H]
\centering
\renewcommand{\arraystretch}{1.2}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lllll}
\toprule
\textbf{Zone} & \textbf{Key States} & \textbf{Lat} & \textbf{Lon} & \textbf{Primary Fire Driver} \\
\midrule
Central Dry Deciduous   & MP, Chhattisgarh, Odisha & 18--26\textdegree N & 74--86\textdegree E & VPD + temperature (moisture stress) \\
North-East Moist        & Assam, Mizoram, Nagaland & 22--28\textdegree N & 89--97\textdegree E & EVI seasonality + cyclical month \\
Western Himalaya        & Himachal, Uttarakhand    & 28--33\textdegree N & 75--82\textdegree E & Elevation (suppressive) + month \\
Peninsular Coast \& Ghats & Kerala, TN, Karnataka  & 8--16\textdegree N  & 74--80\textdegree E & VPD + wind (fire-spread driven) \\
Indo-Gangetic Plains    & UP, Bihar, Punjab        & 26--32\textdegree N & 73--89\textdegree E & Temperature + humidity \\
Rajasthan Arid          & Rajasthan                & 24--30\textdegree N & 69--75\textdegree E & Month (sin) + VPD \\
\bottomrule
\end{tabular}%
}
\caption{Agro-ecological zone definitions used for zone-wise SHAP analysis.}
\label{tab:zones}
\end{table}

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{fig_shap_zone_comparison.png}
    \caption{Zone-wise mean absolute SHAP values (Stacking Ensemble, PermutationExplainer, 200 samples per zone) for the top 12 features ranked by overall mean importance. Elevation is the dominant feature across all zones, consistent with its role as a long-range fire suppressor, but its relative weight differs markedly: in the Western Himalaya (green), elevation has the largest absolute SHAP value of any feature-zone combination, while in the Indo-Gangetic Plains (purple), which are largely flat, its importance is lower and cyclical month signals dominate. VPD is most important in the Central Dry Deciduous zone (red) and Peninsular Coast (orange), consistent with moisture-stress-driven ignition in those ecosystems. In the North-East Moist zone (blue), EVI has elevated importance relative to other zones, reflecting the region's dense, seasonally varying forest canopy as the primary discriminator between fire-prone and safe conditions. The Rajasthan Arid zone (brown) shows the highest month~(sin) importance, consistent with the strongly seasonal nature of its sparse vegetation-fire interaction.}
    \label{fig:shap_zones}
\end{figure}

The zone-differentiated patterns confirm that the Stacking Ensemble has not learned a single global shortcut, but instead captures ecologically distinct fire-risk mechanisms. This is consistent with the date-conditional training design: by forcing the model to discriminate fire from non-fire within the same weather context on each training date, it must learn the local spatial-environmental signals that vary by zone.

\subsubsection{VPD--EVI Feature Interaction and Fire-Ecology Threshold}
\label{sec:shap_vpd_evi}

Vapor Pressure Deficit (VPD) and EVI are the two most important meteorological and vegetation-state features, and their interaction is central to fire ecology: high VPD means strong atmospheric drying demand (the fire ``weather'' driver), while low EVI means reduced live-fuel moisture (the fire ``fuel'' driver). Neither variable alone captures the full picture --- a dry atmosphere is not enough to ignite green fuel, and dry fuel needs a drying trigger. Figure~\ref{fig:shap_vpd_evi} quantifies this joint threshold effect directly.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.88\textwidth]{fig_shap_vpd_evi_interaction.png}
    \caption{Joint SHAP interaction surface for Vapor Pressure Deficit (VPD, y-axis) and EVI (x-axis). Each hexagonal bin shows the mean combined SHAP contribution SHAP(VPD)~+~SHAP(EVI) for test-set samples falling within that VPD--EVI range. The dashed contour marks the boundary where combined SHAP~=~0 (fire-risk neutral). The upper-left region (low EVI, high VPD) produces strongly positive combined SHAP values (deep red), indicating that this combination drives fire probability far above the model's base rate. The lower-right region (high EVI, low VPD) produces strongly negative combined SHAP values (deep green), indicating strong fire suppression. The non-linear threshold crossing from positive to negative risk is consistent with established fire-ecology theory: the multiplicative interaction between atmospheric drying demand and live-fuel moisture creates a step-change in fire susceptibility that neither feature can describe alone. Annotations mark the dominant ecological interpretations of each quadrant.}
    \label{fig:shap_vpd_evi}
\end{figure}

The SHAP = 0 contour in Figure~\ref{fig:shap_vpd_evi} represents the model's learned fire-ecology threshold: locations with (VPD, EVI) combinations above-left of this boundary are classified as fire-prone, while those below-right are fire-safe. This threshold is non-linear and context-dependent, which explains why simple linear rules (e.g., ``fire risk if VPD $>$ threshold'') underperform relative to the gradient-boosted ensemble: the true boundary depends on the concurrent state of both variables simultaneously.

\subsection{Probability Calibration}

The raw probability outputs of the Stacking Ensemble are assessed for calibration quality and corrected using Isotonic Regression post-processing. A well-calibrated probabilistic classifier produces outputs $\hat{p}$ such that, among all predictions assigned probability $\hat{p}$, the true proportion of fires is also $\hat{p}$. Calibration quality is measured using the Expected Calibration Error (ECE), defined as the frequency-weighted average absolute difference between predicted confidence and observed accuracy over $K$ equally-spaced probability bins:

\begin{equation}
\text{ECE} = \sum_{k=1}^{K} \frac{|B_k|}{N} \left| \overline{p}_k - \overline{y}_k \right|
\end{equation}

where $|B_k|$ is the number of samples in bin $k$, $N$ is the total number of samples, $\overline{p}_k$ is the mean predicted probability in bin $k$, and $\overline{y}_k$ is the observed fire rate in bin $k$.

The calibration procedure uses a strict chronological split to prevent data leakage. The 2024--2025 held-out test set (2,168,509 samples) is divided into two consecutive halves by acquisition date:

\begin{itemize}
    \item \textbf{Calibration fitting set:} First chronological half (1,084,254 samples, up to December 2024). The Isotonic Regression post-processor is fitted on this subset using the raw ensemble probabilities as input and the binary fire labels as targets.
    \item \textbf{Calibration evaluation set:} Second chronological half (1,084,255 samples, from December 2024). The fitted post-processor is applied to the raw probabilities and all reliability diagrams and ECE metrics are computed exclusively on this set.
\end{itemize}

This split mirrors the production scenario where the calibrator is fitted on a recent historical holdout and then applied to future, unseen predictions.

Figure~\ref{fig:calibration} presents the before/after reliability diagram. The left panel shows the uncalibrated Stacking Ensemble, which consistently overestimates fire probability in the low-to-mid probability range ($0.0$--$0.6$): a model output of $\hat{p} = 0.5$ corresponds to an observed fire rate of only about 0.67, indicating that the raw probabilities are inflated in this range. This is a well-known phenomenon in stacked classifiers trained on class-imbalanced data, where the meta-learner's logistic output is compressed and shifted by the base model probability distributions. The uncalibrated ECE is 0.0281.

The right panel shows the post-calibrated system after Isotonic Regression is applied. The reliability curve lies almost exactly on the diagonal across the entire probability range, confirming that the calibrated probabilities are accurate quantitative estimates of observed fire frequency. The ECE drops from 0.0281 to \textbf{0.0091} --- a \textbf{67.7\% reduction} --- representing a substantial improvement in probabilistic reliability.

\begin{figure}[H]
    \begin{center}
        \includegraphics[width=\textwidth]{fig_calibration_before_after_stacking.png}
    \end{center}
    \caption{Reliability diagrams for the Stacking Ensemble before (left) and after (right) Isotonic Regression post-calibration, evaluated on the chronologically later half of the 2024--2025 test set. The dashed diagonal represents perfect calibration. Before calibration (ECE~=~0.0281), the raw ensemble probabilities systematically overestimate fire risk in the 0.0--0.6 range (large shaded gap above the diagonal). After Isotonic Regression is applied---fitted on the earlier half of the test set to prevent leakage---the reliability curve lies almost exactly on the diagonal across all bins (ECE~=~0.0091, a 67.7\% reduction). The calibrated two-stage system (Stacking Ensemble + Isotonic Regressor) produces fire probability estimates that are directly interpretable as quantitative fire risk, enabling their use in expected-loss calculations, risk-tier assignment, and probabilistic resource allocation models.}
    \label{fig:calibration}
\end{figure}

Isotonic Regression was preferred over Platt scaling (logistic post-processing) for this application because: (i) the miscalibration pattern is non-sigmoidal and non-monotonic, which Platt scaling cannot fully correct; and (ii) the very large calibration fitting set (1.08M samples) provides sufficient density to learn the isotonic mapping non-parametrically without overfitting.

The production pipeline therefore ships two serialized artefacts: the fitted Stacking Ensemble and the fitted IsotonicRegression object. At inference time, the FastAPI service applies both in sequence --- the ensemble produces raw probabilities from weather and terrain inputs, and the isotonic regressor converts them to calibrated fire probabilities before serving them to the dashboard and downstream consumers.

\section{Limitations and Future Work}
\label{sec:limitations}

While the Stacking Ensemble shows excellent predictive power and robustness, several limitations present opportunities for future work:

\begin{enumerate}
    \item \textbf{Ignition Source Proxies:} A fundamental challenge in wildfire prediction is distinguishing between fire susceptibility (the fuel and meteorological conditions that allow a fire to burn) and fire ignition (the spark that starts it). Our feature set captures fuel status (EVI, land cover) and weather (VPD, temperature, wind), but does not explicitly incorporate ignition proxies like human-activity indices (distance to roads, population density, power line grids) or lightning strike density. Although static land-use indicators (e.g., \texttt{lc\_built\_up}) act as weak spatial proxies, future work will integrate dynamic human-mobility datasets and real-time lightning detection feeds to refine risk predictions at the sub-grid scale.
    \item \textbf{Grid Resolution and Sub-Grid Heterogeneity:} The meteorological features are derived from ERA5 reanalysis at $0.25^\circ$ resolution ($\approx 28~\text{km}$). While our validation proves this scale is highly effective for regional risk assessment, it inevitably smooths out local micro-climates, wind tunnels, and narrow valleys. Future iterations will explore downscaling meteorology using high-resolution regional models like IMDAA ($5~\text{km}$).
    \item \textbf{Drought Index Redundancy:} Traditional fire management relies heavily on drought indices such as the Keetch--Byram Drought Index (KBDI). During our early feature selection phase, KBDI was evaluated but ultimately rejected because its long-term moisture accumulation signal was largely redundant with Vapor Pressure Deficit (VPD). Furthermore, daily precipitation data quality varies significantly across India's monsoon zones, introducing noise into KBDI calculations. Acknowledging this redundancy, future efforts will focus on incorporating direct satellite-derived soil moisture measurements (e.g., SMAP) to complement EVI.
    \item \textbf{Temporal Extrapolation:} The model is trained on 2017--2023 data and evaluated on 2024--2025. While this strict chronological split proves temporal robustness, long-term climate shifts (such as changing monsoon patterns in India) may alter fire-weather relationships over decade-long horizons. Continuous, automated retraining cycles will be necessary to ensure model calibration remains valid in the face of ongoing climate changes.
\end{enumerate}

\end{document}
"""

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(SIMPLIFIED)

print(f"Written {len(SIMPLIFIED)} characters to {OUTPUT_PATH}")
