from fpdf import FPDF
import sys

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        # self.cell(30, 10, 'WILDFIRE PREDICTION SYSTEM (INDIAN REGION)', 0, 0, 'C')
        # Line break
        # self.ln(20)

    def chapter_title(self, num, title):
        # Arial 12
        self.set_font('Arial', 'B', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, title, 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def chapter_body(self, body):
        # Times 12
        self.set_font('Times', '', 11)
        # Output justified text
        self.multi_cell(0, 5, body)
        # Line break
        self.ln()

    def print_chapter(self, num, title, name):
        self.chapter_title(num, title)
        self.chapter_body(name)

pdf = PDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "WILDFIRE PREDICTION SYSTEM (INDIAN REGION)", 0, 1, "C")
pdf.set_font("Arial", "B", 12)
pdf.cell(0, 10, "============================================================", 0, 1, "C")
pdf.cell(0, 10, "System Architecture and Working Mechanism Report", 0, 1, "C")
pdf.ln(5)

section1_body = """This codebase is a machine learning framework designed to predict the risk of wildfires in the Indian region on a daily basis. By analyzing past historical fire events (from 2018 to 2025) and learning the specific weather and environmental conditions present during those fires, the system can predict the likelihood of new fires occurring the next day.

It uses tabular data, which means data arranged in rows and columns like a spreadsheet. The key environmental factors it uses are:
- Weather Data (from ERA5): Temperature, Humidity, Wind speed, and Vapor Pressure Deficit (VPD, which measures air dryness).
- Vegetation Data (from MODIS satellite): Enhanced Vegetation Index (EVI), which tells us how green or dry the plants are.
- Geospatial Data: Elevation and land cover classifications (like tree cover, grassland, built-up areas, etc.).

The end goal is to generate a risk score between 0 and 1 for specific locations, providing an early warning system for forest management and firefighting operations."""
pdf.print_chapter(1, "1. INTRODUCTION\n------------------------------------------------------------", section1_body)

section2_body = """The system architecture follows a clean, organized process to transform raw environmental numbers into highly accurate risk predictions.

- Tabular Data Processing Pipeline: The system first ingests raw fire locations (from NASA FIRMS) and matches each fire with the exact weather and vegetation data for that location on the day before the fire (a 1-day lag). To help the model learn what a "no fire" situation looks like, it synthetically generates negative samples (non-fire events) to create a perfectly balanced training dataset.
- Feature Engineering: To make the raw data more useful for the models, the system adds new calculated columns. It converts the month into a mathematical cycle (using sine and cosine) so the models understand that December is right next to January. It also multiplies Vapor Pressure Deficit (VPD) by Wind to create an 'interaction feature' that highlights extreme fire weather conditions.
- Model Assembly/Ensembling: Instead of relying on just one algorithm, the architecture uses five different machine learning models working in parallel. Once all five models make their individual predictions, the system uses two Ensembling techniques (a "Voting Ensemble" and a "Stacking Ensemble") to combine their guesses into a single, much more reliable and robust final prediction."""
pdf.print_chapter(2, "2. OVERALL SYSTEM ARCHITECTURE\n------------------------------------------------------------", section2_body)

section3_body = """Step 1: Data Ingestion & Preprocessing
Historical fire data is loaded and merged with weather data from the previous day. Missing values are filled in using the median of the data (SimpleImputer). For models that require normalized data (like Logistic Regression), the numbers are scaled to a standard range (StandardScaler). Finally, the data is split by time: 2018-2023 is used for training the models, and 2024-2025 is kept completely separate for testing, ensuring the models are evaluated fairly on future data.

Step 2: Feature Selection & Extraction
The system extracts and relies on 20 specific features. These include temporal features (month_sin, month_cos), weather features (temp, humidity, wind, vpd), vegetation health (evi), and physics-informed interactions (vpd_wind). It also includes geospatial information like elevation (elevation_m) and 11 distinct land cover types (e.g., lc_tree_cover, lc_grassland, lc_cropland).

Step 3: Model Training & Tuning
Five base models are trained on the 2018-2023 dataset. To handle the massive amounts of data, GPU acceleration is utilized for models that support it. The models use a TimeSeriesSplit cross-validation, meaning they test themselves sequentially through time to mimic real-world forecasting.

Step 4: Final Risk Score Prediction
The Stacking Ensemble takes over, using the outputs from the five base models as its own inputs to make a final decision. It outputs a final probability score between 0.0 and 1.0. If the score is above 0.50, it is classified as a 'High Risk' area, and above 0.80, it is a 'Critical Risk'."""
pdf.print_chapter(3, "3. STEP-BY-STEP WORKING MECHANISM\n------------------------------------------------------------", section3_body)


section4_body = """### Tabular Models (Decision Trees / Boosting / Linear / Deep Learning)

- Logistic Regression: A simple linear model that draws a straight mathematical line between fire and non-fire conditions. It is configured with a balanced class weight, a maximum of 1000 iterations (max_iter=1000), and relies on standard scaling.
- Random Forest: An ensemble of many individual decision trees voting together. Its specific parameters in the code are: 200 trees (n_estimators=200), a maximum tree depth of 20 (max_depth=20), a minimum of 10 samples to split a node (min_samples_split=10), a minimum of 10 samples at a leaf (min_samples_leaf=10), and looking at a square root subset of features at each split (max_features="sqrt").
- LightGBM: A very fast, highly efficient gradient boosting model. It is configured to build 300 sequential trees (n_estimators=300) and uses class_weight="balanced".
- XGBoost: A powerful, highly accurate gradient boosting model. It is configured with 300 trees (n_estimators=300), an evaluation metric of "logloss", and gives a doubled weight to fire events (scale_pos_weight=2.0) to ensure the model doesn't miss them.
- CatBoost: A gradient boosting model that is particularly good out-of-the-box. It is configured for 300 iterations (iterations=300) and automatically balances class weights (auto_class_weights="Balanced").

### Model Aggregation / Ensembling

To get a much more reliable final answer, the system combines the guesses of the 5 base models ("experts") instead of trusting just one.

- Voting Ensemble (The "Equal Say" Method): A simple aggregation method that takes a "soft" vote by calculating the exact mathematical average of the probability scores given by all five base models. If one model makes an outlier mistake, the other four will average it out and keep the system grounded.
- Stacking Ensemble (The "Smart Manager" Method): A smart aggregation method that trains a new meta-learner (a Manager) on top of the base models. Instead of taking a simple average, it hands the 5 predictions to a Logistic Regression model (with max_iter=1000). The Manager learns which experts to trust the most based on their past track record (e.g., trusting one model more in high wind conditions). It intelligently weighs their predictions to make the final, highly calibrated decision. It evaluates itself internally using a 5-fold Stratified K-Fold cross-validation (cv=5) on the training set."""
pdf.print_chapter(4, "4. SPECIFIC MODELS & PARAMETERS EXPLAINED\n------------------------------------------------------------", section4_body)

section5_body = """1. Data Ingestion: Raw latitude and longitude coordinates for the Indian region are collected.
2. Preprocessing & Feature Engineering: The system looks up the weather (temperature, humidity, wind, VPD) and vegetation greenness (EVI) for those coordinates from the previous day. Missing values are filled with the median. New features like the cyclical month and the VPD-Wind multiplier are calculated.
3. Model Evaluation & Score Generation: These finalized environmental numbers are fed simultaneously into the five fully trained base models (Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost). Each model individually calculates a probability score that a fire will occur.
4. The Final Decision / Ensemble Integration: Instead of trusting just one model, the Stacking Ensemble takes all five probability scores and feeds them into its final Logistic Regression meta-learner. This meta-learner weighs the predictions based on historical accuracy and generates one unified, definitive score.
5. Output: The system outputs a final prediction number between 0.0 and 1.0. If the score is greater than 0.50, the area is flagged for High Risk. If it exceeds 0.80, it is escalated to Critical Risk, alerting forest management to take immediate preventative action."""
pdf.print_chapter(5, "5. HOW THE PREDICTION WORKS (END-TO-END)\n------------------------------------------------------------", section5_body)

pdf.output("WILDFIRE_PREDICTION_REPORT.pdf")
print("Successfully generated WILDFIRE_PREDICTION_REPORT.pdf")
