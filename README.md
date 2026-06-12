# 🌲 Wildfire Prediction Platform (V2)

Welcome to **Wildfire V2**! This project uses machine learning to predict where wildfires are likely to happen. By looking at weather patterns and past fire data, we've built a system that gives us an early warning on potential fire risks.

## 🎯 What Does This Project Do?

We want to help people stay prepared. This platform:
- Takes daily weather information (like temperature, humidity, and wind).
- Uses smart machine learning models to see if these conditions might start a fire.
- Shows the predictions on an interactive dashboard so it's easy to understand.

## 📊 Where Do We Get Our Data?
To train our models, we used information from trusted sources:
- **NASA FIRMS**: This tells us exactly when and where past wildfires happened.
- **Open-Meteo**: This gives us the weather details (like how hot, dry, or windy it was) before and during the fires.

## 🧠 How It Works

We built a few different AI models to see which one is the smartest at finding fires. Here is our step-by-step process:

1. **Learning from the Past:** We feed the models lots of old data. They learn that hot, dry, and windy days usually mean a higher chance of fire.
2. **Testing the Models:** We check our models using a special method called "Spatial Block Cross-Validation." This simply means we test the models on different map areas to make sure they are really good at predicting fires everywhere, not just in one spot.
3. **The Best Models:** We use a few popular machine learning tools to make predictions:
   - Random Forest
   - XGBoost
   - CatBoost
   - LightGBM
   - Logistic Regression
   
   *Tip: We even created an "Ensemble Model" that combines the best parts of the top models to make the smartest predictions possible!*

## 🖥️ Interactive Dashboard

Reading numbers can be boring, so we built a visual dashboard! 
- Our `run_prediction.py` and `predict_tomorrow.py` scripts act as our prediction engine. They check the weather and generate a file full of predictions.
- The **React Dashboard** (inside the `wildfire-react-dashboard` folder) reads this file and plots the danger zones on a map for you to see.

## 🚀 Future Steps
We are always looking to make this better! In the future, we plan to:
- Add more details to our data, like the type of trees in an area (forest cover).
- Make our map even more accurate and fast.
- Create multi-level alerts (Low, Medium, High risk) instead of just Yes/No predictions.

---
*Inspired by the need to protect our environment and communities.*
