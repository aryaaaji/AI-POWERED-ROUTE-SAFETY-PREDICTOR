 🚦 AI-Powered Route Safety Predictor

A smart Streamlit web application that recommends the **safest driving route** between two places using **real-time weather**, **historical accident data**, and **Machine Learning**.

> Repository: [aryaaaji/AI-POWERED-ROUTE-SAFETY-PREDICTOR](https://github.com/aryaaaji/AI-POWERED-ROUTE-SAFETY-PREDICTOR)

 💡 Problem Statement

Most navigation systems (like Google Maps) focus only on speed or distance — not on **route safety**.  
This project aims to help users choose **safer travel paths** using accident history and weather conditions.


 🧠 Features

- 📍 Take **source** and **destination** inputs from the user
- 🧭 Fetch multiple route alternatives using **GraphHopper API**
- 🌦 Get live weather updates from **OpenWeatherMap API**
- 🤖 Use a **Random Forest Classifier** to predict route risk levels
- 🗺 Display safest and alternative routes on an **interactive map** (Folium)


 ⚙️ Tech Stack

| Component         | Tools Used                              |
|-------------------|-----------------------------------------|
| Language          | Python                                  |
| ML Model          | Random Forest (scikit-learn)            |
| UI Framework      | Streamlit                               |
| Data Handling     | Pandas, NumPy                           |
| Visualization     | Seaborn, Matplotlib, Folium             |
| External APIs     | GraphHopper, OpenWeatherMap             |



 📁 Repository Contents

| File                         | Description                                          |
|------------------------------|------------------------------------------------------|
| `app.py`                     | Streamlit web app (frontend + backend)              |
| `train_model.py`             | Script to train the ML model using accident data    |
| `visualize_data.py`          | Script for EDA and plotting feature importance      |
| `safety_model.pkl`           | Trained Random Forest model file                    |
| `road_accident_cleaned.csv`  | Cleaned dataset used for training and prediction    |
| `README.md`                  | This documentation file                             |


📈 Model Performance

- ✅ **Accuracy**: 95.2%
- 🔍 **Precision / Recall**:
  - **Low Risk**: Precision 0.92, Recall 1.00
  - **High Risk**: Precision 1.00, Recall 0.90


 🚀 How to Run Locally
 
1️⃣ Clone the repository
git clone https://github.com/aryaaaji/AI-POWERED-ROUTE-SAFETY-PREDICTOR.git
cd AI-POWERED-ROUTE-SAFETY-PREDICTOR

2️⃣ Install requirements
Make sure Python 3.8+ is installed, then run:
pip install -r requirements.txt

3️⃣ Launch the app
streamlit run app.py


APIs Used

🌍 GraphHopper Routing API
🌦 OpenWeatherMap API

🔮 Future Scope

📡 Add real-time traffic or accident alert feeds
📱 Build a mobile-friendly version
☁️ Deploy to Streamlit Cloud or Hugging Face Spaces
🔬 Explore deep learning models like XGBoost or LSTM

👩‍💻 Author

Arya Aji

B.Tech, Electrical and Electronics Engineering

National Institute of Technology, Warangal

📧 aryaaji102500@gmail.com

📜 License

This project is licensed under the MIT License.
Feel free to fork, modify, and use it with attribution.
