# 📊 Machine Learning Project - Student Performance Prediction

## 📖 Overview
This project predicts students' math scores based on various features like gender, parental education, lunch type, and test preparation. It includes **data ingestion, transformation, model training, and prediction using Flask**.

---

## 🚀 Project Structure
MLProject/
│── artifacts/                                        # Stores trained model and preprocessor
│── notebook/                                         # Jupyter notebooks for EDA & training
│── src/                                              # Source code
│   ├── components/     ── data_ingestion.py          # Loads dataset & splits it
                        ── data_transformation.py     # Preprocesses data
                        ── model_trainer.py           # Trains models        
│   ├── pipeline/                                     # Prediction pipelines
│   ├── exception.py                                  # Custom exception handling
│   ├── logger.py                                     # Logging module
│   ├── utils.py                                      # Utility functions
│── templates/                                        # HTML templates for Flask app
│── static/                                           # CSS, JS files (if any)
│── app.py                                            # Flask API for predictions
│── requirements.txt                                  # Required Python packages
│── README.md        

🔧 Installation

1️⃣ Clone the Repository

git clone https://github.com/animesh33-ctrl/MLProject.git
cd MLProject

2️⃣ Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Train the Model

python data_ingestion.py

This will:

Load the dataset

Preprocess the data

Train and save the best ML model

🚀 Running the Flask API

1️⃣ Start the Flask App

python app.py

The app will run on http://127.0.0.1:8080/ (if port is set to 8080).

2️⃣ Make Predictions

Open the browser and go to http://127.0.0.1:8080/predictdata.

Enter student details (gender, parental education, etc.).

Click Predict to see the predicted math score.

🛠 Troubleshooting

🔴 Internal Server Error (500)

Run Flask in debug mode:

app.run(debug=True)

Check the logs for missing files or incorrect inputs.

**🔴 **FileNotFoundError: No such file or directory 'artifacts/model.pkl'

Retrain the model:

python data_ingestion.py

**🔴 **KeyError: 'Unknown categories [None] in column X'

Modify OneHotEncoder in data_transformation.py:

OneHotEncoder(handle_unknown="ignore")

Retrain the model.

📌 Future Improvements

✅ Improve model accuracy with hyperparameter tuning✅ Add a frontend for better user interaction✅ Deploy the model using Docker

👨‍💻 Author

Animesh33-ctrl
GitHub: @animesh33-ctrl

🚀 If you found this helpful, give a ⭐ on GitHub!

