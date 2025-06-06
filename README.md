# 💻 Laptop Price Prediction using Machine Learning

  This project is a machine learning-based web application built with **Python**, **Scikit-learn**, and **Streamlit** that predicts the price of a laptop based on various specifications such as CPU, GPU, RAM, storage, and more.

---

## 🚀 Features

- Predicts laptop prices in Euros (€) based on user-input specs.
- Trained using a cleaned and feature-engineered dataset.
- Streamlit UI for real-time, interactive prediction.
- Supports various storage types (HDD, SSD, Flash, Hybrid).
- Model built using regression with performance tuning.

---

## 📊 Input Parameters

- Company
- Laptop Type
- CPU Brand
- GPU Brand
- Operating System
- RAM (GB)
- Weight (kg)
- Touchscreen & IPS Display
- Storage (HDD, SSD, Flash, Hybrid)

---

## 🧰 Technologies Used

- **Python**
- **Pandas, NumPy, Scikit-learn** – Data processing and machine learning
- **Jupyter Notebook** – Model development and testing
- **Streamlit** – Web app for predictions
- **Pickle** – Model serialization

---

## 📁 Files Included

- `app.py` – Streamlit web app
- `laptop_price_prediction.py` – Model training script
- `model_test.py` – Script to test model independently
- `laptop_price_dataset.csv` – Cleaned dataset used for training
- `README.md` – Project documentation
- `requirements.txt` – Requirements for run this system

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/laptop-price-prediction.git
   cd laptop-price-prediction

2. **Create virtual environment (optional)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Run the app**:
   ```bash
   streamlit run app.py

---

## 📂 Dataset

  Ensure `laptop_price_dataset.csv` is present in the project folder. This dataset should contain the cleaned data with columns like `Company`, `TypeName`, `CPU`, `GPU`, `Memory`, `Weight`, `Price`, etc.

---

## 🤝 Contribution

  Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

  This project is licensed under the **MIT License** – see the LICENSE file for details.
