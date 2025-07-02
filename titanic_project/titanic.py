import streamlit as st
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score# ---------- Page Setup ----------


# ---------- Load Pickled Model ----------
@st.cache_resource
def load_model():
    with open('titanic_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model']

model = load_model()

# ---------- Page Setup ----------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.markdown("""
    <style>
        .reportview-container {
            background: #f8f9fa;
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .prediction-box {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚢 Titanic Survival Predictor")
st.caption("Enter the passenger details in the sidebar to predict if they would survive.")

# ---------- Sidebar Inputs ----------
st.sidebar.header("🧍 Passenger Information")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ['male', 'female'])
age = st.sidebar.slider("Age", 0, 80, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.slider("Fare", 0.0, 500.0, 50.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# ---------- Manual Encoding ----------
sex_encoded = 1 if sex == 'male' else 0
embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
embarked_encoded = embarked_dict[embarked]

input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# ---------- Prediction ----------
if st.sidebar.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Output in styled box
    st.markdown(f"""
    <div class="prediction-box">
        <h3 style="color:{'green' if prediction == 1 else 'crimson'};">
            {'✅ Survived' if prediction == 1 else '❌ Did Not Survive'}
        </h3>
        <p><strong>Probability of Survival:</strong> {probability:.2%}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("⬅️ Enter passenger details and click **Predict Survival**.")

# ---------- Footer ----------
st.markdown("""
    <hr style="margin-top:3rem;">
    <center>
        <small>Made with ❤️ using Streamlit</small>
    </center>
""", unsafe_allow_html=True)
