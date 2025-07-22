import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle

# Load your model
with open("modelsales.pkl", "rb") as file:
    model = pickle.load(file)

# Set page config
st.set_page_config(page_title="Sales Prediction", layout="wide")

# -------------------- Custom CSS for Dark UI --------------------
st.markdown("""
    <style>
        .stApp {
            background-color: #000000;
            color: #EAECEE;
        }
        h1, h2, h3, h4 {
            color: #3498DB;
            font-weight: bold;
        }
        input[type="range"]::-webkit-slider-thumb {
            background: #000000;
        }
        input[type="range"]::-webkit-slider-runnable-track {
            background: #000000;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .css-1cpxqw2 {
            padding: 0rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- Title --------------------
st.markdown("<h1 style='text-align: center;'>📊 Sales Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#EAECEE;'>Adjust ad budget to forecast product sales</p>", unsafe_allow_html=True)

# -------------------- Compact Layout with 3 Columns --------------------
col1, col2, col3 = st.columns([1, 1.5, 1])

# ---------- LEFT COLUMN: BAR CHART ----------
with col1:
    st.markdown("### 📊 Ad Budget Split")


def budget_bar_chart(tv, radio, newspaper):
    fig = go.Figure(go.Bar(
        x=['TV', 'Radio', 'Newspaper'],
        y=[tv, radio, newspaper],
        marker_color=['#3498DB', '#3498DB', '#3498DB'],  # New colors
    ))
    fig.update_layout(
        title="Ad Budget Split ($)",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="white"),
        height=250,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig


# ---------- CENTER COLUMN: SLIDERS + PREDICTION ----------
with col2:
    st.markdown("### 🎚️ Adjust Budget ($)")
    tv = st.slider("📺 TV", 0.0, 300_000.0, 100_000.0, 1000.0)
    radio = st.slider("📻 Radio", 0.0, 60_000.0, 25_000.0, 1000.0)
    newspaper = st.slider("📰 Newspaper", 0.0, 120_000.0, 20_000.0, 1000.0)

    # Predict
    input_df = pd.DataFrame({
        'TV': [tv / 1000],
        'Radio': [radio / 1000],
        'Newspaper': [newspaper / 1000]
    })
    predicted_sales = model.predict(input_df)[0]

    # Show result
    st.markdown(f"""
        <div style='padding: 1rem; background-color: #1C2833; border-radius: 10px; text-align: center;'>
            <h3 style='color: #3498DB;'>📈 Predicted Sales: {predicted_sales:.2f} units</h3>
            <p style='color: #EAECEE;'>Total Ad Spend: <strong>${tv + radio + newspaper:,.0f}</strong></p>
        </div>
    """, unsafe_allow_html=True)

# ---------- RIGHT COLUMN: TREND CHART ----------
with col3:
    st.markdown("### 📈 Sales Trend")
    total_spend = [i * 10000 for i in range(1, 16)]
    trend_sales = [model.predict(pd.DataFrame({
        'TV': [i * 0.6 / 1000],
        'Radio': [i * 0.3 / 1000],
        'Newspaper': [i * 0.1 / 1000]
    }))[0] for i in total_spend]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=total_spend, y=trend_sales,
        mode='lines+markers',
        line=dict(color='blue')
    ))
    fig2.update_layout(
        xaxis_title="Total Budget ($)",
        yaxis_title="Predicted Sales",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="white"),
        height=250,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------- Render Bar Chart ----------
with col1:
    st.plotly_chart(budget_bar_chart(tv, radio, newspaper), use_container_width=True)

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 0.9rem; color: #566573;'>💡 Powered by Streamlit + Plotly | Clean Layout Mode</p>",
    unsafe_allow_html=True
)
