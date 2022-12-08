import pandas as pd
import joblib
import streamlit as st
import math
import plotly.express as px
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.title("🏠 House Price Prediction App Using Random Forest")
st.markdown("This app will help you to predict house price on 18 states in the U.S. based on realtor.com dataset")
image = Image.open('map.png')
st.image(image, caption='18 States in the U.S.')
st.write("---")

# load the model from disk
@st.cache(allow_output_mutation=True)
def load_model():
    filename = 'rf_model_comp.pkl'
    model = joblib.load(open(filename, 'rb'))
    return model

rf_model = load_model()

state = {'Connecticut': 0, 'Delaware': 1, 'Georgia': 2, 'Maine': 3, 'Massachusetts': 4, 'New Hampshire': 5,
         'New Jersey': 6, 'New York': 7, 'Pennsylvania': 8, 'Puerto Rico': 9, 'Rhode Island': 10,
         'South Carolina': 11, 'Tennessee': 12, 'Vermont': 13, 'US Virgin Islands': 14, 'Virginia': 15,
         'West Virginia': 16, 'Wyoming': 17}

# Massachusetts
st.markdown("""
**Example 1**: We have data like this ⬇️
\n🛏️Number of Bed = 2, \n🛁Number of Bath = 1, \n🌐Acre Lot = 10, 
\n🚩State = Massachusetts, \n📐House Size = 100.
We can get the house price = $47,500.00
""")

with st.expander("See the new dataset: "):
    bed = 2
    bath = 1
    acre_lot = math.log(10)
    states = 4
    house_size = math.log(100)

    new_data1 = [bed, bath, acre_lot, states, house_size]
    st.write("New data prediction: ", new_data1)
    pred_y_new_data = rf_model.predict([new_data1])
    st.success(f"The House Price Prediction is = $ {math.exp(pred_y_new_data):.2f}")

st.write("---")

# Pennsylvania
st.markdown("""
**Example 2**: We have data like this ⬇️
\n🛏️Number of Bed = 2, \n🛁Number of Bath = 1, \n🌐Acre Lot = 10, 
\n🚩State = Pennsylvania, \n📐House Size = 100
We can get the house price = $92,451.63
""")
with st.expander("See the new dataset: "):
    bed = 2
    bath = 1
    acre_lot = math.log(10)
    states = 8
    house_size = math.log(100)
    new_data2 = [bed, bath, acre_lot, states, house_size]
    st.write("New data prediction: ", new_data2)
    pred_y_new_data2 = rf_model.predict([new_data2])
    st.success(f"The House Price Prediction is= $ {math.exp(pred_y_new_data2):.2f}")

st.write("---")

with st.sidebar:
    st.header("📊 DS 501 - Final Project/💻Group10 ")
    st.subheader("🧮 Lets specify some parameters: ")
    number_of_bed = st.slider("Number of Bed: ", min_value=1, step=1, max_value=5, value=2)
    number_of_bath = st.slider("Number of Bath: ", min_value=1, step=1, max_value=5, value=1)
    number_of_acre_lot = st.number_input("Acre Lot: ", min_value=0.1, step=0.1, max_value=100.0, value=3.0, format="%f")
    us_states = st.selectbox("Select States: ",
                             ('Connecticut', 'Delaware', 'Georgia', 'Maine', 'Massachusetts', 'New Hampshire',
                              'New Jersey', 'New York', 'Pennsylvania', 'Puerto Rico', 'Rhode Island',
                              'South Carolina', 'Tennessee', 'Vermont', 'US Virgin Islands', 'Virginia',
                              'West Virginia', 'Wyoming'))
    number_of_house_size = st.number_input("House Size: ", min_value=100.0, step=0.5, max_value=5000.0, value=150.0, format="%f")

st.subheader("💻 Let's make new price prediction!")
st.write("🛏️Number of Bed ", number_of_bed)
st.write("🛁Number of Bath:  ", number_of_bath)
st.write("🌐Acre Lot: ", number_of_acre_lot)
st.write("🚩States: ", us_states)
st.write("📐House Size: ", number_of_house_size)

pred = [number_of_bed, number_of_bath, math.log(number_of_acre_lot), state[us_states], math.log(number_of_house_size)]

with st.expander("See the new dataset: "):
    st.write("New data prediction: ", pred)

with st.sidebar:
    predict_button = st.button('Predict!')
    if predict_button:
        new_data = pred
        pred_y_new_data = rf_model.predict([new_data])
        st.success(f"The House Price Prediction is = $ {math.exp(pred_y_new_data):.2f}")
        st.balloons()

st.write("---")

st.subheader("👨‍🔬House Price Comparison between States")
st.write("We set 🛏️ Number of Bed = 2, 🛁Number of Bath = 1, 🌐Acre Lot = 10, and 📐House Size = 100")
price_comparison = {'Connecticut': 47951.38, 'Delaware': 47951.38, 'Georgia': 47500.00, 'Maine': 47500.00,
                    'Massachusetts': 47500.00, 'New Hampshire': 47500.00, 'New Jersey': 47500.00, 'New York': 24319.21,
                    'Pennsylvania': 92451.63, 'Puerto Rico': 73509.03, 'Rodhe Island': 69314.51,
                    'South Carolina': 215057.88, 'Tennessee': 215057.88, 'Vermont': 238119.03,
                    'US Virgin Islands': 238119.03, 'Virginia': 238119.03, 'West Virginia': 238119.03,
                    'Wyoming': 238119.03}

price_data_df = pd.DataFrame(price_comparison.items(), columns=['States', 'House Price'])
fig = px.bar(price_data_df, x="States", y="House Price")
fig.update_layout(title_text='Price Comparison between Cab States', title_x=0.5)
st.plotly_chart(fig, use_container_width=True)

st.write("***")

def st_button(icon, url, label, icon_size):
    icon = 'linkedin'
    button_code = f'''
            <p>
                <a href={url} class="btn btn-outline-primary btn-lg btn-block" type="button" aria-pressed="true">
                    <svg xmlns="http://www.w3.org/2000/svg" width={icon_size} height={icon_size} fill="currentColor" class="bi bi-linkedin" viewBox="0 0 16 16">
                        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
                    </svg>
                    {label}
                </a>
            </p>'''
    return st.markdown(button_code, unsafe_allow_html=True)
icon_size=20
st_button('linkedin', 'www.linkedin.com/in/aviv-nur', 'Follow me on LinkedIn', icon_size)
