import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import streamlit_authenticator as stauth
import sqlite3
import os

# New path to the database
db_path = 'users.db'

# Create a new database and user table if it doesn't exist
def create_db():
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
                  CREATE TABLE IF NOT EXISTS users
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  phone TEXT,
                  username TEXT UNIQUE,
                  password TEXT,
                  coach TEXT)
                  ''')
        conn.commit()
        conn.close()

create_db()

# Load users from the database
def load_users():
    credentials = {"usernames": {}}
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT username, name, password FROM users")
    rows = c.fetchall()
    conn.close()

    print("Fetched rows from database:", rows)  # Debug statement

    for row in rows:
        credentials["usernames"][row[0]] = {"name": row[1], "password": row[2]}

    print("Updated credentials:", credentials)  # Debug statement
    return credentials

credentials = load_users()

print("Loaded users:", credentials)  # Debug statement

authenticator = stauth.Authenticate(
    credentials,
    'some_cookie_name',
    'some_signature_key',
    cookie_expiry_days=30
)

try:
    name, authentication_status, username = authenticator.login('main', 'Login')
except KeyError as e:
    st.error(f"Login error: {e}. The user does not exist in the database. Please register first.")
    authentication_status = None

if authentication_status:
    st.sidebar.success(f"Welcome {name}!")

    # Add pages
    page = st.sidebar.selectbox("Choose a page", ["Home"])

    if page == "Home":
        # Load Model
        model_path = 'Bitcoin_Model1.h5'
        if not os.path.exists(model_path):
            st.error(f'Model file {model_path} not found. Please upload the model file.')
        else:
            model = load_model(model_path)

            st.header('Bitcoin Price Prediction Model by Cryptex')

            st.subheader('Bitcoin Price Data')
            data = pd.DataFrame(yf.download('BTC-USD', '2015-01-01', '2024-07-07'))
            data = data.reset_index()
            st.write(data)

            st.subheader('Bitcoin Line Chart')
            close_data = data[['Date', 'Close']]
            data.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
            st.line_chart(close_data.set_index('Date'))

            train_data = close_data['Close'][:-100]
            test_data = close_data['Close'][-200:]

            scaler = MinMaxScaler(feature_range=(0, 1))
            train_data_scale = scaler.fit_transform(train_data.values.reshape(-1, 1))
            test_data_scale = scaler.transform(test_data.values.reshape(-1, 1))
            base_days = 100
            x = []
            y = []
            for i in range(base_days, test_data_scale.shape[0]):
                x.append(test_data_scale[i - base_days:i])
                y.append(test_data_scale[i, 0])

            x, y = np.array(x), np.array(y)
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))

            st.subheader('Predicted vs Original Prices')
            pred = model.predict(x)
            pred = scaler.inverse_transform(pred)
            preds = pred.reshape(-1, 1)
            ys = scaler.inverse_transform(y.reshape(-1, 1))
            preds = pd.DataFrame(preds, columns=['Predicted Price'])
            ys = pd.DataFrame(ys, columns=['Original Price'])
            chart_data = pd.concat((preds, ys), axis=1)
            st.write(chart_data)
            st.subheader('Predicted vs Original Prices Chart')
            st.line_chart(chart_data)

            # Button to load last 100 days data and predict
            if st.button('Load Last 100 Days and Predict'):
                last_100_days_data = yf.download('BTC-USD', period='6mo')['Close'][-100:]  # 6 months is more than 100 days
                if not last_100_days_data.empty and len(last_100_days_data) == base_days:
                    custom_data = last_100_days_data.to_list()

                    try:
                        custom_data = np.array(custom_data)
                        custom_data = custom_data.reshape(-1, 1)
                        custom_data_scaled = scaler.transform(custom_data)
                        custom_data_scaled = np.reshape(custom_data_scaled, (1, custom_data_scaled.shape[0], 1))

                        # Use existing method to predict the future
                        m = custom_data_scaled.flatten()
                        z = []
                        future_days = 5
                        for i in range(base_days, base_days + future_days):
                            inter = m[-base_days:].reshape(1, base_days, 1)
                            pred = model.predict(inter)
                            m = np.append(m, pred)
                            z = np.append(z, pred)

                        z = z.reshape(-1, 1)
                        z = scaler.inverse_transform(z)

                        st.subheader('Predicted Price for the Next 5 Days')
                        st.write(z)

                        st.subheader('Next 5 Days Prediction Chart')
                        future_dates = pd.date_range(start=close_data['Date'].iloc[-1] + pd.Timedelta(days=1),
                                                     periods=future_days)
                        future_df = pd.DataFrame(z, index=future_dates, columns=['Predicted Price'])
                        st.line_chart(future_df)

                    except ValueError as e:
                        st.error(f'Error processing the data for prediction: {e}')
                else:
                    st.error('Not enough data to make predictions.')

            m = y
            z = []
            future_days = 5
            for i in range(base_days, len(m) + future_days):
                m = m.reshape(-1, 1)
                inter = [m[-base_days:, 0]]
                inter = np.array(inter)
                inter = np.reshape(inter, (inter.shape[0], inter.shape[1], 1))
                pred = model.predict(inter)
                m = np.append(m, pred)
                z = np.append(z, pred)
            st.subheader('Predicted Future Days Bitcoin Price')
            z = np.array(z)
            z = scaler.inverse_transform(z.reshape(-1, 1))
            st.line_chart(z)

    # Admin access
    admin_password = "adminpass109"  # Administrator password

    if st.sidebar.checkbox('Admin'):
        admin_input = st.sidebar.text_input("Admin Password", type="password")
        if admin_input == admin_password:
            st.sidebar.success("Admin access granted")
            st.header("User Information")

            # Load user data
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT name, phone, username, coach FROM users")
            users_data = c.fetchall()
            conn.close()

            # Display user data
            df_users = pd.DataFrame(users_data, columns=["Name", "Phone", "Username", "Coach"])
            st.dataframe(df_users)
        else:
            st.sidebar.error("Incorrect admin password")

else:
    if authentication_status == False:
        st.error('Username/password is incorrect')

    elif authentication_status == None:
        st.warning('Please enter your username and password')

    st.subheader('Register')
    name = st.text_input('Name')
    phone = st.text_input('Phone (Optional)')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    coach = st.selectbox('Choose a coach', ['Алексей', 'Григорий', 'Дмитрий', 'Артем'])

    if st.button('Register'):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (name, phone, username, password, coach) VALUES (?, ?, ?, ?, ?)",
                      (name, phone, username, password, coach))
            conn.commit()
            st.success("User registered successfully!")
            credentials = load_users()  # Reload users after registration
        except sqlite3.IntegrityError:
            st.error("Username already exists")
        conn.close()
