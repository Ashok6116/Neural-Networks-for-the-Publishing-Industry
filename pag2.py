import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
upld=pd.read_csv('D:/Book_cpny/mysql/demand.csv')
daily=pd.read_csv('D:/Book_cpny/mysql/demand_for.csv')


ss=upld['order_id'].count()
st.markdown("""
<style>
.color_font {
    font-weight:bold;
    color:red;
.fonts{
            font-weight:bold;
            font-size:40px !important;
            }
}
            
</style>
""", unsafe_allow_html=True)
st.set_page_config(layout="wide")
textp1='<p class="color_font">15395</p>'
textp2='<p class="color_font">13871</p>'
textp3='<p class="color_font">8092</p>'
textp4='<p class="color_font">7055</p>'
textp5='<p class="color_font">620</p>'
textp6='<p class="color_font">407</p>'
cont1=st.container()
head='<p class="fonts">45440</p>'
cond=cont1.container(border=True)
cond.markdown(f'Total Order:{head}', unsafe_allow_html=True)
cols1,cols2,cols3,cols4,cols5,cols6=cont1.columns(6,border=True)
cols1.text('Order Received')
cols1.write(f'Total ord_id :{textp1}',unsafe_allow_html=True)
cols2.text('Pending Delivery')
cols2.write(f'Total ord_id :{textp2}',unsafe_allow_html=True)
cols3.text('Delivery In Progress')
cols3.write(f'Total ord_id :{textp3}',unsafe_allow_html=True)
cols4.text('Delivered')
cols4.write(f'Total ord_id :{textp4}',unsafe_allow_html=True)
cols5.text('Cancelled')
cols5.write(f'Total ord_id :{textp5}',unsafe_allow_html=True)
cols6.text('Returned')
cols6.write(f'Total ord_id :{textp6}',unsafe_allow_html=True)

daily['only_date']=pd.to_datetime(daily['only_date'])
#MODEL_PATH = 'D:/Book_cpny/mysql/demand_ann_model.keras'
#mdls = load_model(MODEL_PATH)

# ----------------- CONFIG: adjust paths to where you saved files -----------------
PIPELINE_PKL = r"D:/Book_cpny/models/pipeline_v1.pkl"   # file you saved with joblib.dump(...)
# fallback options if pipeline doesn't contain model_path
H5_FALLBACK = r"D:/Book_cpny/models/sales_model.h5"
SAVEDMODEL_FALLBACK = r"D:/Book_cpny/models/sales_model_saved"  # folder if you saved TF SavedModel
# ------------------------------------------------------------------------------

@st.cache_resource
def load_pipeline(pipeline_pkl=PIPELINE_PKL):
    # 1) check file exists
    if not os.path.exists(pipeline_pkl):
        raise FileNotFoundError(
            f"Pipeline file not found at: {pipeline_pkl}\n"
            "Please run your training script and create pipeline_v1.pkl that includes scaler_X and scaler_y."
        )

    data = joblib.load(pipeline_pkl)
    # expected keys
    if 'scaler_X' not in data or 'scaler_y' not in data:
        raise RuntimeError(
            f"Loaded pipeline missing scalers. Keys found: {list(data.keys())}\n"
            "You need pipeline with keys: 'scaler_X', 'scaler_y', (optional) 'model_path', 'feature_cols', 'n_lags'."
        )

    scaler_X = data['scaler_X']
    scaler_y = data['scaler_y']
    feature_cols = data.get('feature_cols', None)
    n_lags = data.get('n_lags', None)

    # 2) ensure scalers are fitted
    try:
        check_is_fitted(scaler_X)
        check_is_fitted(scaler_y)
    except NotFittedError as e:
        raise RuntimeError("Loaded scaler is not fitted. Re-run training to fit scalers and save them.") from e

    # 3) load model (prefer model_path inside pipeline, else try fallback)
    model = None
    model_path = data.get('model_path', None)
    tried_paths = []
    if model_path:
        tried_paths.append(model_path)
        try:
            model = load_model(model_path)
        except Exception as e:
            # fallback to safe load
            try:
                model = load_model(model_path, compile=False)
                model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
            except Exception:
                print("Failed to load model from model_path:", model_path, "error:", e)

    # fallback attempts if no model_path or it failed
    if model is None:
        # try H5 fallback
        if os.path.exists(H5_FALLBACK):
            tried_paths.append(H5_FALLBACK)
            try:
                model = load_model(H5_FALLBACK)
            except Exception:
                model = load_model(H5_FALLBACK, compile=False)
                model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
        # try SavedModel folder
        if model is None and os.path.exists(SAVEDMODEL_FALLBACK):
            tried_paths.append(SAVEDMODEL_FALLBACK)
            try:
                model = load_model(SAVEDMODEL_FALLBACK)
            except Exception as e:
                print("Failed to load SavedModel fallback:", e)

    # final check
    if model is None:
        st.warning(f"No model could be loaded. Tried paths: {tried_paths}. Pipeline still provides scalers.")
        # It's OK to continue if you only need scalers (but predictions require model)
    else:
        print("Model loaded from:", model_path or H5_FALLBACK or SAVEDMODEL_FALLBACK)

    return scaler_X, scaler_y, model, feature_cols, n_lags

# call loader and bind names used later in the script
scaler_X, scaler_y, model, feature_cols, n_lags = load_pipeline()
st.write("Pipeline loaded. model loaded?:", model is not None)




n_lags = 7
for i in range(1, n_lags + 1):
    daily[f'sales_lag_{i}'] = daily['sales'].shift(i)

# ----- 4) Rolling features -----
daily['rolling_3']  = daily['sales'].rolling(3).mean()
daily['rolling_7']  = daily['sales'].rolling(7).mean()
daily['rolling_14'] = daily['sales'].rolling(14).mean()

# ----- 5) Drop rows with NaN (first rows without full lags/rollings) -----
daily = daily.dropna().reset_index(drop=True)

# quick sanity check
print("Prepared rows:", len(daily))
print(daily.head(2).T)

# ----- 6) Build feature matrix X and target y -----
feature_cols = [
    'dayofweek','month','day','is_weekend',
    'rolling_3','rolling_7','rolling_14'
] + [f'sales_lag_{i}' for i in range(1, n_lags + 1)]



last_row = daily.iloc[-1].copy()

# Build initial lag list using the last available lags (in order lag1..lag7)
future_lags = [last_row[f'sales_lag_{i}'] for i in range(1, n_lags + 1)]
# For date features, we'll increment from last date
last_date = last_row['only_date']

next_7_days_raw = []
next_7_days_dates = []
future_feature_window = future_lags.copy()

for i in range(1, 8):
    # create feature vector: date features + rolling placeholders + lags
    future_date = last_date + pd.Timedelta(days=i)
    dayofweek = future_date.dayofweek
    month = future_date.month
    day = future_date.day
    is_weekend = int(dayofweek >= 5)

    # rolling features: approximate by using mean of last known sales and forecasts so far
    # compute a simple rolling_3 and rolling_7 from available values (combine actual history + preds)
    # construct a list of recent sales including preds: most recent actuals = future_lags (lag1..lag7)
    recent_sales = future_feature_window.copy()  # lag1 is most recent day (yesterday)
    # When predicting day i, 'recent_sales' contains the latest 7 values (some preds will be inserted below)
    rolling_3 = np.mean(recent_sales[:3])
    rolling_7 = np.mean(recent_sales[:7])
    rolling_14 = rolling_7  # fallback when not enough history

    feature_vector = [
        dayofweek, month, day, is_weekend,
        rolling_3, rolling_7, rolling_14
    ] + recent_sales

    



# Example: Predict next day manually

    # create your input feature vector here...
    x_input = np.array(feature_vector).reshape(1, -1)

    x_scaled = scaler_X.transform(x_input)
    pred_scaled = model.predict(x_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).ravel()[0]

    #st.write("Predicted sales:", round(pred))

    next_7_days_raw.append(pred)
    next_7_days_dates.append(future_date)

    # update the "future_feature_window": new predicted becomes lag1 for next iteration
    future_feature_window = [pred] + future_feature_window[:-1]

if st.button("Predict Next Day"):  
    for dt, val in zip(next_7_days_dates, next_7_days_raw):
        st.write(f"{dt.date()} -> {val:.2f}  (rounded: {int(round(max(0, val)))})")

st.subheader('Churn Prediction')
cont2=st.container()
colss1,colss2=cont2.columns(2,border=True)
upld['only_date']=pd.to_datetime(upld['only_date'])
with colss1:
    #st.write(upld)
    last_order=upld.groupby('customer_id')['only_date'].max()
    new_frame=pd.DataFrame(last_order)
    st.write(new_frame)
with colss2:
    st.subheader('Less than 3 month it give churn')
    st.write('so,churn=1,no-churn=0')
    cutoff = upld['only_date'].max() - pd.Timedelta(days=90)
    customer_churn = (last_order < cutoff).astype(int)
    new_frame['churn']=customer_churn
    st.write(new_frame)