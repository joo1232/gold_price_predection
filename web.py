import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import tempfile
import requests

# 🔗 Model URL (Raw .h5 file on GitHub)
MODEL_URL = "https://github.com/joo1232/gold_price_predection/blob/main/final_gold_model.h5"

# 🛠 Page settings
st.set_page_config(page_title="Gold Price Predictor", layout="centered")
st.title("🟡 Gold Price Predictor")
st.markdown("Upload your gold price CSV to predict closing prices for any date, even beyond your data.")

# 📁 Upload CSV
csv_file = st.file_uploader("📄 Upload CSV (must contain Date, Close)", type=["csv"])

if csv_file:
    try:
        # 📥 Download model from GitHub and write it to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            model_path = tmp.name

        # 🧠 Load the model
        model = load_model(model_path)

        # 📊 Load and preprocess CSV
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        if len(df) < 60:
            st.error("❌ Your CSV must contain at least 60 rows.")
        else:
            # 📅 User date picker
            min_date = df['Date'].iloc[59].date()
            max_date = df['Date'].max().date() + pd.Timedelta(days=30)
            selected_date = st.date_input(
                "📅 Select any date to predict the closing price:",
                value=df['Date'].max().date(),
                min_value=min_date,
                max_value=max_date
            )
            selected_datetime = pd.to_datetime(selected_date)

            # 🔍 Scale close prices
            scaler = MinMaxScaler()
            scaled_close = scaler.fit_transform(df[['Close']])
            last_date = df['Date'].max()
            last_scaled_seq = scaled_close[-60:].reshape(1, 60, 1)

            if selected_datetime <= last_date:
                # Predict inside known data range
                idx = df.index[df['Date'] >= selected_datetime][0]
                if idx < 60:
                    st.error("❌ Not enough data before selected date to make prediction (need 60 previous points).")
                else:
                    input_seq = scaled_close[idx-60:idx].reshape(1, 60, 1)
                    predicted_scaled = model.predict(input_seq)
                    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

                    actual_price = df.loc[idx, 'Close']
                    pct_change = ((predicted_price - actual_price) / actual_price) * 100

                    st.markdown(f"**Predicted next closing price after {selected_datetime.date()}**: ${predicted_price:.2f}")
                    st.markdown(f"**Actual closing price at {selected_datetime.date()}**: ${actual_price:.2f}")
                    st.markdown(f"**Percentage change (prediction vs actual)**: {pct_change:.2f}%")

            else:
                # Predict future prices step-by-step
                hours_to_predict = int((selected_datetime - last_date).total_seconds() / 3600)
                if hours_to_predict <= 0:
                    st.error("Selected date must be after last date in your data.")
                else:
                    seq = last_scaled_seq.copy()
                    preds_scaled = []

                    for _ in range(hours_to_predict):
                        pred_scaled = model.predict(seq)
                        preds_scaled.append(pred_scaled[0, 0])
                        pred_reshaped = pred_scaled.reshape(1, 1, 1)
                        seq = np.concatenate((seq[:, 1:, :], pred_reshaped), axis=1)

                    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
                    preds_real = scaler.inverse_transform(preds_scaled)

                    predicted_price = preds_real[-1, 0]
                    last_known_price = df['Close'].iloc[-1]
                    pct_change = ((predicted_price - last_known_price) / last_known_price) * 100

                    st.markdown(f"**Predicted closing price on {selected_datetime.date()}**: ${predicted_price:.2f}")
                    st.markdown(f"**Last known closing price on {last_date.date()}**: ${last_known_price:.2f}")
                    st.markdown(f"**Percentage change from last known price**: {pct_change:.2f}%")

                    # Show intermediate predictions
                    pred_dates = [last_date + pd.Timedelta(hours=i+1) for i in range(hours_to_predict)]
                    pred_df = pd.DataFrame({'Date': pred_dates, 'Predicted Close': preds_real.flatten()})
                    st.subheader("📊 Predicted Prices Leading Up To Selected Date")
                    st.dataframe(pred_df)

            # Show plot
            st.subheader("📈 Gold Price Trend")
            chart_df = df[['Date', 'Close']].copy()
            if selected_datetime > last_date:
                pred_plot_df = pred_df.rename(columns={'Predicted Close': 'Close'})
                chart_df = pd.concat([chart_df, pred_plot_df], ignore_index=True)

            chart_df = chart_df.set_index('Date').sort_index()
            st.line_chart(chart_df, height=300)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
