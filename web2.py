import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_absolute_error
import requests, io, zipfile
###############################
# 1. Data Loading Functions
###############################

@st.cache_data
def load_xgboost_data():
    # URL for the zip file on GitHub (raw link)
    zip_url = "https://github.com/Meaad1704/Rossman-Sales-Forecasting/raw/main/archive.zip"
    
    # Download the zip file
    response = requests.get(zip_url)
    if response.status_code != 200:
        st.error("Failed to download the data.")
        return None
    
    # Read the zip file from the in-memory bytes and extract 'rossman.csv'
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('rossman.csv') as f:
            df = pd.read_csv(f, parse_dates=['Date'], index_col='Date')
    
    # Sort the DataFrame by the index (Date) and drop the 'Unnamed: 0' column if present
    df = df.sort_index()
    if 'Unnamed: 0' in df.columns:
        df.drop(columns='Unnamed: 0', inplace=True)
    
    return df

# Example usage in a Streamlit app:
df_xgb = load_xgboost_data()
if df_xgb is not None:
    st.write("### XGBoost Data Preview")
    st.write(df_xgb.head())
else:
    st.error("Data could not be loaded.")



@st.cache_data
def load_prophet_data():
    # URL to the zip file on GitHub (raw link)
    zip_url = "https://github.com/Meaad1704/Rossman-Sales-Forecasting/raw/main/archive.zip"
    
    # Download the zip file
    response = requests.get(zip_url)
    if response.status_code != 200:
        st.error("Failed to download the data.")
        return None

    # Read the zip file from the in-memory bytes buffer and extract "rossman.csv"
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open('rossman.csv') as f:
            data = pd.read_csv(f, parse_dates=['Date'])
    
    # Prepare the data for Prophet forecasting
    data = data.rename(columns={'Date': 'ds', 'Sales': 'y'})
    # Keep only the needed columns; adjust if necessary
    data = data[['ds', 'y', 'Customers', 'Promo', 'Open', 'StateHoliday', 'Store', 'DayOfWeek']]
    
    return data

# Example usage in a Streamlit app:
data = load_prophet_data()
if data is not None:
    st.write("### Data Preview")
    st.write(data.head())
else:
    st.error("Data could not be loaded.")

###############################
# 2. Streamlit App Layout
###############################

#############################
# Navigation: Choose Page
#############################
st.set_page_config(layout="wide", page_title="Rossmann Sales App")
st.title("Rossmann Store Sales App")

# Use a radio button for navigation instead of st.tabs for more control
page = st.radio("Select Page", ("XGBoost Prediction", "Prophet Prediction", "Dashboard"))

#############################
# Page: XGBoost Prediction
#############################
if page == "XGBoost Prediction":
    # Sidebar: only inputs are here
    with st.sidebar:
        st.header("XGBoost Input Features")
        StoreType = st.selectbox("Store Type", ['a', 'b', 'c', 'd'])
        Assortment = st.selectbox("Assortment", ['a', 'b', 'c'])
        PromoInterval = st.selectbox("Promo Interval", ["0", "Feb,May,Aug,Nov", "Jan,Apr,Jul,Oct", "Mar,Jun,Sept,Dec"])
        CompetitionDistance = st.number_input("Competition Distance", min_value=0.0)
        Promo2 = st.selectbox("Promo2", [0, 1])
        Open = st.selectbox("Open", [0, 1])
        SchoolHoliday = st.selectbox("School Holiday", [0, 1])
        DayOfWeek = st.selectbox("Day of Week", [1, 2, 3, 4, 5, 6, 7])
        Customers = st.number_input("Customers", min_value=0.0)
        Promo = st.selectbox("Promo", [0, 1])
        StateHoliday = st.selectbox("State Holiday", [0, 1])
        CompetitionOpenSinceMonth = st.selectbox("Competition Open Since Month", list(range(1, 13)))
        Promo2SinceWeek = st.number_input("Promo2 Since Week", min_value=0.0)
        Promo2SinceYear = st.number_input("Promo2 Since Year", min_value=2000.0)
        CompetitionOpenSinceYear = st.number_input("Competition Open Since Year", min_value=1900.0)
        Year = st.number_input("Year", min_value=2000.0)
        Month = st.selectbox("Month", list(range(1, 13)))
        WeekOfYear = st.number_input("Week of Year", min_value=1.0, max_value=53.0)
        DayOfYear = st.number_input("Day of Year", min_value=1.0, max_value=366.0)

    # Main area: header and prediction button/output    
    st.header("Sales Prediction App")
    
    # One-hot encoding for categorical features
    store_type_dummies = {f"StoreType_{c}": int(StoreType == c) for c in ['a', 'b', 'c', 'd']}
    assortment_dummies = {f"Assortment_{c}": int(Assortment == c) for c in ['a', 'b', 'c']}
    promo_interval_dummies = {
        "PromoInterval_0": int(PromoInterval == "0"),
        "PromoInterval_Feb,May,Aug,Nov": int(PromoInterval == "Feb,May,Aug,Nov"),
        "PromoInterval_Jan,Apr,Jul,Oct": int(PromoInterval == "Jan,Apr,Jul,Oct"),
        "PromoInterval_Mar,Jun,Sept,Dec": int(PromoInterval == "Mar,Jun,Sept,Dec")
    }
    
    # Build the input data dictionary
    input_data = {
        'DayOfWeek': DayOfWeek,
        'Customers': Customers,
        'Open': Open,
        'Promo': Promo,
        'StateHoliday': StateHoliday,
        'SchoolHoliday': SchoolHoliday,
        'CompetitionDistance': CompetitionDistance,
        'CompetitionOpenSinceMonth': CompetitionOpenSinceMonth,
        'CompetitionOpenSinceYear': CompetitionOpenSinceYear,
        'Promo2': Promo2,
        'Promo2SinceWeek': Promo2SinceWeek,
        'Promo2SinceYear': Promo2SinceYear,
        'Year': Year,
        'Month': Month,
        'WeekOfYear': WeekOfYear,
        'DayOfYear': DayOfYear
    }
    input_data.update(store_type_dummies)
    input_data.update(assortment_dummies)
    input_data.update(promo_interval_dummies)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Load the XGBoost model
    try:
        model = pickle.load(open(r"D:\Courses\Data Science - Depi\Final Project\Project Code\Model Deployment\xgboost_model.pkl", "rb"))
    except Exception as e:
        st.error("Error loading model. Please check the model path.")
        st.stop()
    
    if st.button("Predict Sales"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Sales: {prediction:.2f}")

###############################
# 4. Tab 2: Prophet Prediction
###############################
elif page == "Prophet Prediction":

    # Clear the sidebar if not on XGBoost page
    st.sidebar.empty()
    
    st.header("Prophet Sales Forecasting")

    prophet_data = load_prophet_data()

    # Extract unique stores from the Prophet dataset
    store_options = sorted(prophet_data['Store'].unique())
    col1, col2 = st.columns(2)

    with col1:
        store_val = st.selectbox("Select Store Number:", options=store_options, index=0)
    with col2:
        Forecast_Days = st.number_input("Enter Number of Forecast Days:", min_value=1, value=90, step=1)

    if st.button("Generate Forecast"):
        # Filter data for the selected store
        filtered_data = prophet_data[prophet_data['Store'] == store_val].sort_values('ds')
        if filtered_data.empty:
            st.warning("No data available for the selected store.")
        else:
            # Create train-test split: use the latest 6 months as test data
            split_date_recent = filtered_data['ds'].max() - pd.DateOffset(months=6)
            train = filtered_data[filtered_data['ds'] < split_date_recent].copy()
            test = filtered_data[filtered_data['ds'] >= split_date_recent].copy()

            # Create holiday dataframe (example: using StateHoliday==1)
            holiday_df = filtered_data[filtered_data['StateHoliday'] == 1][['ds']].drop_duplicates().copy()
            holiday_df['holiday'] = 'state_holiday'

            #####################
            # Forecast Model
            #####################
            model_prophet = Prophet(daily_seasonality=True, holidays=holiday_df)
            model_prophet.fit(train)
            # Forecast periods include test period length for evaluation
            future = model_prophet.make_future_dataframe(periods=int(Forecast_Days + len(test)))
            forecast = model_prophet.predict(future)

            # Create Plotly figure for forecast (with test data overlay)
            forecast_fig = plot_plotly(model_prophet, forecast)
            forecast_fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(visible=True),
                    rangeslider=dict(visible=False),
                    type="date"
                )
            )
            # Add test data as red markers
            forecast_fig.add_trace(go.Scatter(
                x=test['ds'],
                y=test['y'],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Test Data'
            ))
            # Add vertical lines for train/test and future forecast split
            split_date_train = train['ds'].max().to_pydatetime()
            split_end_date = filtered_data['ds'].max().to_pydatetime()
            forecast_fig.add_vline(x=split_date_train, line=dict(color='red', dash='dash', width=2))
            forecast_fig.add_vline(x=split_end_date, line=dict(color='violet', dash='dash', width=2))
            max_y = filtered_data['y'].max()
            forecast_fig.add_annotation(x=split_date_train, y=max_y, text="Train/Test Split",
                                        showarrow=True, arrowhead=1, ax=0, ay=-40)
            forecast_fig.add_annotation(x=split_end_date, y=max_y, text="Future Predict Split",
                                        showarrow=True, arrowhead=1, ax=0, ay=-40)
            # Compute MAE on the test set (forecast segment corresponding to test length)
            Performance = pd.merge(
                test,
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-int(Forecast_Days + len(test)):],
                on='ds'
            )
            mae_forecast = mean_absolute_error(Performance['y'], Performance['yhat'])
            forecast_fig.update_layout(
                title={
                    'text': f"Forecast Model <br> MAE {mae_forecast:.2f}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Date",
                yaxis_title="Sales",
                width=800,
                height=600
            )

            #####################
            # Baseline Model (without holiday features, etc.)
            #####################
            model_baseline = Prophet()
            model_baseline.fit(train)
            future_baseline = model_baseline.make_future_dataframe(periods=len(test))
            forecast_baseline = model_baseline.predict(future_baseline)

            baseline_fig = plot_plotly(model_baseline, forecast_baseline)
            baseline_fig.add_vline(x=split_date_train, line=dict(color='red', dash='dash', width=2))
            baseline_fig.add_trace(go.Scatter(
                x=test['ds'],
                y=test['y'],
                mode='markers',
                marker=dict(color='red', size=8),
                name='Test Data'
            ))
            baseline_fig.add_annotation(x=split_date_train, y=max_y, text="Train/Test Split",
                                        showarrow=True, arrowhead=1, ax=0, ay=-40)
            performance_baseline = pd.merge(
                test,
                forecast_baseline[['ds', 'yhat']][-int(len(test)):],
                on='ds'
            )
            mae_baseline = mean_absolute_error(performance_baseline['y'], performance_baseline['yhat'])
            baseline_fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(visible=True),
                    rangeslider=dict(visible=False),
                    type="date"
                ),
                title={
                    'text': f"Baseline Model <br> MAE {mae_baseline:.2f}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Date",
                yaxis_title="Sales",
                width=800,
                height=600
            )

            # Display the plots side by side if space permits
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(forecast_fig, use_container_width=True)
            with col_b:
                st.plotly_chart(baseline_fig, use_container_width=True)

            # Prepare forecast data for download (last n forecast days only)
            forecast_data = forecast[['ds', 'yhat']].tail(int(Forecast_Days)).copy()
            forecast_data.rename(columns={'ds': 'Date', 'yhat': 'Predicted Future Sales'}, inplace=True)
            # Example: set Sundays to 0 if needed (optional)
            forecast_data['Day'] = forecast_data['Date'].dt.day_name()
            forecast_data.loc[forecast_data['Day'] == 'Sunday', 'Predicted Future Sales'] = 0

            # Convert forecast data to Excel binary data
            towrite = io.BytesIO()
            forecast_data.to_excel(towrite, index=False, sheet_name="Forecast")
            towrite.seek(0)
            st.download_button(
                label="Download Forecast Data as Excel",
                data=towrite,
                file_name="Predicted_Future_Sales.xlsx",
                mime="application/vnd.ms-excel"
            )

###############################
# 5. Tab 3: dash (Dashboard)
###############################
elif page == "Dashboard":
    st.sidebar.empty()
    st.header("Sales Dashboard")
    # Use the same raw dataframe as in XGBoost prediction, or reload if necessary
    df_raw = load_xgboost_data()

    # Verify required columns exist
    if all(col in df_raw.columns for col in ['Year', 'StoreType', 'Promo', 'Sales']):
        # Dashboard filters
        selected_years = st.multiselect(
            "Select Year:",
            options=sorted(df_raw['Year'].unique()),
            default=sorted(df_raw['Year'].unique())
        )
        selected_store_types = st.multiselect(
            "Select Store Type:",
            options=sorted(df_raw['StoreType'].dropna().unique()),
            default=sorted(df_raw['StoreType'].dropna().unique())
        )
        selected_promo = st.multiselect(
            "Select Promo:",
            options=sorted(df_raw['Promo'].dropna().unique()),
            default=sorted(df_raw['Promo'].dropna().unique())
        )

        # Apply filters on a copy of the dataframe
        filtered_df = df_raw[
            (df_raw['Year'].isin(selected_years)) &
            (df_raw['StoreType'].isin(selected_store_types)) &
            (df_raw['Promo'].isin(selected_promo))
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sales Trend Over Time")
            sales_line = px.line(
                filtered_df, x=filtered_df.index, y='Sales',
                title="Sales Trend Over Time",
                labels={'Sales': "Sales", 'Date': "Date"}
            )
            st.plotly_chart(sales_line, use_container_width=True)

            st.subheader("Sales By Store Type")
            store_type = filtered_df.groupby('StoreType')['Sales'].sum().reset_index()
            sales_by_store_type = px.bar(
                store_type, x='StoreType', y='Sales', color='StoreType',
                title="Sales per Store Type",
                labels={'StoreType': "Store Type", 'Sales': "Sales"}
            )
            st.plotly_chart(sales_by_store_type, use_container_width=True)

            st.subheader("Sales Per Month Per Store Type")
            sales_per_month = filtered_df.groupby(["Month", "StoreType"])["Sales"].sum().reset_index()
            sales_per_month_fig = px.line(
                sales_per_month, x='Month', y='Sales', color='StoreType', markers=True,
                title="Sales Per Month Per Store Type",
                labels={"StoreType": "Store Category"}
            )
            st.plotly_chart(sales_per_month_fig, use_container_width=True)

        with col2:
            st.subheader("Sales By Year")
            sales_per_year = filtered_df.groupby('Year')['Sales'].sum().reset_index()
            sales_by_year = px.line(
                sales_per_year, x='Year', y='Sales',
                title="Total Sales Over the Years",
                markers=True
            )
            st.plotly_chart(sales_by_year, use_container_width=True)

            st.subheader("Sales By Competition Distance")
            sales_per_comp = filtered_df.groupby('CompetitionDistance')['Sales'].sum().reset_index()
            sales_by_competition_distance = px.line(
                sales_per_comp, x='CompetitionDistance', y='Sales',
                title="Sales by Competition Distance",
                labels={'CompetitionDistance': "Competition Distance", 'Sales': "Sales"}
            )
            st.plotly_chart(sales_by_competition_distance, use_container_width=True)

            st.subheader("Sales Per Day Of Week Per Store Type")
            sales_per_day = filtered_df.groupby(["DayOfWeek", "StoreType"])["Sales"].sum().reset_index()
            sales_per_day_fig = px.bar(
                sales_per_day, x='DayOfWeek', y='Sales', color="StoreType",
                title="Sales Per Day Of Week Per Store Type",
                labels={"StoreType": "Store Category"}
            )
            st.plotly_chart(sales_per_day_fig, use_container_width=True)
    else:
        st.warning("Dataset is missing required columns like 'StoreType'. Please check the CSV file.")
