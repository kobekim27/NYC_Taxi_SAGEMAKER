import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np

# -----------------------
# Athena Query Function
# -----------------------
@st.cache_data
def run_athena_query(hour, query: str, database: str, s3_output: str) -> pd.DataFrame:
    athena_client = boto3.client('athena', region_name="us-east-1")

    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': s3_output}
    )

    query_execution_id = response['QueryExecutionId']
    state = 'RUNNING'

    while state in ['RUNNING', 'QUEUED']:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        state = response['QueryExecution']['Status']['State']
        if state in ['RUNNING', 'QUEUED']:
            time.sleep(1)

    if state != 'SUCCEEDED':
        reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
        raise Exception(f"Athena query failed: {state} - {reason}")

    results = []
    columns = []
    next_token = None
    first_page = True

    while True:
        if next_token:
            result_set = athena_client.get_query_results(
                QueryExecutionId=query_execution_id,
                NextToken=next_token
            )
        else:
            result_set = athena_client.get_query_results(QueryExecutionId=query_execution_id)

        if first_page:
            columns = [col['Label'] for col in result_set['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            first_page = False

        rows = result_set['ResultSet']['Rows']
        if next_token is None:
            rows = rows[1:]  # skip header

        for row in rows:
            results.append([field.get('VarCharValue', '') for field in row['Data']])

        next_token = result_set.get('NextToken')
        if not next_token:
            break

    df = pd.DataFrame(results, columns=columns)

    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

    return df

# -----------------------
# Streamlit App
# -----------------------
st.title("NYC Taxi Rides Forecast")

# Tabs
tab1, tab2 = st.tabs(["Athena", "RDS"])

# -----------------------
# Tab: Athena
# -----------------------
with tab1:
    st.subheader("Athena: Actual vs Predicted Rides")

    location_map = {
        43: "Times Square",
        161: "JFK Airport",
        236: "Brooklyn Heights",
        138: "Union Square",
        140: "Harlem"
    }

    selected_names = st.multiselect(
        "Select up to 3 pickup locations",
        options=list(location_map.values()),
        default=["Times Square"],
        max_selections=3
    )

    eastern = ZoneInfo("America/New_York")
    now_ny = datetime.now(tz=eastern)
    end_date = now_ny - timedelta(days=358)
    start_date = now_ny - timedelta(days=365)

    start_rounded = start_date.replace(minute=0, second=0, microsecond=0)
    end_rounded = (end_date + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0) if end_date.minute > 0 or end_date.second > 0 or end_date.microsecond > 0 else end_date.replace(minute=0, second=0, microsecond=0)

    start_str = start_rounded.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_rounded.strftime("%Y-%m-%d %H:%M:%S")

    s3_output = 's3://jkim27-etl-5b9d2da3-5f5d-4ab5-bda1-80307b8dc702/athena/'

    for name in selected_names:
        location_id = [k for k, v in location_map.items() if v == name][0]

        actual_query = f"""
        SELECT DISTINCT
            pickup_hour,
            rides
        FROM glue_transformed
        WHERE
            pickup_location_id = {location_id}
            AND pickup_hour BETWEEN '{start_str}' AND '{end_str}'
        ORDER BY pickup_hour;
        """

        predicted_query = f"""
        SELECT DISTINCT
            prediction_datetime,
            predicted_rides
        FROM test_predicted_values
        WHERE
            pickup_location_id = '{location_id}'
            AND prediction_datetime BETWEEN '{start_str}' AND '{end_str}'
        ORDER BY prediction_datetime;
        """

        try:
            actual_df = run_athena_query(now_ny.hour, actual_query, 'etl_taxi_transformed', s3_output)
            predicted_df = run_athena_query(now_ny.hour, predicted_query, 'taxi_predictions', s3_output)

            fig = px.line()
            fig.add_scatter(x=actual_df['pickup_hour'], y=actual_df['rides'], name='Actual Rides')
            fig.add_scatter(x=predicted_df['prediction_datetime'], y=predicted_df['predicted_rides'], name='Predicted Rides')
            fig.update_layout(title=f"Taxi Rides Forecast for Location {location_id}",
                              xaxis_title="Time",
                              yaxis_title="Number of Rides")

            st.plotly_chart(fig)

            merged_df = pd.merge(
                actual_df,
                predicted_df,
                left_on='pickup_hour',
                right_on='prediction_datetime',
                how='inner'
            )

            if not merged_df.empty:
                epsilon = 1e-8
                y_true = merged_df['rides'].to_numpy()
                y_pred = merged_df['predicted_rides'].to_numpy()

                mae = np.mean(np.abs(y_true - y_pred))
                mask = y_true != 0
                if mask.sum() > 0:
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                    st.metric(" MAE", f"{mae:.2f}")
                    st.metric(" MAPE", f"{mape:.2f}%")
            else:
                st.warning("Error.")

        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")

# -----------------------
# Tab: RDS
# -----------------------
with tab2:
    st.subheader("RDS: Actual vs Predicted Rides")

    location_map = {
        43: "Times Square",
        161: "JFK Airport",
        236: "Brooklyn Heights",
        138: "Union Square",
        140: "Harlem"
    }

    selected_names_rds = st.multiselect(
        "Select up to 3 pickup locations (RDS)",
        options=list(location_map.values()),
        default=["Times Square"],
        max_selections=3,
        key="rds_location_multiselect"
    )

    import sqlalchemy
    from sqlalchemy import create_engine, text

    rds_host = "taxi-db.cyhsik28ubia.us-east-1.rds.amazonaws.com"
    rds_db = "taxidata"
    rds_user = "taxiuser"
    rds_password = "Occupier-Dismount-Unmovable-Fading-Defender"
    rds_port = 5432

    rds_engine = create_engine(
        f"postgresql+psycopg2://{rds_user}:{rds_password}@{rds_host}:{rds_port}/{rds_db}"
    )

    eastern = ZoneInfo("America/New_York")
    now_ny = datetime.now(tz=eastern)
    end_date = now_ny - timedelta(days=358)
    start_date = now_ny - timedelta(days=365)

    start_rounded = start_date.replace(minute=0, second=0, microsecond=0)
    end_rounded = (end_date + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0) if end_date.minute > 0 or end_date.second > 0 or end_date.microsecond > 0 else end_date.replace(minute=0, second=0, microsecond=0)

    start_str = start_rounded.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_rounded.strftime("%Y-%m-%d %H:%M:%S")

    for name in selected_names_rds:
        location_id_rds = [k for k, v in location_map.items() if v == name][0]

        try:
            with rds_engine.connect() as conn:
                actual_sql = text(f"""
                    SELECT pickup_hour, rides
                    FROM taxi_rides
                    WHERE pickup_location_id = :loc
                    AND pickup_hour BETWEEN :start AND :end
                    ORDER BY pickup_hour;
                """)
                predicted_sql = text(f"""
                    SELECT prediction_datetime, predicted_rides
                    FROM predicted_rides
                    WHERE pickup_location_id = :loc
                    AND prediction_datetime BETWEEN :start AND :end
                    ORDER BY prediction_datetime;
                """)

                actual_df_rds = pd.read_sql(actual_sql, conn, params={
                    'loc': location_id_rds,
                    'start': start_str,
                    'end': end_str
                })

                predicted_df_rds = pd.read_sql(predicted_sql, conn, params={
                    'loc': location_id_rds,
                    'start': start_str,
                    'end': end_str
                })

                fig_rds = px.line()
                fig_rds.add_scatter(x=actual_df_rds['pickup_hour'], y=actual_df_rds['rides'], name='Actual Rides')
                fig_rds.add_scatter(x=predicted_df_rds['prediction_datetime'], y=predicted_df_rds['predicted_rides'], name='Predicted Rides')
                fig_rds.update_layout(title=f"RDS - Taxi Rides Forecast for Location {location_id_rds}",
                                      xaxis_title="Time",
                                      yaxis_title="Number of Rides")
                st.plotly_chart(fig_rds)

                actual_df_rds['pickup_hour'] = pd.to_datetime(actual_df_rds['pickup_hour']).dt.floor('h')
                predicted_df_rds['prediction_datetime'] = pd.to_datetime(predicted_df_rds['prediction_datetime']).dt.floor('h')

                merged_df = pd.merge(
                    actual_df_rds,
                    predicted_df_rds,
                    left_on='pickup_hour',
                    right_on='prediction_datetime',
                    how='inner'
                )

                if not merged_df.empty:
                    epsilon = 1e-8
                    y_true = merged_df['rides'].to_numpy()
                    y_pred = merged_df['predicted_rides'].to_numpy()

                    mae = np.mean(np.abs(y_true - y_pred))
                    mask = y_true != 0
                    if mask.sum() > 0:
                        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                        st.metric(" MAE", f"{mae:.2f}")
                        st.metric(" MAPE", f"{mape:.2f}%")


        except Exception as e:
            st.error(f"❌ Error connecting to RDS: {e}")