import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -----------------------
# Athena Query Function
# -----------------------
def run_athena_query(query: str, database: str, s3_output: str) -> pd.DataFrame:
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
    # Pickup Location Input
    location_id = st.number_input("Enter Pickup Location ID", min_value=1, max_value=300, value=43)

    # Use Eastern Time (New York)
    eastern = ZoneInfo("America/New_York")
    now_ny = datetime.now(tz=eastern)

    # Calculate the same week last year in NY time
    end_date = now_ny - timedelta(days=358)
    start_date = now_ny - timedelta(days=365)

    # Round down to start of the hour
    start_rounded = start_date.replace(minute=0, second=0, microsecond=0)

    # Round up to next full hour if needed
    if end_date.minute > 0 or end_date.second > 0 or end_date.microsecond > 0:
        end_rounded = (end_date + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        end_rounded = end_date.replace(minute=0, second=0, microsecond=0)

    # Format for Athena query (still using naive-looking string)
    start_str = start_rounded.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_rounded.strftime("%Y-%m-%d %H:%M:%S")


    # Athena setup
    s3_output = 's3://jkim27-etl-5b9d2da3-5f5d-4ab5-bda1-80307b8dc702/athena/'

    # Queries
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

    # Run Queries
    try:
        actual_df = run_athena_query(actual_query, 'etl_taxi_transformed', s3_output)
        predicted_df = run_athena_query(predicted_query, 'taxi_predictions', s3_output)

        # Plot
        fig = px.line()
        fig.add_scatter(x=actual_df['pickup_hour'], y=actual_df['rides'], name='Actual Rides')
        fig.add_scatter(x=predicted_df['prediction_datetime'], y=predicted_df['predicted_rides'], name='Predicted Rides')
        fig.update_layout(title=f"Taxi Rides Forecast for Location {location_id}",
                          xaxis_title="Time",
                          yaxis_title="Number of Rides")

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")

# -----------------------
# Tab: RDS (Empty Placeholder)
# -----------------------
with tab2:
    st.markdown("🚧 RDS integration coming soon...")