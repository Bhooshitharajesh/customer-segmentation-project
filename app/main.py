from fastapi import FastAPI
import pandas as pd
import joblib
from app.schema import InputData

app = FastAPI()

model = joblib.load("model/kmeans.pkl")
scaler = joblib.load("model/scaler.pkl")


def create_rfm(df):
    import datetime as dt

    df.rename(columns={"Customer ID": "Customer_ID"}, inplace=True)

    df = df.dropna(subset=["Customer_ID"])
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    df["Total price"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    today = df["InvoiceDate"].max()

    rfm = df.groupby("Customer_ID").agg({
        "InvoiceDate": lambda x: (today - x.max()).days,
        "Invoice": "count",
        "Total price": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    return rfm



@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input JSON → DataFrame
        df = pd.DataFrame([data.dict()])

        # Step 1: Create RFM
        rfm = create_rfm(df)

        # Step 2: Scale
        scaled = scaler.transform(rfm)

        # Step 3: Predict
        clusters = model.predict(scaled)

        # Step 4: Attach result
        rfm["cluster"] = clusters

        return {
            "status": "success",
            "result": rfm.reset_index().to_dict(orient="records")

        }

    except Exception as e:
        return {"error": str(e)}
    
    