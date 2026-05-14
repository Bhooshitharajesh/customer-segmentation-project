import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import fpgrowth, association_rules

if "processed" not in st.session_state:
    st.session_state["processed"] = False

st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="centered")

@st.cache_resource
def load_models():
    kmeans_model = joblib.load("model/kmeans.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return kmeans_model, scaler

kmeans_model, scaler = load_models()
st.success("Model and scaler loaded successfully")


def create_rfm_single(invoice_date, quantity, price):
    today = pd.Timestamp.today().normalize()
    invoice_date = pd.to_datetime(invoice_date)

    recency = (today - invoice_date).days
    frequency = 1
    monetary = quantity * price

    rfm_df = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    }])

    return rfm_df

def create_rfm_bulk(df):
    df = df.copy()

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

def get_recommendation(cluster):
    if cluster == 0:
        return [
            "Discount offers",
            "Popular low-cost products",
            "Buy 1 Get 1 deals"
        ]
    elif cluster == 1:
        return [
            "Premium products",
            "New arrivals",
            "Exclusive membership offers"
        ]
    elif cluster == 2:
        return [
            "Win-back coupons",
            "Special discounts",
            "Limited-time offers"
        ]
    
@st.cache_data
def build_product_rules(df):
    # Keep only needed columns
    basket_df = df[["Invoice", "Description"]].dropna()

    # Remove duplicate product entries in same invoice
    basket_df = basket_df.drop_duplicates()

    # Create basket table
    basket = basket_df.assign(value=1).pivot_table(
        index="Invoice",
        columns="Description",
        values="value",
        fill_value=0
    )

    basket = basket.astype(bool)
    
    # FP-Growth instead of Apriori
    frequent_items = fpgrowth(basket, min_support=0.005, use_colnames=True)

    # Safety check (VERY IMPORTANT)
    if frequent_items.empty:
        return pd.DataFrame()

    rules = association_rules(frequent_items, metric="lift", min_threshold=1)

    if rules.empty:
        return pd.DataFrame()

    # Sort for better recommendations
    rules = rules.sort_values(by="lift", ascending=False)
  
    return rules
   

def recommend_products(product_name, rules):
    recommendations = []

    for _, row in rules.iterrows():
        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])

        if product_name in antecedents:
            recommendations.extend(consequents)
        # optional: also check if selected product is in consequents
        elif product_name in consequents:
            recommendations.extend(antecedents)

    # remove same product if present
    recommendations = [item for item in recommendations if item != product_name]


    return list(set(recommendations))


def get_most_bought_products(df, top_n=10):
    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return top_products


def recommend_for_customer(customer_id, df, rules):
    customer_products = df[df["Customer ID"] == customer_id]["Description"].dropna().unique().tolist()

    recommendations = set()

    for product in customer_products:
        for _, row in rules.iterrows():
            antecedents = list(row["antecedents"])
            consequents = list(row["consequents"])

            if product in antecedents:
                recommendations.update(consequents)

    # Remove already bought products
    recommendations = recommendations - set(customer_products)

    return customer_products, list(recommendations)


st.title("📊 Customer Segmentation Dashboard")
st.markdown("Predict customer segments using *single customer input* or *CSV upload*.")
st.info("This app uses RFM features and KMeans clustering for segmentation.")

st.sidebar.header("About")
st.sidebar.write("Customer segmentation using RFM analysis and KMeans clustering.")

st.sidebar.header("Required CSV Columns")
st.sidebar.write("""
- Customer ID
- Invoice
- InvoiceDate
- Quantity
- Price
""")


mode = st.radio(
    "Choose input method",
    ["Single Customer Input", "Upload CSV File"]
)

if mode == "Single Customer Input":

    df = pd.read_csv("data/online_retail_II.csv")

    df["Invoice"] = df["Invoice"].astype(str)
    df["Description"] = df["Description"].astype(str)
    df["Customer ID"] = df["Customer ID"].astype(str)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["Invoice", "Description", "Customer ID"])
    df = df[df["Quantity"] > 0]
    

    rules = build_product_rules(df)

    # Create product list from rules (IMPORTANT)
    rule_products = set()

    for _, row in rules.iterrows():
        rule_products.update(list(row["antecedents"]))
        rule_products.update(list(row["consequents"]))

    product_list = sorted(rule_products)


    st.subheader("Customer Input")

    customer_id = st.text_input("Customer ID", key="customer_id")
    invoice_number = st.text_input("Invoice Number", key="invoice_number")
    invoice_date = st.date_input("Invoice Date", key="invoice_date")
    quantity = st.number_input("Quantity", key="quantity")
    price = st.number_input("Price", key="price")
    product_name = st.selectbox("Select Product Purchased",product_list)

    if st.button("Show Input Data"):
        try:
            if quantity <= 0 or price <= 0:
                st.error("Quantity and Price must be greater than 0")
            else:
                input_df = pd.DataFrame([{
                    "Customer ID": customer_id,
                    "Invoice": invoice_number,
                    "InvoiceDate": str(invoice_date),
                    "Quantity": quantity,
                    "Price": price
                }])

                st.write("Input Data:")
                st.dataframe(input_df)

                # Step 1: Create RFM
                rfm_df = create_rfm_single(invoice_date, quantity, price)

                st.write("Generated RFM Features:")
                st.dataframe(rfm_df)

                # Step 2: Scale
                scaled_data = scaler.transform(rfm_df)

                # Step 3: Predict
                cluster = kmeans_model.predict(scaled_data)[0]
                label_map = {
                0: "Regular Customers",
                1:"Low Engagement Customers",
                2: "Premium Customers"
                }


                segment = label_map.get(cluster, "Unknown")

                # Step 4: Show result
                st.success("Prediction completed successfully!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Recency", int(rfm_df["Recency"].iloc[0]))
                col2.metric("Frequency", int(rfm_df["Frequency"].iloc[0]))
                col3.metric("Monetary", float(rfm_df["Monetary"].iloc[0]))

                st.subheader("Prediction Result")
                st.write(f"Predicted segment: **{segment}*")


                st.subheader("Customer Segment")
                st.write(f"Cluster: {cluster}")

                recommendations = get_recommendation(cluster)

                st.subheader("Recommended for Customer")
                for rec in recommendations:
                    st.write(f"- {rec}")

                st.subheader("Product-Based Recommendations")

                recommended_products = recommend_products(product_name, rules)

                if recommended_products:
                    st.write(f"Customers who bought **{product_name}** also bought:")

                    for item in recommended_products[:10]:
                        st.write(f"✅ {item}")
                else:
                    st.warning("No related products found for this product.")

        except Exception as e:
                st.error(f"Error: {str(e)}")
        

elif mode == "Upload CSV File":
    st.subheader("Upload Customer Transactions CSV")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        df["Invoice"] = df["Invoice"].astype(str)
        df["Description"] = df["Description"].astype(str)
        df["Customer ID"] = df["Customer ID"].astype(str)
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

        df = df.dropna(subset=["Invoice", "Description", "Customer ID"])
        df = df[df["Quantity"] > 0]

        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("Predict Segments from CSV"):
            try:
                rfm = create_rfm_bulk(df)

                st.write("Generated RFM Data:")
                st.dataframe(rfm.head())

                scaled_data = scaler.transform(rfm)
                clusters = kmeans_model.predict(scaled_data)

                rfm["Cluster"] = clusters

                result_df = rfm.reset_index()

            

                st.subheader("Cluster Analysis")

                cluster_analysis = result_df.groupby("Cluster").agg({
                    "Monetary": "mean",
                    "Frequency": "mean",
                    "Recency": "mean"
                }).reset_index()

                st.dataframe(cluster_analysis)


                cluster_avg = result_df.groupby("Cluster")["Monetary"].mean().sort_values()

                cluster_labels = {}

                labels = ["Low Engagement Customers", "Regular Customers", "Premium Customers"]

                for i, cluster in enumerate(cluster_avg.index):
                    cluster_labels[cluster] = labels[i]

                result_df["Segment"] = result_df["Cluster"].map(cluster_labels)

                result_df["Recommendation"] = result_df["Cluster"].apply(
                lambda x: ", ".join(get_recommendation(x))
                )

                st.subheader("Customer Segments Distribution")

                segment_counts = result_df["Segment"].value_counts().reset_index()
                segment_counts.columns = ["Segment", "Customer Count"]

                st.dataframe(segment_counts)

        
                st.subheader("Customer Segments Chart")

                fig, ax = plt.subplots()

                ax.bar(
                    segment_counts["Segment"],
                    segment_counts["Customer Count"]
                )

                ax.set_xlabel("Segment")
                ax.set_ylabel("Customer Count")
                ax.set_title("Customer Segmentation")

                st.pyplot(fig)


                st.success("Bulk prediction completed successfully!")

                col1, col2 = st.columns(2)
                col1.metric("Total Customers", result_df.shape[0])
                col2.metric("Total Clusters Found", result_df["Cluster"].nunique())

                st.subheader(" Segmented Customers with rule based Recommendations")
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="customer_segmentation_recommendations_results.csv",
                mime="text/csv"
                )
                cluster_counts = result_df["Cluster"].value_counts().reset_index()
                cluster_counts.columns = ["Cluster", "Customer Count"]


                monetary_cluster = result_df.groupby("Cluster")["Monetary"].mean().reset_index()

                st.subheader("Average Monetary by Cluster")
                fig2, ax2 = plt.subplots()
                ax2.bar(monetary_cluster["Cluster"].astype(str), monetary_cluster["Monetary"])
                ax2.set_xlabel("Cluster")
                ax2.set_ylabel("Average Monetary")
                ax2.set_title("Average Spending by Cluster")
                st.pyplot(fig2)

                frequency_cluster = result_df.groupby("Cluster")["Frequency"].mean().reset_index()

                st.subheader("Average Frequency by Cluster")
                fig3, ax3 = plt.subplots()
                ax3.bar(frequency_cluster["Cluster"].astype(str), frequency_cluster["Frequency"])
                ax3.set_xlabel("Cluster")
                ax3.set_ylabel("Average Frequency")
                ax3.set_title("Average Purchase Frequency by Cluster")
                st.pyplot(fig3)

                st.subheader("Customer Segments Scatter Plot")

                fig, ax = plt.subplots()

                # Map segment to colors
                color_map = {
                    "Low Engagement Customers": "red",
                    "Regular Customers": "orange",
                    "Premium Customers": "green"
                }

                colors = result_df["Segment"].map(color_map)

                scatter = ax.scatter(
                    result_df["Frequency"],
                    result_df["Monetary"],
                    c=colors
                )

                ax.set_xlabel("Frequency")
                ax.set_ylabel("Monetary")
                ax.set_title("Customer Segmentation")

                # Create legend manually
                for segment, color in color_map.items():
                    ax.scatter([], [], c=color, label=segment)

                ax.legend(title="Customer Segments")

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Product recommendation could not be generated: {e}")

        st.subheader("Product Recommendation System")

        try:
            rules = build_product_rules(df)

            st.write("Total rules generated:", len(rules))

            if rules is None or rules.empty:
                st.warning("No association rules could be generated from the data.")
            else:

                # Product-based recommendation
                st.markdown("### Recommend by Product")
                rule_products = set()

                for _, row in rules.iterrows():
                    rule_products.update(list(row["antecedents"]))
                    rule_products.update(list(row["consequents"]))

                product_list = sorted(rule_products)

                selected_product = st.selectbox("Select a product", product_list)

                if st.button("Show Related Products", key="Show_Related_Products"):

                    recommended_products = recommend_products(selected_product, rules)

                    if recommended_products:
                        st.markdown(f"### 🛍️ Recommended Products for '{selected_product}'")
                        for item in recommended_products[:10]:
                            st.write(f"✅ {item}")
                    else:
                        st.warning("No related product recommendations found.")

                # Customer-based recommendation
                st.markdown("### Recommend by Customer ID")
                customer_list = sorted(df["Customer ID"].dropna().unique())
                selected_customer = st.selectbox("Select a customer ID", customer_list)

                if st.button("Show Customer Recommendations",key="Show_Customer_Recommendations"):
                    bought_products, customer_recs = recommend_for_customer(selected_customer, df, rules)

                    st.write("*Products already bought by this customer:*")
                    for item in bought_products[:10]:
                        st.write(f"🛒 {item}")

                    st.write("*Recommended products for this customer:*")
                    if customer_recs:
                        for item in customer_recs[:10]:
                            st.write(f"✅ {item}")
                    else:
                        st.warning("No customer-specific recommendations found.")

                # Most bought products
                st.markdown("### Most Bought Products")
                top_products = get_most_bought_products(df, top_n=10)
                st.dataframe(top_products.reset_index().rename(columns={"Quantity": "Total Quantity"}))                

                        
        except Exception as e:
            st.error(f"Error: {str(e)}")    
      








