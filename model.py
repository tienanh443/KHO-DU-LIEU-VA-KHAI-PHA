import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    conn_str = (
        r'DRIVER={ODBC Driver 18 for SQL Server};'
        r'SERVER=ANH\ANH,1433;'  
        r'DATABASE=FishDw;'
        r'UID=sa;'
        r'PWD=0968216425;'
        r'TrustServerCertificate=yes;'
    )
    conn = pyodbc.connect(conn_str)
    
    query = """
    SELECT 
        S.FishID,
        F.FishName,
        T.Date AS SaleDate,
        T.Year,
        T.Month,
        T.Quarter,
        S.Quantity AS QuantitySold,
        S.Revenue,
        S.Profit,
        I.Quantity AS InventoryQuantity,
        F.Price
    FROM 
        Fact_Sales S
    JOIN 
        Dim_Fish F ON S.FishID = F.FishID
    JOIN 
        Dim_Inventory I ON F.FishID = I.FishID
    JOIN 
        Dim_Time T ON S.TimeID = T.TimeID;
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df

def preprocess_data(df):
    # Xử lý giá trị NaN nếu có
    df.dropna(inplace=True)
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    cols_to_scale = ['QuantitySold', 'Revenue', 'Profit', 'InventoryQuantity', 'Price']
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    return df

def split_data(df):
    X = df[['InventoryQuantity', 'Price', 'Revenue', 'Profit']]
    y = df['QuantitySold']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("▶ Linear Regression:")
    print(f"MSE: {mse}")
    print(f"R-squared: {r2}")
      # Lưu kết quả vào file CSV
    result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    result_df.to_csv('Linear_Results.csv', index=False)
    print("\n✅ Kết quả dự đoán đã được lưu vào 'Linear_Results.csv'")

def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("▶ Linear Regression:")
    print(f"MSE: {mse}")
    print(f"R-squared: {r2}")



def train_ridge_regression(X_train, X_test, y_train, y_test):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("▶ Ridge Regression:")
    print(f"MSE: {mse}")
    print(f"R-squared: {r2}")


def train_lasso_regression(X_train, X_test, y_train, y_test):
    model = Lasso(alpha=0.01)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("▶ Lasso Regression:")
    print(f"MSE: {mse}")
    print(f"R-squared: {r2}")

def plot_results(y_test, y_pred, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color="blue", label="Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.legend()
    plt.show()


def apply_discount_on_unpopular_products(df, discount_rate=0.05):
    # Xác định ngưỡng trung bình
    mean_quantity = df['QuantitySold'].mean()
    mean_revenue = df['Revenue'].mean()
    mean_inventory = df['InventoryQuantity'].mean()

    # Đánh dấu các mặt hàng không bán chạy
    df['Low_Sales'] = (df['QuantitySold'] < mean_quantity) & (df['Revenue'] < mean_revenue)

    # Tính giá sau khi giảm cho các mặt hàng không bán chạy
    df['Discounted_Price'] = df['Price']
    df.loc[df['Low_Sales'], 'Discounted_Price'] = df['Price'] * (1 - discount_rate)

    # Lưu vào file CSV
    output_file = 'Discounted_Fish_Data.csv'
    df.to_csv(output_file, index=False)

    print(f"Dữ liệu đã được xử lý và lưu vào '{output_file}'.")
    return df[['FishName', 'QuantitySold', 'InventoryQuantity', 'Price', 'Discounted_Price']]


def main():
    # 1. Load dữ liệu
    df = load_data()
    
    # 2. Tiền xử lý dữ liệu
    df = preprocess_data(df)
    
    # 3. Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 4. Linear Regression
    train_linear_regression(X_train, X_test, y_train, y_test)
    
    # 5. Ridge Regression
    train_ridge_regression(X_train, X_test, y_train, y_test)
    
    # 6. Lasso Regression
    train_lasso_regression(X_train, X_test, y_train, y_test)
    
    # 7. Vẽ biểu đồ kết quả cho Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    plot_results(y_test, y_pred, title="Ridge Regression Results")
    
    # 8. Lưu kết quả vào file CSV
    df.to_csv("Processed_Fish_Data.csv", index=False)
    print("\n✅ Dữ liệu đã được lưu vào 'Processed_Fish_Data.csv'.")

    result = apply_discount_on_unpopular_products(df, discount_rate=0.05)
    print(result)

if __name__ == "__main__":
    main()

