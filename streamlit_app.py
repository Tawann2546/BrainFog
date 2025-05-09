import streamlit as st
st.set_page_config(page_title='BrainFog', layout='wide')
st.markdown("""
<h1 style='text-align: left; font-size: 60px;'>🧠 BrainFog 🌬️</h1>
""", unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor




# โหลดข้อมูล
@st.cache_data
def load_data():
    df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
    df = df.dropna(subset=['Year_of_Release', 'Genre', 'Global_Sales'])
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    return df

df = load_data()

# คำนวณยอดขายแนวเกมต่อปี
genre_year = df.groupby(['Year_of_Release', 'Genre'])['Global_Sales'].sum().reset_index()

# UI รับจำนวนปีจากผู้ใช้
st.title("ข้อ 1 :คาดการณ์แนวเกมที่จะเติบโต")
years_forward = st.slider("เลือกจำนวนปีในอนาคตที่ต้องการทำนาย", 1, 5, 5)

# จำลองยอดขายในอนาคต
future_years = pd.DataFrame([(year, genre) for year in range(2025, 2025 + years_forward) for genre in genre_year['Genre'].unique()],
                            columns=['Year_of_Release', 'Genre'])
future_years['Predicted_Sales'] = 0  # สมมติว่าเริ่มที่ 0 ทุกปี แล้วใช้ slope ทำนาย

# เตรียมข้อมูลทั้งหมด
genre_all_years = pd.concat([
    genre_year[['Year_of_Release', 'Genre', 'Global_Sales']],
    future_years.rename(columns={'Predicted_Sales': 'Global_Sales'})[['Year_of_Release', 'Genre', 'Global_Sales']]
])

# คำนวณ slope แนวโน้มการเติบโตของแต่ละแนวเกม
growth_results = []
for genre in genre_all_years['Genre'].unique():
    subset = genre_all_years[genre_all_years['Genre'] == genre]
    X = subset[['Year_of_Release']]
    y = subset['Global_Sales']
    
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    growth_results.append((genre, slope))

# แสดงผลลัพธ์
trend_df = pd.DataFrame(growth_results, columns=['Genre', 'Growth_Trend'])
trend_df = trend_df.sort_values(by='Growth_Trend', ascending=False)

st.dataframe(trend_df.set_index('Genre').style.format("{:.3f}"))

# ----------------------------
# ข้อ 2: คาดการณ์ยอดขายของแนวเกมแต่ละประเภท แยกตามภูมิภาค
# ----------------------------

# ประกาศคอลัมน์ภูมิภาคที่ใช้ในการทำนาย
region_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

with st.container():
    st.title("ข้อ 2: คาดการณ์ยอดขายของแนวเกมแต่ละประเภท แยกตามภูมิภาค (ล้านหน่วย)")

    # UI ให้ผู้ใช้เลือกจำนวนปีที่ต้องการทำนาย
    st.markdown("เลือกจำนวนปีในอนาคตที่ต้องการทำนาย")
    predict_years = st.slider(" ", 1, 5, 3, key="region_years")
    future_years = list(range(2017, 2017 + predict_years))

    # รวมยอดขายแนวเกมต่อปี
    genre_region_year = df.groupby(['Year_of_Release', 'Genre'])[region_cols].sum().reset_index()
    last_year = genre_region_year['Year_of_Release'].max()
    latest_sales = genre_region_year[genre_region_year['Year_of_Release'] == last_year].copy()

    # สร้างข้อมูลยอดขายจำลองสำหรับปีอนาคต
    future_data = []
    for year in future_years:
        temp = latest_sales.copy()
        temp['Year_of_Release'] = year
        future_data.append(temp)
    future_df = pd.concat(future_data, ignore_index=True)

    # รวมอดีตและอนาคต
    all_region_data = pd.concat([genre_region_year, future_df], ignore_index=True)
    all_region_sales = all_region_data.groupby('Genre')[region_cols].sum().reset_index()

    # แปลง Genre เป็นตัวเลข
    le = LabelEncoder()
    all_region_sales['Genre_encoded'] = le.fit_transform(all_region_sales['Genre'])

    # เทรนโมเดลใหม่
    X = all_region_sales[['Genre_encoded']]
    y = all_region_sales[region_cols]
    model2 = LinearRegression()
    model2.fit(X, y)

    # ทำนาย
    preds = model2.predict(X)
    pred_df = pd.DataFrame(preds, columns=region_cols)
    pred_df['Genre'] = all_region_sales['Genre']

    # แสดงตารางผลลัพธ์
    st.dataframe(pred_df.set_index('Genre').round(2))

    # แสดงกราฟแท่ง stacked
    st.markdown("กราฟเปรียบเทียบยอดขายแต่ละแนวเกมในแต่ละภูมิภาค")
    st.bar_chart(pred_df.set_index('Genre')[region_cols].round(2))

# ----------------------------
# ข้อ 3: ความสัมพันธ์ของ Publisher กับยอดขายในอนาคต
# ----------------------------

st.title("ข้อ 3: ความสัมพันธ์ของ Publisher กับยอดขายในอนาคต")

# UI: เลือกจำนวนปีล่วงหน้า
n_years_pub = st.slider("เลือกจำนวนปีในอนาคตเพื่อพยากรณ์ยอดขาย (Publisher)", 1, 5, 5)

# เตรียมข้อมูล
df_pub = df[['Year_of_Release', 'Publisher', 'Global_Sales']].dropna()
df_pub['Year_of_Release'] = df_pub['Year_of_Release'].astype(int)
df_pub = df_pub[df_pub['Year_of_Release'] >= 2010]

# รวมยอดขายของแต่ละ Publisher ต่อปี
pub_sales = df_pub.groupby(['Year_of_Release', 'Publisher'])['Global_Sales'].sum().reset_index()

# วนลูปเทรน + พยากรณ์
future_predictions = []
for pub in pub_sales['Publisher'].unique():
    subset = pub_sales[pub_sales['Publisher'] == pub]
    X = subset[['Year_of_Release']]
    y = subset['Global_Sales']

    if len(X) >= 3:
        model = LinearRegression()
        model.fit(X, y)

        for year in range(2025, 2025 + n_years_pub):
            pred = model.predict(pd.DataFrame({'Year_of_Release': [year]}))[0]
            future_predictions.append((pub, year, pred))

# สรุปผลลัพธ์
future_df = pd.DataFrame(future_predictions, columns=['Publisher', 'Year', 'Predicted_Sales'])
publisher_summary = future_df.groupby('Publisher')['Predicted_Sales'].sum().reset_index()
publisher_summary = publisher_summary.sort_values(by='Predicted_Sales', ascending=False).head(10)

# แสดงผล
st.subheader(f"10 อันดับ Publisher ที่คาดว่าจะมียอดขายรวมสูงสุดใน {n_years_pub} ปีข้างหน้า")
st.dataframe(publisher_summary.set_index('Publisher').round(2))

# กราฟ
st.bar_chart(publisher_summary.set_index('Publisher'))

# ----------------------------
#  ข้อ 4: คาดการณ์จำนวนเกมใหม่ของแต่ละค่ายเกมในอนาคต (ด้วย XGBoost + UI)
# ----------------------------

st.header("ข้อ 4: คาดการณ์จำนวนเกมใหม่ของแต่ละค่ายเกมในอนาคต")

n_years_future = st.slider("เลือกจำนวนปีในอนาคตเพื่อทำนายจำนวนเกม (Publisher)", 1, 5, 5, key="pub_year_slider")
future_years = list(range(2025, 2025 + n_years_future))

# เตรียมข้อมูล
df_filtered = df[['Year_of_Release', 'Publisher']].copy()
df_filtered = df_filtered.dropna()
df_filtered['Year_of_Release'] = df_filtered['Year_of_Release'].astype(int)
df_filtered = df_filtered[(df_filtered['Year_of_Release'] >= 2010) & (df_filtered['Year_of_Release'] <= 2016)]

# สร้างตารางจำนวนเกมต่อค่ายต่อปี
publisher_year = df_filtered.groupby(['Year_of_Release', 'Publisher']).size().reset_index(name='Game_Count')

# เข้ารหัส Publisher
publisher_le = LabelEncoder()
publisher_year['Publisher_encoded'] = publisher_le.fit_transform(publisher_year['Publisher'])

# เตรียมข้อมูลเทรน
X = publisher_year[['Year_of_Release', 'Publisher_encoded']]
y = publisher_year['Game_Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล + Hyperparameter Tuning
param_grid = {
    'n_estimators': [20],
    'max_depth': [3],
    'learning_rate': [0.1],
    'subsample': [1.0],
    'colsample_bytree': [0.8]
}
grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=0
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# ทำนายจำนวนเกมในอนาคต
top_publishers = publisher_year['Publisher_encoded'].value_counts().index[:10]
future_pub = pd.DataFrame({
    'Year_of_Release': np.repeat(future_years, len(top_publishers)),
    'Publisher_encoded': np.tile(top_publishers, len(future_years))
})
future_pub['Predicted_Games'] = best_model.predict(future_pub)
future_pub['Publisher'] = publisher_le.inverse_transform(future_pub['Publisher_encoded'])

# สรุปผล
publisher_total_games = future_pub.groupby('Publisher')['Predicted_Games'].sum().reset_index()
publisher_total_games.columns = ['Publisher', 'Total_Predicted_Games']
publisher_total_games = publisher_total_games.sort_values(by='Total_Predicted_Games', ascending=False).head(10)

# แสดงผล
st.subheader(f" 10 อันดับ Publisher ที่คาดว่าจะออกเกมมากที่สุดใน {n_years_future} ปีข้างหน้า")
st.dataframe(publisher_total_games.set_index('Publisher').round(0))
st.bar_chart(publisher_total_games.set_index('Publisher'))

