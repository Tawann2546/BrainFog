import streamlit as st
st.set_page_config(page_title='BrainFog', layout='wide')
st.markdown("""
<h1 style='text-align: left; font-size: 60px;'>🧠 BrainFog</h1>
""", unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder 

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
st.title("📈 คาดการณ์แนวเกมที่จะเติบโต")
years_forward = st.slider("📅 เลือกจำนวนปีในอนาคตที่ต้องการทำนาย", 1, 5, 5)

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

st.subheader("📊 แนวเกมที่มีแนวโน้มเติบโต")
st.dataframe(trend_df.set_index('Genre').style.format("{:.3f}"))

# ----------------------------
# 🔢 ข้อ 2: คาดการณ์ยอดขายของแนวเกมแต่ละประเภท แยกตามภูมิภาค
# ----------------------------

# ✅ ประกาศคอลัมน์ภูมิภาคที่ใช้ในการทำนาย
region_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

with st.container():
    st.markdown("## 🧩 ข้อ 2: คาดการณ์ยอดขายของแนวเกมแต่ละประเภท แยกตามภูมิภาค (ล้านหน่วย)")

    # ✅ UI ให้ผู้ใช้เลือกจำนวนปีที่ต้องการทำนาย
    st.markdown("### 🗓️ เลือกจำนวนปีในอนาคตที่ต้องการทำนาย")
    predict_years = st.slider(" ", 1, 5, 3, key="region_years")
    future_years = list(range(2017, 2017 + predict_years))

    # ✅ รวมยอดขายแนวเกมต่อปี
    genre_region_year = df.groupby(['Year_of_Release', 'Genre'])[region_cols].sum().reset_index()
    last_year = genre_region_year['Year_of_Release'].max()
    latest_sales = genre_region_year[genre_region_year['Year_of_Release'] == last_year].copy()

    # ✅ สร้างข้อมูลยอดขายจำลองสำหรับปีอนาคต
    future_data = []
    for year in future_years:
        temp = latest_sales.copy()
        temp['Year_of_Release'] = year
        future_data.append(temp)
    future_df = pd.concat(future_data, ignore_index=True)

    # ✅ รวมอดีตและอนาคต
    all_region_data = pd.concat([genre_region_year, future_df], ignore_index=True)
    all_region_sales = all_region_data.groupby('Genre')[region_cols].sum().reset_index()

    # ✅ แปลง Genre เป็นตัวเลข
    le = LabelEncoder()
    all_region_sales['Genre_encoded'] = le.fit_transform(all_region_sales['Genre'])

    # ✅ เทรนโมเดลใหม่
    X = all_region_sales[['Genre_encoded']]
    y = all_region_sales[region_cols]
    model2 = LinearRegression()
    model2.fit(X, y)

    # ✅ ทำนาย
    preds = model2.predict(X)
    pred_df = pd.DataFrame(preds, columns=region_cols)
    pred_df['Genre'] = all_region_sales['Genre']

    # ✅ แสดงตารางผลลัพธ์
    st.dataframe(pred_df.set_index('Genre').round(2))

    # ✅ แสดงกราฟแท่ง stacked
    st.markdown("### 📊 กราฟเปรียบเทียบยอดขายแต่ละแนวเกมในแต่ละภูมิภาค")
    st.bar_chart(pred_df.set_index('Genre')[region_cols].round(2))

# ----------------------------
# 🔮 ข้อ 3: ความสัมพันธ์ของ Publisher กับยอดขายในอนาคต
# ----------------------------

st.header("ข้อ 3: ความสัมพันธ์ของ Publisher กับยอดขายในอนาคต")

# 📅 UI: เลือกจำนวนปีล่วงหน้า
n_years_pub = st.slider("เลือกจำนวนปีในอนาคตเพื่อพยากรณ์ยอดขาย (Publisher)", 1, 5, 5)

# ✅ เตรียมข้อมูล
df_pub = df[['Year_of_Release', 'Publisher', 'Global_Sales']].dropna()
df_pub['Year_of_Release'] = df_pub['Year_of_Release'].astype(int)
df_pub = df_pub[df_pub['Year_of_Release'] >= 2010]

# ✅ รวมยอดขายของแต่ละ Publisher ต่อปี
pub_sales = df_pub.groupby(['Year_of_Release', 'Publisher'])['Global_Sales'].sum().reset_index()

# ✅ วนลูปเทรน + พยากรณ์
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

# ✅ สรุปผลลัพธ์
future_df = pd.DataFrame(future_predictions, columns=['Publisher', 'Year', 'Predicted_Sales'])
publisher_summary = future_df.groupby('Publisher')['Predicted_Sales'].sum().reset_index()
publisher_summary = publisher_summary.sort_values(by='Predicted_Sales', ascending=False).head(10)

# ✅ แสดงผล
st.subheader(f"📊 10 อันดับ Publisher ที่คาดว่าจะมียอดขายรวมสูงสุดใน {n_years_pub} ปีข้างหน้า")
st.dataframe(publisher_summary.set_index('Publisher').round(2))

# ✅ กราฟ
st.bar_chart(publisher_summary.set_index('Publisher'))
