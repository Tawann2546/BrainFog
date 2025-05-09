
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title='BrainFog', layout='wide')
st.markdown("""<h1 style='text-align: left; font-size: 60px;'>🧠 BrainFog</h1>""", unsafe_allow_html=True)

# โหลดข้อมูล
@st.cache_data
def load_data():
    df = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")
    df = df.dropna(subset=['Year_of_Release', 'Genre', 'Global_Sales', 'Publisher', 'Platform'])
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    return df

df = load_data()

# ----------------------------
# ✅ ข้อ 1: แนวเกมที่จะเติบโต
# ----------------------------

st.header("ข้อ 1: คาดการณ์แนวเกมที่จะเติบโตในอนาคต")
years_forward = st.slider("เลือกจำนวนปีในอนาคต", 1, 5, 3)
genre_year = df.groupby(['Year_of_Release', 'Genre'])['Global_Sales'].sum().reset_index()

genre_le = LabelEncoder()
genre_year['Genre_encoded'] = genre_le.fit_transform(genre_year['Genre'])

X = genre_year[['Year_of_Release', 'Genre_encoded']]
y = genre_year['Global_Sales']
model = LinearRegression()
model.fit(X, y)

future_years = pd.DataFrame({
    'Year_of_Release': np.repeat(np.arange(2017, 2017 + years_forward), len(genre_le.classes_)),
    'Genre_encoded': np.tile(np.arange(len(genre_le.classes_)), years_forward)
})
future_years['Genre'] = genre_le.inverse_transform(future_years['Genre_encoded'])
future_years['Predicted_Sales'] = model.predict(future_years[['Year_of_Release', 'Genre_encoded']])

# สรุปแนวโน้ม
genre_all = pd.concat([
    genre_year[['Year_of_Release', 'Genre', 'Global_Sales']],
    future_years.rename(columns={'Predicted_Sales': 'Global_Sales'})[['Year_of_Release', 'Genre', 'Global_Sales']]
])
trend = []
for genre in genre_all['Genre'].unique():
    subset = genre_all[genre_all['Genre'] == genre]
    X = subset[['Year_of_Release']]
    y = subset['Global_Sales']
    reg = LinearRegression().fit(X, y)
    trend.append((genre, reg.coef_[0]))

trend_df = pd.DataFrame(trend, columns=['Genre', 'Growth_Trend']).sort_values(by='Growth_Trend', ascending=False)
st.dataframe(trend_df, use_container_width=True)
st.bar_chart(trend_df.set_index('Genre').head(5))

# ----------------------------
# ✅ ข้อ 2: ยอดขายในแต่ละภูมิภาค + ปีล่วงหน้า
# ----------------------------

st.header("ข้อ 2: คาดการณ์ยอดขายของแนวเกมในแต่ละภูมิภาค")

region_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
predict_years = st.slider("เลือกจำนวนปีในอนาคตที่ต้องการทำนาย", 1, 5, 3, key="region_slider")

# คำนวณยอดขายเฉลี่ยตาม genre/region
region_df = df.groupby(['Year_of_Release', 'Genre'])[region_cols].sum().reset_index()
last_year = region_df['Year_of_Release'].max()
latest_sales = region_df[region_df['Year_of_Release'] == last_year].copy()

# จำลองยอดขายเพิ่มขึ้นตาม growth rate
growth_rate = {'NA_Sales': 0.03, 'EU_Sales': 0.025, 'JP_Sales': 0.01, 'Other_Sales': 0.015}
future_rows = []
for year in range(last_year + 1, last_year + 1 + predict_years):
    temp = latest_sales.copy()
    temp['Year_of_Release'] = year
    for col in region_cols:
        temp[col] *= (1 + growth_rate[col]) ** (year - last_year)
    future_rows.append(temp)
future_df = pd.concat(future_rows, ignore_index=True)
all_data = pd.concat([region_df, future_df], ignore_index=True)

# สรุปยอดขายรวมทั้งหมด
summary_df = all_data.groupby('Genre')[region_cols].sum().reset_index()
st.dataframe(summary_df.set_index('Genre').round(2))
st.bar_chart(summary_df.set_index('Genre')[region_cols])
