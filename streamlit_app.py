
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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
