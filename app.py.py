# -*- coding: utf-8 -*-
# AI DataInsight Dashboard - أداة تحليل البيانات بالذكاء الاصطناعي
# للمحاضر هاشم غريب - كورس تحليل البيانات بالذكاء الاصطناعي

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ========== إعدادات الصفحة ==========
st.set_page_config(
    page_title="AI DataInsight Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== العنوان الرئيسي ==========
st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>
        📊 AI DataInsight Dashboard
    </h1>
    <p style='text-align: center; font-size: 18px;'>
        أداة تحليل البيانات بالذكاء الاصطناعي – للمحاضر هاشم غريب
    </p>
""", unsafe_allow_html=True)

st.divider()

# ========== رفع الملف ==========
uploaded_file = st.file_uploader(
    "📂 حمّل ملف البيانات (Excel أو CSV)",
    type=["xlsx", "xls", "csv"]
)

# ========== معالجة البيانات ==========
if uploaded_file is not None:
    # قراءة الملف
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ تم تحميل الملف بنجاح!")
    except Exception as e:
        st.error(f"❌ حدث خطأ في قراءة الملف: {e}")
        st.stop()

    # عرض عينة من البيانات
    st.subheader("📋 نظرة عامة على البيانات")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**عدد الصفوف:** {df.shape[0]}")
        st.write(f"**عدد الأعمدة:** {df.shape[1]}")
    with col2:
        st.write("**أسماء الأعمدة:**")
        st.write(df.columns.tolist())

    with st.expander("🔍 عرض أول 10 صفوف من البيانات"):
        st.dataframe(df.head(10), use_container_width=True)

    # ========== تحليل البيانات الأساسي ==========
    st.divider()
    st.subheader("📈 الإحصائيات الوصفية")
    
    # اختيار الأعمدة الرقمية فقط
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # ========== رسم بياني تفاعلي ==========
        st.divider()
        st.subheader("📊 الرسوم البيانية التفاعلية")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox(
                "اختر نوع الرسم البياني:",
                ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram", "Box Plot"]
            )
        
        with col2:
            x_axis = st.selectbox("اختر المحور X:", numeric_cols)
            y_axis = st.selectbox("اختر المحور Y (اختياري لبعض الرسوم):", [None] + numeric_cols)
        
        # إنشاء الرسم البياني حسب الاختيار
        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart - {x_axis} vs {y_axis}")
        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"Line Chart - {x_axis} vs {y_axis}")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_axis, values=y_axis, title=f"Pie Chart - {x_axis}")
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot - {x_axis} vs {y_axis}")
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_axis, title=f"Histogram - {x_axis}")
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_axis, y=y_axis, title=f"Box Plot - {x_axis} vs {y_axis}")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ========== التنبؤ بالذكاء الاصطناعي ==========
        st.divider()
        st.subheader("🤖 التنبؤ والتوصيات الذكية")
        
        # التنبؤ باستخدام Linear Regression (إذا كان لدينا على الأقل عمودين رقميين)
        if len(numeric_cols) >= 2:
            st.write("**تنبؤ قيمة مستقبلية بناءً على البيانات الحالية:**")
            
            target_col = st.selectbox("اختر العمود المراد التنبؤ به (المتغير التابع):", numeric_cols)
            feature_cols = st.multiselect("اختر الأعمدة المستقلة (المتغيرات المُفسرة):", [col for col in numeric_cols if col != target_col])
            
            if len(feature_cols) > 0:
                X = df[feature_cols]
                y = df[target_col]
                
                # تقسيم البيانات
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # تدريب النموذج
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # عرض النتائج
                st.write(f"**دقة النموذج (R²):** {r2_score(y_test, y_pred):.2f}")
                st.write(f"**متوسط الخطأ التربيعي (MSE):** {mean_squared_error(y_test, y_pred):.2f}")
                
                # رسم المقارنة بين القيم الفعلية والمتوقعة
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='التنبؤات'))
                fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                              mode='lines', name='الخط المثالي'))
                fig_pred.update_layout(title="مقارنة القيم الفعلية مقابل المتوقعة")
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # توصيات ذكية
                st.subheader("💡 توصيات مبنية على التحليل:")
                if r2_score(y_test, y_pred) > 0.7:
                    st.success("✅ النموذج قوي ويمكن الاعتماد عليه في التنبؤات.")
                else:
                    st.warning("⚠️ النموذج يحتاج إلى تحسين، جرب إضافة المزيد من المتغيرات أو استخدام نموذج أكثر تعقيدًا.")
                
                # التنبؤ بقيمة جديدة (وهمية)
                st.write("**جرب التنبؤ بقيمة جديدة:**")
                input_values = []
                for col in feature_cols:
                    val = st.number_input(f"أدخل قيمة لـ {col}:", value=float(df[col].mean()))
                    input_values.append(val)
                
                if st.button("🔮 توقع القيمة"):
                    new_pred = model.predict([input_values])
                    st.success(f"القيمة المتوقعة لـ {target_col} هي: **{new_pred[0]:.2f}**")
        else:
            st.info("ℹ️ تحتاج إلى عمودين رقميين على الأقل للقيام بالتنبؤ.")
    
    else:
        st.warning("⚠️ لم يتم العثور على أعمدة رقمية في الملف. يرجى رفع ملف يحتوي على بيانات رقمية.")
    
    # ========== تصدير التقرير ==========
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 تنزيل البيانات النظيفة (CSV)"):
            df.to_csv("data_cleaned.csv", index=False)
            st.success("تم تنزيل الملف بنجاح!")
    
    with col2:
        if st.button("📊 تنزيل التقرير الإحصائي (CSV)"):
            if len(numeric_cols) > 0:
                df[numeric_cols].describe().to_csv("statistical_report.csv")
                st.success("تم تنزيل التقرير بنجاح!")
    
    with col3:
        if st.button("🧹 إعادة تعيين التحليل"):
            st.experimental_rerun()

else:
    # إذا لم يتم رفع ملف
    st.info("👆 يرجى رفع ملف Excel أو CSV لبدء التحليل.")
    st.image("https://img.icons8.com/fluency/96/000000/upload--v2.png", width=150)