import pandas as pd
import joblib

# โหลดโมเดลที่ฝึกแล้ว
model = joblib.load('predict_category_model.pkl')

# โหลด LabelEncoder ที่ใช้ฝึก
label_encoder_age = joblib.load('label_encoder_age.pkl')
label_encoder_interested = joblib.load('label_encoder_interested.pkl')
label_encoder_category = joblib.load('label_encoder_category.pkl')
label_encoder_gender = joblib.load('label_encoder_gender.pkl')  # โหลด LabelEncoder สำหรับ gender

# โหลดข้อมูลจากไฟล์ test_precategory.csv
test_data = pd.read_csv(r'C:\Users\User\Desktop\ML\.venv\data\real_data.csv')

# ตรวจสอบว่ามีคอลัมน์ 'age', 'interesting', และ 'gender' หรือไม่
if 'age' not in test_data.columns or 'interested' not in test_data.columns or 'gender' not in test_data.columns:
    raise ValueError("DataFrame must contain 'age', 'interested', and 'gender' columns")

# จัดการข้อมูลที่ขาดหายไป (ถ้ามี)
test_data.dropna(subset=['age', 'interested', 'gender'], inplace=True)

# จัดกลุ่มช่วงอายุให้ตรงกับโมเดล
age_bins = [18, 29, 39, float('inf')]
age_labels = ['18-29', '30-39', '40+']
test_data['age_group'] = pd.cut(test_data['age'], bins=age_bins, labels=age_labels, right=True)

# แปลงข้อมูล 'interestied', 'age_group', และ 'gender' ให้เป็นค่าที่ใช้ได้
test_data['interested_encoded'] = label_encoder_interested.transform(test_data['interested'])
test_data['age_group_encoded'] = label_encoder_age.transform(test_data['age_group'])
test_data['gender_encoded'] = label_encoder_gender.transform(test_data['gender'])  # แปลง gender

# สร้าง DataFrame สำหรับข้อมูลผู้ใช้
X_test = test_data[['age_group_encoded', 'interested_encoded', 'gender_encoded']]  # เพิ่ม gender_encoded

# ทำนาย category สำหรับผู้ใช้
predicted_category_encoded = model.predict(X_test)
predicted_category = label_encoder_category.inverse_transform(predicted_category_encoded)

# เพิ่มคอลัมน์ 'predicted_category' ลงใน DataFrame
test_data['predicted_category'] = predicted_category

# สร้าง DataFrame ใหม่สำหรับผลลัพธ์ที่ต้องการ
result_df = test_data[['user_id', 'age', 'gender', 'interested', 'predicted_category']]

# แสดงผลลัพธ์
print(result_df)

# บันทึกผลลัพธ์เป็นไฟล์ใหม่
result_df.to_csv('predicted_test_results_with_gender.csv', index=False)
print("ผลลัพธ์ที่ทำนายได้ถูกบันทึกเป็นไฟล์ predicted_test_results_with_gender.csv")
