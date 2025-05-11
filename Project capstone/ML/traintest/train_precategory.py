import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold

# โหลดข้อมูลจากไฟล์
df = pd.read_csv(r'C:\Users\User\Desktop\ML\.venv\data\train_precategory.csv')

# แสดงข้อมูลเบื้องต้น
print(df.head())

# จัดกลุ่มช่วงอายุ
age_bins = [18, 29, 39, float('inf')]
age_labels = ['18-29', '30-39', '40+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=True)


# การเข้ารหัสข้อมูล (Label Encoding)
label_encoder_age = LabelEncoder()
label_encoder_interested = LabelEncoder()
label_encoder_category = LabelEncoder()
label_encoder_gender = LabelEncoder()  # เพิ่มการเข้ารหัส gender

df['age_group_encoded'] = label_encoder_age.fit_transform(df['age_group'])
df['interested_encoded'] = label_encoder_interested.fit_transform(df['interested'])
df['category_encoded'] = label_encoder_category.fit_transform(df['category'])
df['gender_encoded'] = label_encoder_gender.fit_transform(df['gender'])  # เพิ่ม gender_encoded

# บันทึก LabelEncoder
joblib.dump(label_encoder_age, 'label_encoder_age.pkl')
joblib.dump(label_encoder_interested, 'label_encoder_interested.pkl')
joblib.dump(label_encoder_category, 'label_encoder_category.pkl')
joblib.dump(label_encoder_gender, 'label_encoder_gender.pkl')  # บันทึก LabelEncoder สำหรับ gender
print("บันทึก LabelEncoder สำเร็จ")

# เลือก features (age_group_encoded, interested_encoded, gender_encoded) และ target (category_encoded)
X = df[['age_group_encoded', 'interested_encoded', 'gender_encoded']]  # เพิ่ม gender_encoded
y = df['category_encoded']

# แบ่งข้อมูลเป็นชุดฝึก (train) และชุดทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# สร้างโมเดล Decision Tree
model = DecisionTreeClassifier(random_state=42)

# ใช้ KFold สำหรับ 4-fold cross-validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)

accuracies = []  # เก็บค่า accuracy ของแต่ละ fold

# ทำการฝึกและทดสอบในแต่ละ fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # ฝึกโมเดล
    model.fit(X_train, y_train)
    
    # ทำนายผล
    y_pred = model.predict(X_test)
    
    # ประเมินความแม่นยำ
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# คำนวณค่าเฉลี่ยของ accuracy จากทั้ง 4 รอบ
average_accuracy = sum(accuracies) / len(accuracies)
print(f'Average Accuracy across 4 folds: {average_accuracy * 100:.2f}%')

# ตอนนี้, คุณต้องใช้ชุดทดสอบ (X_test, y_test) สำหรับการประเมินผล
# ทำนายผลบนข้อมูลทดสอบ
y_pred_test = model.predict(X_test)

# คำนวณค่า accuracy, precision, recall, f1-score และ confusion matrix
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred_test)

# แสดงผลลัพธ์
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# บันทึกโมเดลที่ฝึกแล้ว
joblib.dump(model, 'predict_category_model.pkl')
print("บันทึกโมเดลสำเร็จ: predict_category_model.pkl")
