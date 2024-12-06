import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv(r"Predictive_Maintenance_Dataset.csv")

# Initialize LabelEncoders
label_encoder_product_id = LabelEncoder()
label_encoder_type = LabelEncoder()

# Encode 'Product ID' and 'Type' columns
df['Product ID'] = label_encoder_product_id.fit_transform(df['Product ID'])
df['Type'] = label_encoder_type.fit_transform(df['Type'])

# Separate features and target
X = df.drop(["Machine failure", "UDI"], axis=1)  # Features
y = df["Machine failure"]  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Apply StandardScaler to the numeric features only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB()
}

# Train models and evaluate performance
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred)}")
    print("-" * 50)

# Save the model and scaler for future use
with open('machine_pred.pkl', 'wb') as model_file:
    pickle.dump(models['Decision Tree'], model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('label_encoder_product_id.pkl', 'wb') as le_product_id_file:
    pickle.dump(label_encoder_product_id, le_product_id_file)

with open('label_encoder_type.pkl', 'wb') as le_type_file:
    pickle.dump(label_encoder_type, le_type_file)

print("Model, scaler, and label encoders saved successfully.")

# Optionally, save additional files like a dataset visualization or model evaluation
# Visualizing the distribution of selected features (e.g., Air temperature [K])
sns.histplot(df['Air temperature [K]'], kde=True)
plt.title('Distribution of Air temperature [K]')
plt.show()
