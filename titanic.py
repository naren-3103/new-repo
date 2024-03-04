import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
import pandas as pd

url = "https://raw.githubusercontent.com/naren-3103/new-repo/test/output.xlsx"

# Try reading the CSV file with different parameters
try:
    titanic_data = pd.read_excel(url)
except pd.errors.ParserError as e:
    print("Error:", e)


# Feature selection
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = titanic_data[features]
y = titanic_data["Survived"]

# Convert categorical variables to numerical
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
