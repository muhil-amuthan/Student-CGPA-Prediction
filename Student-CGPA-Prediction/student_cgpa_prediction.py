import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = {'Attendance': [95, 78, 88, 65, 92, 70, 85, 60],'Internal_Marks': [85, 72, 80, 60, 87, 68, 78, 55],
        'Aptitude_Score': [78, 65, 75, 55, 82, 60, 72, 50],'Programming_Score': [82, 70, 79, 58, 85, 63, 75, 52],'Final_CGPA': [8.9, 7.2, 8.4, 6.5, 9.1, 6.9, 8.0, 6.0]}
df = pd.DataFrame(data)
X = df[['Attendance', 'Internal_Marks', 'Aptitude_Score', 'Programming_Score']]
y = df['Final_CGPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Actual CGPA:", y_test.values)
print("Predicted CGPA:", y_pred)
print("Mean Squared Error (MSE):", round(mse, 4))

importance = pd.DataFrame({'Feature': X.columns,'Coefficient': model.coef_})

print("\nFeature Importance:")
print(importance.sort_values(by='Coefficient', ascending=False))
