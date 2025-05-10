import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tkinter as tk
from tkinter import messagebox

# Read database
df = pd.read_csv('diabetes.csv')
df = df[df['Insulin'] != 0]

# Train the model
X = df.drop(columns=['Insulin', 'Outcome']) 
y = df['Insulin']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13) 
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R² score:", r2_score(y_test, y_pred))
print("Mean squared error:", mean_squared_error(y_test, y_pred))

# Make the interface
features = X.columns.tolist()  
root = tk.Tk()  
root.title('Insulin Dosage Predictor') 
entries = {}  

def predict_insulin():
    try:
        values = [float(entries[feature].get()) for feature in features]
        input_df = pd.DataFrame([values], columns=features)
        prediction = model.predict(input_df)[0]
        messagebox.showinfo("Prediction", f"Predicted Insulin Level: {prediction:.2f} μU/mL")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for all fields.")

for idx, feature in enumerate(features):
    label = tk.Label(root, text=feature)
    label.grid(row=idx, column=0, padx=10, pady=5, sticky='e')
    
    entry = tk.Entry(root)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    entries[feature] = entry

predict_button = tk.Button(root, text="Predict Insulin Level", command=predict_insulin)
predict_button.grid(row=len(features), column=0, columnspan=2, pady=20)

root.mainloop()

#first ML mini project:)
