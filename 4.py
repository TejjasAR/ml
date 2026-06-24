import pandas as pd

# Create dataset
data = pd.DataFrame({
    "Weather": ["Sunny", "Sunny", "Rainy", "Sunny"],
    "Temperature": ["Warm", "Warm", "Cold", "Warm"],
    "Humidity": ["Normal", "High", "High", "High"],
    "Wind": ["Strong", "Strong", "Strong", "Weak"],
    "PlayTennis": ["Yes", "Yes", "No", "Yes"]
})

# Save as CSV
data.to_csv("training_data.csv", index=False)

print("CSV file created successfully!\n")

# Read CSV
data = pd.read_csv("training_data.csv")

print("Training Data:\n")
print(data)

# Separate features and target
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

# Find-S Algorithm
h = None

for i in range(len(Y)):
    if Y[i] == "Yes":
        if h is None:
            h = X[i]
        else:
            for j in range(len(h)):
                if h[j] != X[i][j]:
                    h[j] = '?'

print("\nFinal Hypothesis:")
print(h)