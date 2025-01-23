import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Function to calculate molecular descriptors from SMILES
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "MolecularWeight": Descriptors.MolWt(mol),
            "NumAtoms": Descriptors.HeavyAtomCount(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            "LogP": Descriptors.MolLogP(mol),
        }
    else:
        return {"MolecularWeight": np.nan, "NumAtoms": np.nan, "NumRotatableBonds": np.nan, "LogP": np.nan}

# Load the dataset
file_path = r"C:\Users\Sushma\Desktop\Acadamics\4th sem\BIO-2\project\tox21.csv"
data = pd.read_csv(file_path)

# Apply descriptors calculation
descriptors = data["smiles"].apply(calculate_descriptors)
descriptors_df = pd.DataFrame(descriptors.tolist())

# Combine descriptors with labels
data = pd.concat([descriptors_df, data["label"]], axis=1).dropna()

# Split dataset into features and target
X = data.drop("label", axis=1)
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a model (e.g., SVM)
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Function to predict whether a given SMILES is toxic or non-toxic
def predict_toxicity(smiles):
    # Calculate descriptors for the new SMILES
    descriptors = calculate_descriptors(smiles)
    descriptors_df = pd.DataFrame([descriptors])
    
    # Standardize the new data using the already fitted scaler
    descriptors_scaled = scaler.transform(descriptors_df)
    
    # Predict the class
    prediction = model.predict(descriptors_scaled)
    
    # Return the result
    if prediction == 1:
        return "Toxic"
    else:
        return "Non-Toxic"

# Example usage: Predict toxicity for a new SMILES string
new_smiles = input("Enter SMILES : ")  # Replace with your own SMILES
result = predict_toxicity(new_smiles)
print(f"The SMILES string '{new_smiles}' is predicted to be: {result}")