import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Load dataset
file_path = r"C:\Users\Sushma\Desktop\Acadamics\4th sem\BIO-2\project\tox21.csv"
data = pd.read_csv(file_path)

# Preprocessing SMILES strings to molecular descriptors
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

# Define models
models = {
    "SVM": SVC(kernel="rbf", probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"\nModel: {name}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Toxic", "Toxic"]))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    if y_proba is not None:
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.2f}")