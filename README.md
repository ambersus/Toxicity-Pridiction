# **Tox21 Toxicity Prediction**

## **Overview**
This project focuses on predicting the toxicity of chemical compounds using machine learning techniques. The dataset used for this study is the **Tox21 dataset**, a widely recognized resource in **computational toxicology**. The dataset provides molecular data in the form of **SMILES (Simplified Molecular Input Line Entry System) strings**, representing the chemical structures of various molecules.

The primary goal of this project is to analyze the relationship between molecular structures and their toxicity levels. The dataset consists of binary labels (**toxic** or **non-toxic**), allowing classification models to learn from molecular descriptors and predict toxicity outcomes.

## **Dataset**
The **Tox21 dataset** includes:
- **SMILES Strings**: Encoded chemical structures of compounds.
- **Binary Toxicity Labels**: Each molecule is classified as either **toxic** or **non-toxic** based on its biological impact.
- **Molecular Descriptors**: Extracted using **RDKit**, including:
  - Molecular Weight
  - Number of Atoms
  - Number of Rotatable Bonds
  - LogP (Partition Coefficient)

## **Project Files**
This repository includes the following key files:

### **1. `tox21.csv`**
- Contains the dataset with SMILES strings and their corresponding toxicity labels.
- Used as the primary input for the machine learning models.

### **2. `toxicity_prediction.py`**
- Loads the dataset and extracts molecular descriptors using **RDKit**.
- Preprocesses the data by handling missing values and standardizing features.
- Splits the dataset into training and testing sets.
- Implements an **SVM (Support Vector Machine)** model to classify molecules as toxic or non-toxic.
- Includes a function to predict toxicity for a given **SMILES string**.

### **3. `tox21_model_evaluation.py`**
- Implements multiple machine learning models (**SVM, KNN, and RandomForest**).
- Trains and evaluates models using accuracy, classification reports, and ROC-AUC scores.
- Helps in comparing different algorithms to determine the most effective one for toxicity prediction.

### **Running the Toxicity Prediction Model**
To train the model and predict toxicity for a new SMILES string:

```bash
python toxicity_prediction.py
```

Enter a **SMILES** string when prompted to get the toxicity prediction.

### **Running Model Evaluation**
To compare different machine learning models:

```bash
python tox21_model_evaluation.py
```

## **Applications**
- Predicting the toxicity of unknown chemical compounds.
- Identifying hazardous substances in environmental research.
- Enhancing computational toxicology through machine learning models.
