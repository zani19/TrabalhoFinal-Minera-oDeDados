import os
import pandas as pd 
from data_preprocessing import carregar_dados
from model import treinar_modelo, avaliar_modelo
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE # Importar SMOTE

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_csv = os.path.join(base_dir, "data", "data_preprocessed.csv")
    X, y = carregar_dados(caminho_csv)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # Adicionar stratify=y

    # Aplicar SMOTE aos dados de treinamento
    print(f"Distribuição original das classes no treino: {pd.Series(y_train).value_counts(normalize=True)}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Distribuição das classes no treino após SMOTE: {pd.Series(y_train_resampled).value_counts(normalize=True)}")

    # Treinar com os dados reamostrados
    clf = treinar_modelo(X_train_resampled, y_train_resampled)
    # Avaliar no conjunto de teste original (não reamostrado)
    avaliar_modelo(clf, X_test, y_test)