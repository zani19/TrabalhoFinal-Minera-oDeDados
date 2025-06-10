import os
import pandas as pd
from data_preprocessing_logreg import carregar_dados_logreg
from model_logreg import treinar_modelo_logreg, avaliar_modelo_logreg
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":
    # Define o diretório base do projeto para construir caminhos relativos
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_csv = os.path.join(base_dir, "data", "data_preprocessed.csv")

    print(f"Carregando dados de: {caminho_csv}")
    X, y = carregar_dados_logreg(caminho_csv)

    print("Dividindo dados em treino e teste...")
    # stratify=y é importante para manter a proporção das classes na divisão
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"Distribuição original das classes no treino: \n{pd.Series(y_train).value_counts(normalize=True)}")

    # Aplicar SMOTE para lidar com desbalanceamento de classes no conjunto de treino
    # Se não quiser usar SMOTE, comente as linhas abaixo e treine com X_train, y_train
    print("Aplicando SMOTE aos dados de treinamento...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"Distribuição das classes no treino após SMOTE: \n{pd.Series(y_train_resampled).value_counts(normalize=True)}")

    print("Treinando o modelo de Regressão Logística...")
    # Usar dados reamostrados para o treino
    model, scaler = treinar_modelo_logreg(X_train_resampled, y_train_resampled)
    # Se não usou SMOTE: model, scaler = treinar_modelo_logreg(X_train, y_train)


    print("Avaliando o modelo...")
    avaliar_modelo_logreg(model, scaler, X_test, y_test)

    print("Processo de Regressão Logística concluído.")