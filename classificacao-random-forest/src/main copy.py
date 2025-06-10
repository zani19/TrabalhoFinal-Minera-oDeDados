import os
from data_preprocessing import carregar_dados
from model import treinar_modelo, avaliar_modelo
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_csv = os.path.join(base_dir, "data", "data_preprocessed.csv")
    X, y = carregar_dados(caminho_csv)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = treinar_modelo(X_train, y_train)
    avaliar_modelo(clf, X_test, y_test)