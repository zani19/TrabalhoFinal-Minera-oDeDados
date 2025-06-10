from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler # Para o caso de precisar no futuro
from utils_logreg import plotar_matriz_confusao_logreg
import pandas as pd
import matplotlib.pyplot as plt

def treinar_modelo_logreg(X_train, y_train):
    """
    Treina um modelo de Regressão Logística.
    """
    # Padronização dos dados (importante para Regressão Logística)
    # Embora seus dados já pareçam padronizados, é uma boa prática.
    # Se você tem certeza que já estão padronizados, pode comentar/remover.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Aumentar max_iter se houver aviso de convergência
    # class_weight='balanced' para lidar com desbalanceamento
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', solver='liblinear')
    model.fit(X_train_scaled, y_train)
    return model, scaler # Retorna o scaler para usar no conjunto de teste

def avaliar_modelo_logreg(model, scaler, X_test, y_test):
    """
    Avalia o modelo de Regressão Logística no conjunto de teste.
    """
    X_test_scaled = scaler.transform(X_test) # Aplica a mesma transformação do treino
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusão (Regressão Logística):")
    print(cm)
    print("\nRelatório de classificação (Regressão Logística):")
    print(classification_report(y_test, y_pred))

    plotar_matriz_confusao_logreg(cm, labels=['Dropout', 'Enrolled', 'Graduate'])

    # Coeficientes (importância das features para Regressão Logística)
    # Para modelos multi-classe com 'ovr', model.coef_ terá forma (n_classes, n_features)
    # Para 'multinomial', também. Vamos pegar para a primeira classe como exemplo ou somar.
    if hasattr(model, "coef_"):
        try:
            # Para o caso multinomial ou OVR, pegamos a magnitude média dos coeficientes por feature
            if model.coef_.shape[0] > 1:
                 importances = pd.DataFrame(abs(model.coef_).mean(axis=0), index=X_test.columns, columns=['Importancia'])
            else: # Caso binário ou coef_ seja (1, n_features)
                importances = pd.DataFrame(model.coef_[0], index=X_test.columns, columns=['Importancia'])
            
            importances = importances.reindex(importances['Importancia'].abs().sort_values(ascending=False).index) # Ordena pela magnitude
            
            print("\nImportância das variáveis (Regressão Logística - Magnitude dos Coeficientes):")
            print(importances.head(10))

            importances.head(10).plot(kind='barh', legend=False) # Gráfico de barras horizontais
            plt.title("Top 10 Variáveis Mais Importantes (Reg. Logística)")
            plt.gca().invert_yaxis() # Inverte o eixo y para a mais importante no topo
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Não foi possível plotar a importância das features: {e}")
