from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils import plotar_matriz_confusao
import pandas as pd
import matplotlib.pyplot as plt

def treinar_modelo(X_train, y_train):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def avaliar_modelo(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusão:")
    print(cm)
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))

    # Exibe matriz de confusão como gráfico
    plotar_matriz_confusao(cm, labels=['Dropout', 'Enrolled', 'Graduate'])

    # Importância das features
    importancias = clf.feature_importances_
    nomes = X_test.columns
    importancias_df = pd.DataFrame({'feature': nomes, 'importancia': importancias})
    importancias_df = importancias_df.sort_values('importancia', ascending=False)
    print("\nTop 10 variáveis mais importantes:")
    print(importancias_df.head(10))

    # Gráfico das 10 variáveis mais importantes
    importancias_df.head(10).plot.bar(x='feature', y='importancia', legend=False)
    plt.title("Top 10 variáveis mais importantes")
    plt.tight_layout()
    plt.show()

    # Histograma de algumas variáveis relevantes
    variaveis_hist = importancias_df.head(3)['feature'].tolist()
    for var in variaveis_hist:
        X_test[var].plot.hist(bins=30, alpha=0.7)
        plt.title(f'Histograma da variável: {var}')
        plt.xlabel(var)
        plt.ylabel('Frequência')
        plt.tight_layout()
        plt.show()

    # Distribuição das classes no conjunto de teste
    pd.Series(y_test).value_counts().sort_index().plot.pie(
        labels=['Dropout', 'Enrolled', 'Graduate'],
        autopct='%1.1f%%',
        startangle=90,
        legend=False
    )
    plt.title('Distribuição das classes no conjunto de teste')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()