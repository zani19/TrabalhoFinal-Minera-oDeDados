import matplotlib.pyplot as plt
import seaborn as sns

def plotar_matriz_confusao_logreg(cm, labels):
    """
    Plota a matriz de confusão.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de Confusão - Regressão Logística")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()