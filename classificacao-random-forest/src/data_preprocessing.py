import pandas as pd

def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)
    # Remove colunas não utilizadas (exemplo: Target textual)
    if 'Target' in df.columns:
        df = df.drop('Target', axis=1)
    # Separa features e target
    X = df.drop('Target_encoded', axis=1)
    y = df['Target_encoded']
    return X, y