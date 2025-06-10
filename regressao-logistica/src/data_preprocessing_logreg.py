import pandas as pd

def carregar_dados_logreg(caminho_csv):
    """
    Carrega os dados do CSV, remove colunas textuais desnecessárias
    e separa as features (X) do target (y).
    """
    df = pd.read_csv(caminho_csv)
    # Remove a coluna 'Target' textual se existir, pois usaremos 'Target_encoded'
    if 'Target' in df.columns:
        df = df.drop('Target', axis=1)
    
    # Verifica se a coluna 'Target_encoded' existe
    if 'Target_encoded' not in df.columns:
        raise ValueError("A coluna 'Target_encoded' não foi encontrada no CSV. Verifique o arquivo de dados.")

    X = df.drop('Target_encoded', axis=1)
    y = df['Target_encoded']
    return X, y