import pandas as pd
import numpy as np
from scipy import stats

# Lê o arquivo com os resultados de múltiplas execuções
df = pd.read_csv("summary_runs.csv")

# Remove a coluna 'run' se existir
if 'run' in df.columns:
    df = df.drop(columns=['run'])

# Inicializa lista de resultados
resultados = []

# Número de execuções
n = len(df)

# Para cada métrica, calcula as estatísticas
for coluna in df.columns:
    valores = df[coluna].dropna()
    media = valores.mean()
    desvio = valores.std(ddof=1)  # ddof=1 para amostra
    erro_padrao = desvio / np.sqrt(n)

    # Intervalo de confiança 95%
    t_critico = stats.t.ppf(0.975, df=n - 1)
    margem_erro = t_critico * erro_padrao
    ic_inferior = media - margem_erro
    ic_superior = media + margem_erro

    resultados.append({
        "Métrica": coluna,
        "Média": round(media, 4),
        "Desvio Padrão": round(desvio, 4),
        "IC 95% Inferior": round(ic_inferior, 4),
        "IC 95% Superior": round(ic_superior, 4)
    })

# Salva no CSV
df_resultado = pd.DataFrame(resultados)
df_resultado.to_csv("summary_statistics.csv", index=False)
print("✅ Estatísticas salvas em: summary_statistics.csv")
