import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Lê o CSV com as estatísticas
df = pd.read_csv("summary_statistics.csv")

# Define quais métricas deseja plotar (ordem opcional)
metricas_desejadas = ["pP_avg", "pR_avg", "pF1_avg", "ACP_avg", "AAP_avg", "K_avg"]
df = df[df["Métrica"].isin(metricas_desejadas)].copy()

# Ordena conforme ordem desejada
df["ordem"] = df["Métrica"].apply(lambda x: metricas_desejadas.index(x))
df = df.sort_values("ordem")

# Extrai dados
metricas = df["Métrica"]
medias = df["Média"]
desvios = df["Desvio Padrão"]
ic_inferior = df["IC 95% Inferior"]
ic_superior = df["IC 95% Superior"]

# Posições das barras
x = np.arange(len(metricas))

# Tamanho da figura
plt.figure(figsize=(10, 6))

# Barras com desvio padrão
plt.bar(x, medias, yerr=desvios, capsize=8, color="#4c72b0", alpha=0.9, label="Média ± DP")

# Linhas tracejadas para IC 95%
for i in range(len(x)):
    plt.plot([x[i] - 0.2, x[i] + 0.2], [ic_inferior.iloc[i]] * 2, color="red", linestyle="--", linewidth=1)
    plt.plot([x[i] - 0.2, x[i] + 0.2], [ic_superior.iloc[i]] * 2, color="red", linestyle="--", linewidth=1)

# Rótulos
plt.xticks(x, metricas, fontsize=12)
plt.ylabel("Valor", fontsize=13)
plt.title("Estatísticas das Métricas - AMiner-12 (20 Execuções)", fontsize=14)
plt.ylim(min(ic_inferior) - 0.01, max(ic_superior) + 0.01)
plt.legend()
plt.grid(axis='y', linestyle="--", alpha=0.6)

# Salvar e exibir
plt.tight_layout()
plt.savefig("grafico_metricas_aminer12_com_pp_pr.png", dpi=300)
plt.show()
