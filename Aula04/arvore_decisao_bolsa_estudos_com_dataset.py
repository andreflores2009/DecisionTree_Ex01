# Importa a biblioteca pandas para manipulação de dados em DataFrames
import pandas as pd  # pandas permite ler CSV, filtrar colunas e criar novas colunas

# Importa a árvore de decisão do scikit-learn para classificação
from sklearn.tree import DecisionTreeClassifier  # modelo supervisionado 

# Importa funções de visualização de árvore e relatório textual 
from sklearn.tree import plot_tree, export_text  # utilitário para visualizar e extrair regras em texto

# Importa matplotlib para exibir o gráfico da árvore (opcional)
import matplotlib.pyplot as plt  # biblioteca de plots 2D

# ---------------------------
# 1) CARREGAR O DATASET DE TREINO (ROTULADO)
# ---------------------------

# Lê o CSV de treino com separador ';' e vírgula como separador decimal
treino = pd.read_csv("/content/drive/MyDrive/COLABS/SISTEMAS_APOIO_E_DECISAO/bolsas_treinamento_1000.csv", sep=";", decimal=",")  # contém 'desempenho', 'renda' e 'bolsa'

# Separa as features (X) e o alvo (y) a partir do DataFrame de treino
X_train = treino[["desempenho", "renda"]]  # duas colunas numéricas usadas como entradas do modelo
y_train = treino["bolsa"]                  # coluna de saída (classe), calculada pela regra oficial (política)

# ---------------------------
# 2) TREINAR A ÁRVORE DE DECISÃO
# ---------------------------

# Cria o classificador de árvore de decisão, definindo critério, profundidade e semente para reprodutibilidade
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)  # limita complexidade e fixa random_state
'''
max_depth=3
É o limite de profundidade (quantos “SE…ENTÃO…” seguidos a árvore pode fazer no pior caminho).
Aumentar (ex.: 5) → permite mais cortes → folhas menores → tende a acertar mais no treino (até 100%), mas arranca ruído/outliers e pode overfittar (piorar fora do treino).
Diminuir (ex.: 2) → menos cortes → árvore mais simples → pode não separar todas as faixas (ex.: não distinguir bem 50% vs 30% ou 90% vs 70%).
random_state=42
É a semente de aleatoriedade do estimador, controla desempates.
'''

# Ajusta (treina) o modelo com os dados de treino
clf.fit(X_train, y_train)  # o modelo aprende a separar as classes seguindo divisões que maximizam informação

# ---------------------------
# 3) CARREGAR O DATASET DE ENTRADA (50 LINHAS, SEM RÓTULO)
# ---------------------------

# Lê o CSV de entrada com as mesmas configurações de separador e decimal
entrada = pd.read_csv("/content/drive/MyDrive/COLABS/SISTEMAS_APOIO_E_DECISAO/bolsas_entrada_50.csv", sep=";", decimal=",")  # contém 'desempenho' e 'renda' apenas

# ---------------------------
# 4) DEFINIR A REGRA OFICIAL (VERDADE-TERRENO) E ROTULAR A ENTRADA
# ---------------------------

# Define a função que implementa exatamente a política do enunciado (regra oficial da bolsa)
def regra_bolsa(d, r):  # recebe desempenho (d) e renda por pessoa (r)
    if d < 5.0:  # se desempenho for menor que 5.0
        return 0  # não contempla bolsa (0%)
    if d <= 8.0:  # se desempenho entre 5.0 e 8.0 (inclusive)
        return 50 if r <= 1.5 else 30  # 50% se renda <= 1.5; senão 30%
    return 90 if r <= 1.5 else 70  # se desempenho > 8.0: 90% se renda <= 1.5; senão 70%

# Cria a coluna 'bolsa_verdade' aplicando a regra oficial linha a linha na entrada
entrada["bolsa_verdade"] = [regra_bolsa(d, r) for d, r in zip(entrada["desempenho"], entrada["renda"])]  # rótulo real

# ---------------------------
# 5) REALIZAR A PREDIÇÃO DO MODELO E COMPARAR
# ---------------------------

# Usa o modelo treinado para prever a bolsa com base em 'desempenho' e 'renda' da entrada
entrada["bolsa_prevista"] = clf.predict(entrada[["desempenho", "renda"]])  # prediz a classe aprendida pela árvore

# Cria a coluna 'resultado' marcando ACERTOU quando previsão = verdade e ERROU caso contrário
entrada["resultado"] = ["ACERTOU" if p == v else "ERROU" for p, v in zip(entrada["bolsa_prevista"], entrada["bolsa_verdade"])]  # comparação
'''
zip(a, b) é uma função nativa do Python que percorre duas (ou mais) sequências em paralelo, gerando pares.
Ex.: list(zip([1,2,3],[10,20,30])) → [(1,10), (2,20), (3,30)].
lê cada par (p, v); se forem iguais, coloca "ACERTOU", senão "ERROU".
'''

# Calcula a acurácia simples no conjunto de entrada (proporção de acertos)
acuracia = (entrada["bolsa_prevista"] == entrada["bolsa_verdade"]).mean()  # valor entre 0 e 1

# ---------------------------
# 6) IMPRIMIR AS 50 LINHAS COM O MARCADOR DE ACERTO/ERRO
# ---------------------------

# Seleciona as colunas de interesse para exibição
colunas_exibir = ["desempenho", "renda", "bolsa_verdade", "bolsa_prevista", "resultado"]  # ordem de visualização
#cria uma lista de strings. Ela não “cria colunas”; só guarda quais colunas você quer ver e em qual ordem.

# Imprime o cabeçalho informando acurácia no conjunto de entrada
print(f"Acurácia no conjunto de entrada (50 linhas): {acuracia:.3f}")  # mostra métrica agregada

# Imprime a tabela completa (as 50 linhas), já que o arquivo de entrada tem exatamente 50
print(entrada[colunas_exibir].to_string(index=False))  # exibe sem índice para ficar mais limpo
'''
diz ao pandas: “me devolva um DataFrame contendo essas colunas, nessa ordem”. Isso funciona porque o DataFrame entrada já tem essas colunas 
(bolsa_verdade, bolsa_prevista, resultado). Se algum nome não existir, dá KeyError.
to_string(index=False) #só formata a saída em texto, sem a coluna de índice, para ficar limpo no console.
'''

# ---------------------------
# 7) (OPCIONAL) VISUALIZAR A ÁRVORE E AS REGRAS TEXUAIS PARA SLIDES
# ---------------------------

# Cria uma figura para o gráfico da árvore de decisão
plt.figure(figsize=(10, 6))  # define tamanho da figura para facilitar leitura

# Desenha a árvore com nomes das features e classes, incluindo caixas coloridas
plot_tree(
    clf,  # modelo treinado
    feature_names=["desempenho", "renda"],  # nomes das entradas
    class_names=[f"{c}%" for c in clf.classes_],  # rótulos de classe com símbolo de porcentagem
    filled=True,  # preenche os nós com cor indicando pureza
    rounded=True,  # deixa as caixas com cantos arredondados
    precision=2    # número de casas exibidas nas condições
)  # finaliza chamada do plot_tree

# Adiciona um título ao gráfico
plt.title("Árvore de Decisão – Cálculo de Bolsa de Estudo (Treinada no conjunto de 1000 linhas)")  # título do gráfico

# Ajusta layout para não cortar textos
plt.tight_layout()  # melhora o espaçamento automaticamente

# Exibe o gráfico na tela
plt.show()  # mostra a figura renderizada

# Gera uma representação textual das regras aprendidas (útil para colar nos slides)
print("\nRegras aproximadas aprendidas pela árvore:\n")  # texto introdutório
print(export_text(clf, feature_names=["desempenho", "renda"]))  # imprime as regras em formato de árvore textual
