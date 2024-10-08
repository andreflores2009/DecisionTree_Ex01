'''
Faça a especificação do processo descrito abaixo, usando a Árvore de Decisão. 
Processo: Calcular valor de bolsa de estudo
Uma universidade particular concede bolsas de estudos levando em consideração o desempenho dos candidatos em uma prova 
específica e o nível de carência. Candidatos com desempenho na prova abaixo de 5,0 não são contemplados com bolsa. 
Alunos com desempenho na prova entre 5,0 e 8,0 são contemplados com 50 % de bolsa se a renda familiar for de até 1,5 
salários mínimos por pessoa ou com 30% de bolsa se a renda familiar for maior que 1,5 salários mínimos por pessoa.
 No caso do desempenho na prova ser acima de 8,0, as bolsas são de 70% para candidatos com renda por pessoa acima de 1,5 
 salários mínimos ou 90% se a renda for de até 1,5 salários mínimos por pessoa.
 '''

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Função para calcular e preencher a bolsa
def calcular_bolsa(desempenho, renda):
    bolsa = [0] * len(desempenho)  # Inicializa a bolsa com zero para todos os candidatos
    
    # Algoritmo para calcular a bolsa com base nas regras fornecidas
    for i in range(len(desempenho)):
        if desempenho[i] < 5.0:
            bolsa[i] = 0
        elif 5.0 <= desempenho[i] <= 8.0:
            if renda[i] <= 1.5:
                bolsa[i] = 50
            else:
                bolsa[i] = 30
        elif desempenho[i] > 8.0:
            if renda[i] <= 1.5:
                bolsa[i] = 90
            else:
                bolsa[i] = 70

    return bolsa

# Dados fixos fornecidos no código
desempenho = [4.0, 5.5, 7.0, 8.5, 9.0, 9.2]  # Adiciona um valor extra para representar a bolsa de 70%
renda = [2.0, 1.2, 2.5, 1.0, 1.5, 2.0]      # Adiciona um valor extra para a nova entrada

# Calculando a bolsa com o método
bolsa = calcular_bolsa(desempenho, renda)

# Criando um DataFrame com os dados fornecidos
data = {
    'Desempenho': desempenho,
    'Renda': renda,
    'Bolsa': bolsa
}

df = pd.DataFrame(data)

# Exibindo os dados antes de gerar a árvore
print("Dados utilizados para gerar a árvore de decisão:")
print(df)

# Verificando se todos os valores de bolsa são únicos e estão representados
unique_bolsas = df['Bolsa'].unique()
print("\nValores únicos de bolsa encontrados:")
print(unique_bolsas)

# Separando as variáveis independentes (X) e a variável dependente (y)
X = df[['Desempenho', 'Renda']]  # X contém Desempenho e Renda (features)
y = df['Bolsa']                  # y contém a Bolsa (valor a ser previsto)

# Treinando a árvore de decisão com todos os dados disponíveis
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)  # Utilizando entropia como critério de divisão
clf = clf.fit(X, y)  # Treinando o modelo com as variáveis X e y

# Plotando a árvore de decisão
plt.figure(figsize=(12, 8))

# plot_tree gera a árvore de decisão. As features são Desempenho e Renda, 
# e as classes são os valores de Bolsa (0%, 30%, 50%, 70%, 90%)
plot_tree(clf, feature_names=['Desempenho', 'Renda'], class_names=['0%', '30%', '50%', '70%', '90%'], filled=True)

plt.title('Árvore de Decisão para Cálculo de Bolsa de Estudo')
plt.show()  # Exibindo a árvore de decisão


'''  NOVAS PREVISOES'''
# Novos dados para previsão
novos_dados = pd.DataFrame({
    'Desempenho': [6.0, 7.5, 8.2],
    'Renda': [1.4, 1.8, 1.2]
})

# Fazendo previsões
previsoes = clf.predict(novos_dados)
print("Previsões para os novos dados de bolsa de estudo:")
print(previsoes)

# Adicionando uma coluna para mostrar as previsões como porcentagens
novos_dados['Bolsa Prevista'] = previsoes
print("\nNovos dados com bolsas previstas:")
print(novos_dados)




# Explicações sobre os termos na árvore de decisão:

"""
Explicação sobre os termos da árvore de decisão:

1. `samples`:
   - O que significa: Indica a quantidade de amostras (linhas de dados) que chegaram até aquele nó na árvore.
   - Exemplo: Se em um nó específico houver `samples = 3`, significa que três exemplos/candidatos estão representados nesse nó.

2. `value`:
   - O que significa: `value` é uma lista que mostra a distribuição das amostras entre as diferentes classes naquele nó.
   - Exemplo: Se `value = [2, 1]`, significa que naquele nó existem 2 amostras da primeira classe e 1 da segunda classe. 
     Se temos 5 classes (0%, 30%, 50%, 70%, 90%), o array pode ser algo como `[1, 0, 1, 0, 3]`, indicando quantas amostras pertencem a cada classe.
   
3. `entropy`:
   - O que significa: Entropia é uma medida de impureza ou incerteza naquele nó. Quanto maior a entropia, mais misturadas estão as classes no nó.
   - Cálculo: A entropia é calculada com base na probabilidade de cada classe naquele nó. Se todos os exemplos em um nó pertencem à mesma classe, a entropia é 0 (nó puro).
   - Exemplo: Se `entropy = 0.0`, significa que todos os exemplos naquele nó pertencem à mesma classe. Se `entropy = 1.0`, as classes estão misturadas igualmente.
   
4. `class`:
   - O que significa: A classe é a previsão feita para aquele nó com base na maioria das amostras. Ou seja, a classe corresponde ao valor de bolsa (0%, 30%, 50%, 70%, 90%) mais comum entre as amostras do nó.
   - Exemplo: Se `class = 50%`, isso significa que a maioria das amostras naquele nó corresponde à classe de bolsa de 50%.

Processo de Geração da Árvore:

1. Divisão dos Dados:
   - A árvore de decisão começa no topo (nó raiz) com todos os exemplos. Ela tenta dividir os dados em grupos mais puros com base nas variáveis independentes (neste caso, `Desempenho` e `Renda`).
   
2. Critério de Divisão (Entropia):
   - A divisão dos nós é feita para minimizar a entropia, ou seja, para maximizar a pureza dos nós resultantes. Cada nó tenta agrupar amostras de uma mesma classe.
   
3. Profundidade da Árvore (`max_depth=3`):
   - A profundidade da árvore foi limitada a 3 níveis para evitar que ela se torne muito complexa. A profundidade determina o número de vezes que a árvore pode dividir os dados.
   
4. Previsão:
   - No final de cada caminho na árvore, a classe (o valor de bolsa previsto) é escolhida com base na maioria das amostras que chegaram até aquele nó.
"""
