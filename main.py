import random
import numpy as np


class RedeEnergia:
    def __init__(self, num_subestacoes, num_consumidores):
        self.num_subestacoes = num_subestacoes
        self.num_consumidores = num_consumidores

        # capacidades das subestações (mw)
        self.capacidades = np.random.uniform(50, 200, num_subestacoes)

        # demandas dos consumidores (mw)
        self.demandas = np.random.uniform(5, 30, num_consumidores)

        # distâncias entre subestações e consumidores (km)
        self.distancias = np.random.uniform(1, 50, (num_subestacoes, num_consumidores))

        # custos de transmissão por km ($/mw/km)
        self.custo_transmissao = 100


class AlgoritmoGenetico:
    def __init__(self, rede, tamanho_populacao=50, geracoes=100, taxa_mutacao=0.01):
        self.rede = rede
        self.tamanho_populacao = tamanho_populacao
        self.geracoes = geracoes
        self.taxa_mutacao = taxa_mutacao
        self.tamanho_cromossomo = rede.num_subestacoes * rede.num_consumidores

    def criar_individuo(self):
        """Cria um indivíduo representando as conexões entre subestações e consumidores"""
        # cada consumidor é conectado a exatamente uma subestação
        individuo = []
        for _ in range(self.rede.num_consumidores):
            conexoes = [0] * self.rede.num_subestacoes
            conexoes[random.randint(0, self.rede.num_subestacoes - 1)] = 1
            individuo.extend(conexoes)
        return individuo

    def inicializar_populacao(self):
        """Inicializa a população com indivíduos válidos"""
        return [self.criar_individuo() for _ in range(self.tamanho_populacao)]

    def decodificar_solucao(self, individuo):
        """Converte o cromossomo em uma matriz de conexões"""
        matriz = np.array(individuo).reshape(
            self.rede.num_consumidores, self.rede.num_subestacoes
        )
        return matriz

    def calcular_fitness(self, individuo):
        """Avalia a qualidade da solução considerando restrições e custos"""
        matriz = self.decodificar_solucao(individuo)
        custo_total = 0
        penalidade = 0

        # calcula carga em cada subestação
        cargas_subestacoes = np.zeros(self.rede.num_subestacoes)
        for i in range(self.rede.num_consumidores):
            for j in range(self.rede.num_subestacoes):
                if matriz[i][j] == 1:
                    # adiciona demanda à subestação
                    cargas_subestacoes[j] += self.rede.demandas[i]
                    # calcula custo de transmissão
                    custo_total += (
                        self.rede.demandas[i]
                        * self.rede.distancias[j][i]
                        * self.rede.custo_transmissao
                    )

        # penaliza violações de capacidade
        for j in range(self.rede.num_subestacoes):
            if cargas_subestacoes[j] > self.rede.capacidades[j]:
                penalidade += 1000000 * (
                    cargas_subestacoes[j] - self.rede.capacidades[j]
                )

        return 1 / (
            custo_total + penalidade + 1
        )  # quanto menor o custo, maior o fitness

    def selecionar_pais(self, populacao, fitness):
        """Seleciona pais usando torneio"""
        tamanho_torneio = 3
        indices_populacao = range(len(populacao))

        def torneio():
            competidores = random.sample(list(indices_populacao), tamanho_torneio)
            return populacao[max(competidores, key=lambda i: fitness[i])]

        return torneio(), torneio()

    def crossover(self, pai1, pai2):
        """Realiza crossover preservando a validade das soluções"""
        ponto_corte = (
            random.randint(1, self.rede.num_consumidores - 1)
            * self.rede.num_subestacoes
        )

        filho1 = pai1[:ponto_corte] + pai2[ponto_corte:]
        filho2 = pai2[:ponto_corte] + pai1[ponto_corte:]

        return filho1, filho2

    def mutar(self, individuo):
        """Realiza mutação preservando a validade da solução"""
        for i in range(0, len(individuo), self.rede.num_subestacoes):
            if random.random() < self.taxa_mutacao:
                # zera todas as conexões do consumidor
                for j in range(self.rede.num_subestacoes):
                    individuo[i + j] = 0
                # conecta a uma nova subestação aleatória
                nova_conexao = random.randint(0, self.rede.num_subestacoes - 1)
                individuo[i + nova_conexao] = 1
        return individuo

    def executar(self):
        """Executa o algoritmo genético"""
        populacao = self.inicializar_populacao()
        melhor_historico = []

        for geracao in range(self.geracoes):
            fitness = [self.calcular_fitness(ind) for ind in populacao]

            melhor_ind = populacao[fitness.index(max(fitness))]
            melhor_historico.append(max(fitness))

            if geracao % 10 == 0:
                print(f"geracao {geracao}: melhor fitness = {max(fitness):.6f}")

            nova_populacao = []
            while len(nova_populacao) < self.tamanho_populacao:
                pai1, pai2 = self.selecionar_pais(populacao, fitness)
                filho1, filho2 = self.crossover(pai1, pai2)

                nova_populacao.append(self.mutar(filho1))
                if len(nova_populacao) < self.tamanho_populacao:
                    nova_populacao.append(self.mutar(filho2))

            populacao = nova_populacao

        # retorna a melhor solução encontrada
        fitness_final = [self.calcular_fitness(ind) for ind in populacao]
        melhor_solucao = populacao[fitness_final.index(max(fitness_final))]
        return melhor_solucao, melhor_historico


if __name__ == "__main__":
    # criar uma rede de exemplo com 5 subestações e 20 consumidores
    rede = RedeEnergia(num_subestacoes=5, num_consumidores=20)

    # configurar e executar o algoritmo genético
    ag = AlgoritmoGenetico(rede, tamanho_populacao=50, geracoes=100, taxa_mutacao=0.01)
    melhor_solucao, historico = ag.executar()

    # analisar a solução
    matriz_solucao = ag.decodificar_solucao(melhor_solucao)
    print("\nmelhor solucao encontrada:")
    print("matriz de conexoes (linhas=consumidores, colunas=subestacoes):")
    print(matriz_solucao)

    # calcular estatísticas da solução
    cargas = np.zeros(rede.num_subestacoes)
    custo_total = 0

    for i in range(rede.num_consumidores):
        for j in range(rede.num_subestacoes):
            if matriz_solucao[i][j] == 1:
                cargas[j] += rede.demandas[i]
                custo_total += (
                    rede.demandas[i] * rede.distancias[j][i] * rede.custo_transmissao
                )

    print("\nestatisticas da solucao:")
    print(f"custo total: ${custo_total:,.2f}")
    print("\ncargas nas subestacoes:")
    for i, (carga, capacidade) in enumerate(zip(cargas, rede.capacidades)):
        print(
            f"> subestacao {i}: {carga:.2f}MW / {capacidade:.2f}MW ({(carga/capacidade)*100:.1f}% utilizacao)"
        )

