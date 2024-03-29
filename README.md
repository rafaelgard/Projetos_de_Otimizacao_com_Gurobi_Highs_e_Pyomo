# Projetos de Otimização com Gurobi, Highs e Pyomo
![Alt text](images/cover.jpg)
Este repositório contém implementações de 4 problemas de otimização, todos desenvolvidos utilizando as bibliotecas Gurobipy, Pyomo, Highspy e o solver GLPK. As soluções oferecem abordagens otimizadas para desafios relacionados à roteirização de embarcações, alocação de combustíveis em navios e roteirização de drones para vigilância de áreas de interesse.

## Questões e Artigo

1. **Problema de Roterização de Embarcações com Múltiplas Capacidades e Minimização de Custos**
   - Solução para a roteirização de embarcações considerando múltiplas capacidades e minimização de custos de treinamento e transporte.

![Alt text](images/Questão_1.png)

2. **Problema de Roteirização de Embarcações com Múltiplas Capacidades e Programação de Embarques**
   - Abordagem para o problema de roteirização de embarcações com múltiplas capacidades e programação eficiente de embarques.

![Alt text](images/Questão_2.png)

3. **Problema de Alocação de Combustíveis em Navios com Múltiplas Capacidades em Terminais Portuários**
   - Solução otimizada para a alocação de combustíveis de diversos tipos em navios, considerando múltiplas capacidades em terminais portuários.
   
![Alt text](images/Questão_3.png)

4. **Artigo: "Drones Aerial Monitoring Problem"**
   - Baseado no artigo "Drones Aerial Monitoring Problem" (DOI: [10.1016/j.cor.2019.01.001](https://doi.org/10.1016/j.cor.2019.01.001)), este problema foca na roteirização de drones para vigilância de áreas estratégicas.

## Funções Úteis

Além das implementações específicas, este repositório também disponibiliza algumas funções úteis que podem ser utilizadas em conjunto com as bibliotecas Gurobipy, Highspy e Pyomo.

- Imprimir e salvar resultados de um modelo criado na biblioteca Gurobipy
- Otimizar um modelo criado no pyomo pelo solver Highs podendo acompanhar a execução do processo de otimização, assim como a otimização de hiperparâmetros do otimizador.

## Pacotes Requeridos
- Python 3
- Pandas
- NumPy
- Gurobipy
- Highspy
- Time
- Random

## Outros projetos de otimização:
Também mantenho um repositório onde você encontrará outro projeto que teve como objetivo criar uma aplicação simples e visual da meta-heurística Simulated Annealing no contexto de processamento de imagens. Embora envolva problemas e abordagens distintas dos apresentados neste repositório, pode ser interessante explorar caso possua interesse em meta-heurísticas. 

- [Simulated Annealing aplicado no Processamento de Imagens](https://github.com/rafaelgard/Simulated-annealing)
