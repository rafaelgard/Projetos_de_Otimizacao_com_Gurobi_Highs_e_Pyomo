import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.contrib.appsi.solvers.highs import Highs

'''################## Define os Parâmetros utilizados no modelo ##################'''
Data_entrega = np.array(
    [[0, 0, 0, 10, 15],
     [0, 0, 0, 4, 5],
     [0, 0, 0, 0, 18],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]])

Tempo_viagem = np.array(
    [[0, 0, 0, 3, 4],
     [0, 0, 0, 3, 2],
     [0, 0, 0, 3, 5],
     [2, 2, 2, 0, 0],
     [3, 1, 4, 0, 0]])

'''Cria o modelo'''
model = ConcreteModel()

'''################## Define as variáveis utilizados no modelo ##################'''
quantidade_de_navios = 10
navios = range(0, quantidade_de_navios)
origens_j = range(0, 5)
dentinos_k = range(0, 5)
M = quantidade_de_navios*5*5

'''Define a rota do navio. Onde o indice i e j repretam os pares de coordendas (i,j) e k representam a 
quantidade máxima de navios que se tem disponível'''
model.Rota_navio = Var(navios,
                       origens_j,
                       dentinos_k,
                       within=Binary,
                       bounds=(0, 1))

'''Tempo inicial em que um navio i realiza uma atividade na coordenada (j,k)'''
model.T_inicial = Var(origens_j,
                      within=NonNegativeIntegers,
                      bounds=(0, None))

'''Tempo final em que um navio i finaliza uma atividade na coordenada (j,k)'''
model.T_final = Var(navios,
                    origens_j,
                    dentinos_k,
                    within=NonNegativeIntegers,
                    bounds=(0, None))


'''Tempo de deslocamento de cada navio k para realizar uma atividade na coordenada (i,j)'''
model.Tempo_Deslocamento = Var(navios,
                               origens_j,
                               dentinos_k,
                               within=NonNegativeIntegers,
                               bounds=(0, None))

'''Variável binária que representa se um navio foi utilzado para realizar uma atividade na coordenada (i,j)'''
model.k_navio = Var(navios,
                    within=Binary,
                    bounds=(0, 1))

'''Variável binária que representa se uma rota (j,k) foi atendida por algum navio i'''
model.soma_rotas = Var(origens_j,
                       dentinos_k,
                       within=Binary,
                       bounds=(0, 1))

'''Variável que identifica a quantidade de navios utilizados'''
model.total_de_navios_utilizados = Var(within=NonNegativeIntegers, 
                                       bounds=(0, quantidade_de_navios))

'''################## Define as restrições utilizadas no modelo ##################'''

'''Define o tempo de deslocamento de cada navio i para realizar uma rota (j,k) com duração do tempo de viagem de (j->k)'''
model.R1 = ConstraintList()
for i in navios:
    for j in origens_j:
        for k in dentinos_k:
            model.R1.add(expr=(
                model.Tempo_Deslocamento[i, j, k] == model.Rota_navio[i, j, k] * Tempo_viagem[j][k]))


'''Define o tempo final de cada navio dado considerando o tempo inicial e o tempo de deslocamento de (j->k)'''
model.R2 = ConstraintList()
for i in navios:
    for j in origens_j:
        for k in dentinos_k:
            # model.R2.add(expr=(model.T_final[i, j, k] == model.T_inicial[i, j, k] + model.Tempo_Deslocamento[i, j, k]))
            model.R2.add(expr=(
                model.T_final[i, j, k] == model.T_inicial[j] + model.Tempo_Deslocamento[i, j, k]))

''' garante que caso um navio execute uma rota j->k a variável k_navio[i] seja igual a 1 '''
model.R3 = ConstraintList()
for i in navios:
    model.R3.add(expr=(model.k_navio[i]*M >= quicksum(
        model.Rota_navio[i, j, k] for j in origens_j for k in dentinos_k)))

'''Define que o tempo final de cada navio deve ser menor ou igual a data de entrega (i,j)'''
model.R4 = ConstraintList()
for i in navios:
    for j in origens_j:
        for k in dentinos_k:
            model.R4.add(expr=(model.T_final[i, j, k] <= Data_entrega[j][k]))

'''Garante que um navio i com origem j só possua um destino k'''
model.R5 = ConstraintList()
for i in navios:
    for j in origens_j:
        model.R5.add(
            expr=(quicksum(model.Rota_navio[i, j, k] for k in dentinos_k) <= 1))

'''Garante que um navio i com destino k só possua uma origem j'''
model.R6 = ConstraintList()
for i in navios:
    for k in dentinos_k:
        model.R6.add(
            expr=(quicksum(model.Rota_navio[i, j, k] for j in origens_j) <= 1))

'''Garante que se um navio i faça uma viagem de j para k, ele retorne a base j antes de fazer outra viagem'''
# model.R99 = ConstraintList()
# for i in navios:
#     for j in origens_j:
#         for k in dentinos_k:
#             model.R99.add(expr=(model.Rota_navio[i, k, j] >= model.Rota_navio[i, j, k]))


'Restrige diversas rotas proibidas que não devem ser percorridas pelos navios'
model.R7 = ConstraintList()
for i in navios:
    model.R7.add(expr=(model.Rota_navio[i, 0, 0] == 0))
    model.R7.add(expr=(model.Rota_navio[i, 0, 1] == 0))
    model.R7.add(expr=(model.Rota_navio[i, 0, 2] == 0))

    model.R7.add(expr=(model.Rota_navio[i, 1, 0] == 0))
    model.R7.add(expr=(model.Rota_navio[i, 1, 1] == 0))
    model.R7.add(expr=(model.Rota_navio[i, 1, 2] == 0))

    model.R7.add(expr=(model.Rota_navio[i, 2, 0] == 0))
    model.R7.add(expr=(model.Rota_navio[i, 2, 1] == 0))
    model.R7.add(expr=(model.Rota_navio[i, 2, 2] == 0))

    model.R7.add(expr=(model.Rota_navio[i, 3, 3] == 0))
    model.R7.add(expr=(model.Rota_navio[i, 3, 4] == 0))
    model.R7.add(expr=(model.Rota_navio[i, 4, 3] == 0))
    model.R7.add(expr=(model.Rota_navio[i, 4, 4] == 0))

'''Esta restrição identifica se a rota (j,k) foi atendida por alguma sonda'''
model.R8 = ConstraintList()
for j in origens_j:
    for k in dentinos_k:
        model.R8.add(expr=(model.soma_rotas[j, k] == quicksum(
            model.Rota_navio[i, j, k] for i in navios)))

'''Estas restrições garantem que as rotas a seguir sejam atendida por alguma sonda'''
model.R9 = ConstraintList()
model.R9.add(expr=(model.soma_rotas[0, 3] == 1))
model.R9.add(expr=(model.soma_rotas[0, 4] == 1))
model.R9.add(expr=(model.soma_rotas[1, 3] == 1))
model.R9.add(expr=(model.soma_rotas[1, 4] == 1))
model.R9.add(expr=(model.soma_rotas[2, 4] == 1))

'''Impede rotas para um mesmo destino, ou seja, i para i'''
model.R10 = ConstraintList()
for i in navios:
    for k in dentinos_k:
        model.R10.add(expr=(model.Rota_navio[i, k, k] == 0))


'''Define o total de navios utilizados'''
model.R11 = ConstraintList()
model.R11.add(expr=(model.total_de_navios_utilizados == quicksum(model.k_navio[i] for i in navios)))

################################################################
'''Define a função objetivo como a minimização do total de navios utilizados'''
model.obj = Objective(expr=(model.total_de_navios_utilizados),
                      sense=minimize
                      )

optimizer = Highs()
# optimizer = SolverFactory('glpk')
# optimizer = SolverFactory('gurobi')

'''Otimiza o modelo'''
results = optimizer.solve(model)

valor_final_fo = model.obj.expr()
print(f'Valor final da função objetivo: {valor_final_fo}')

'''Acessando os valores das variáveis após a otimização'''
for i in navios:
    valor = model.k_navio[i].value
    if valor > 0.0:
        print(f'k_navio[{i}] = {valor}')


for i in navios:
    for j in origens_j:
        for k in dentinos_k:
            valor = model.T_final[i, j, k].value
            if valor > 0.0:
                print(f'T_final[{i}, {j}, {k}] = {valor}')


for i in navios:
    for j in origens_j:
        # for k in dentinos_k:
        # valor = model.T_inicial[i, j, k].value
        valor = model.T_inicial[j].value
        if valor > 0.0:
            # print(f'T_inicial[{i}, {j}, {k}] = {valor}')
            print(f'T_inicial[{j}] = {valor}')

for i in navios:
    for j in origens_j:
        for k in dentinos_k:
            valor = model.Tempo_Deslocamento[i, j, k].value
            if valor > 0.0:
                print(f'Tempo_Deslocamento[{i}, {j}, {k}] = {valor}')

for i in navios:
    for j in origens_j:
        for k in dentinos_k:
            valor = model.Rota_navio[i, j, k].value
            if valor > 0.0:
                print(f'Rota_navio[{i}, {j}, {k}] = {valor}')

for j in origens_j:
    for k in dentinos_k:
        valor = model.soma_rotas[j, k].value
        if valor > 0.0:
            print(f'Soma_rotas[{j}, {k}] = {valor}')
            