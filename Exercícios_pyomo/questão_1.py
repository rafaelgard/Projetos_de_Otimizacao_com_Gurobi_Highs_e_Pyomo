import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.contrib.appsi.solvers.highs import Highs

'''################## Define os Parâmetros utilizados no modelo ##################'''
custo_de_treinamento_centro_base = np.array(
    [
        [200, 200, 300],
        [300, 400, 220],
        [300, 400, 250]
    ]
)
distancia = np.array(
    [
        [200, 200, 300],
        [300, 400, 220],
        [300, 400, 250]
    ]
)

# define o big-m
M = 1000000

'''Cria o modelo'''
model = ConcreteModel()

'''################## Define as variáveis utilizados no modelo ##################'''

'''XCB representa a quantidade treinada em cada centro i e base j'''
model.XCB = Var(range(0, 3), 
                range(0, 3),
                within = NonNegativeIntegers, 
                bounds=(0, None))


'''Q_T_P_B representa a quantidade treinada por base'''
model.Q_Treinada_Por_Base = Var(range(0, 3),
                                within = NonNegativeIntegers,
                                bounds=(0, None))

'''Rota_NP representa a rota percorrida pelos navios pequenos'''
model.Rota_NP = Var(range(4), 
                range(4),
                range(7),
                within = Binary, 
                bounds=(0, 1))

'''Rota_NG representa a rota percorrida pelos navios grandes'''
model.Rota_NG = Var(range(4), 
                range(4),
                range(5),
                within = Binary, 
                bounds=(0, 1))

'''Identifica a quantidade transportada em um navio pequeno em uma rota (i,j) em um navio k'''
model.QTD_NP = Var(range(4), 
                range(4),
                range(7),
                within = NonNegativeIntegers, 
                bounds=(0, None))

'''Identifica a quantidade transportada em um navio grande em uma rota (i,j) em um navio k'''
model.QTD_NG = Var(range(4), 
                range(4),
                range(5),
                within = NonNegativeIntegers, 
                bounds=(0, None))

'''Identifica a quantidade transportada por cada navio pequeno'''
model.QTD_transp_NP = Var(range(7),
                within = NonNegativeIntegers, 
                bounds=(0, 200))


'''Identifica a quantidade transportada por cada navio grande'''
model.QTD_transp_NG = Var(range(5),
                within = NonNegativeIntegers, 
                bounds=(0, 500))

'''Identifica a quantidade transportada por base em um navio pequeno'''
model.QTD_transp_por_base_NP = Var(range(3),
                within = NonNegativeIntegers, 
                bounds=(0, None))

'''Identifica a quantidade transportada por base em um navio grande'''
model.QTD_transp_por_base_NG = Var(range(3),
                within = NonNegativeIntegers, 
                bounds=(0, None))


'''Identifica a distancia percorrida por um navio pequeno'''
model.Distancia_rota_NP = Var(range(7),
                within = NonNegativeIntegers, 
                bounds=(0, None))

'''Identifica a distancia percorrida por um navio grande'''
model.Distancia_rota_NG = Var(range(7),
                within = NonNegativeIntegers, 
                bounds=(0, None))

'''Identifica se um navio pequeno foi utilizado'''
model.NP_utilizado = Var(range(7),
                within = Binary, 
                bounds=(0, 1))

'''Identifica se um navio grande foi utilizado'''
model.NG_utilizado = Var(range(5),
                within = Binary, 
                bounds=(0, 1))

'''Identifica a quantidade total trasnsportada'''
model.QTD_transp_total = Var(within = NonNegativeIntegers, 
                bounds=(0, None))

'''Identifica a distância percorrida por um navio pequeno'''
model.Distancia_percorrida_NP = Var(within = NonNegativeIntegers, 
                bounds=(0, None))

'''Identifica a distância percorrida por um navio grande'''
model.Distancia_percorrida_NG = Var(within = NonNegativeIntegers, 
                bounds=(0, None))

'''Identifica o custo fixo de navio pequeno'''
model.Custo_fixo_NP = Var(within = NonNegativeIntegers, 
                bounds=(0, None))

'''Identifica o custo fixo de navio grande'''
model.Custo_fixo_NG = Var(within = NonNegativeIntegers, 
                bounds=(0, None))


'''################## Define as restrições utilizadas no modelo ##################'''

'''Define a variavel que identifica a quantidade treinada por base j'''
model.R1 = ConstraintList()
for j in range(0, 3):
    model.R1.add(expr=(model.Q_Treinada_Por_Base[j] == quicksum(model.XCB[i, j] for i in range(0, 3))))
    
'''Define o custo de treinamento'''
model.custo_de_treinamento = quicksum(custo_de_treinamento_centro_base[i][j] * model.XCB[i, j]
                                   for i in range(0, 3)
                                   for j in range(0, 3))

'''Restrições de capacidade de demanda de treinamento por centro'''
model.R2 = ConstraintList()
model.R2.add(expr=(quicksum(model.XCB[0, j] for j in range(0, 3)) == 1000))
model.R2.add(expr=(quicksum(model.XCB[1, j] for j in range(0, 3)) == 600))
model.R2.add(expr=(quicksum(model.XCB[2, j] for j in range(0, 3)) == 700))

'''Restrições de capacidade de treinamento por base'''
model.R2.add(expr=(quicksum(model.XCB[i, 0] for i in range(0, 3)) <= 1000))
model.R2.add(expr=(quicksum(model.XCB[i, 1] for i in range(0, 3)) <= 600))
model.R2.add(expr=(quicksum(model.XCB[i, 2] for i in range(0, 3)) <= 700))

model.R3 = ConstraintList()
'''Define a variavel que identifica a quantidade treinada por base j'''
for j in range(0, 3):
    model.R3.add(expr=(model.Q_Treinada_Por_Base[j] == quicksum(model.XCB[i, j] for i in range(0, 3)) ))


'''==================Transporte=================='''
'''Identifica a quantidade transportada em cada navio pequeno'''
for k in range(0, 7):
    model.R3.add(expr=(model.QTD_transp_NP[k] == quicksum(model.QTD_NP[i, j, k] for i in range(0, 4) for j in range(0, 4))))
'''Identifica a quantidade transportada em cada navio grande'''
for k in range(0, 5):
    model.R3.add(expr=(model.QTD_transp_NG[k] == quicksum(model.QTD_NG[i, j, k] for i in range(0, 4) for j in range(0, 4)) ))

'''Define o número máximo de bases percorridas pela sonda pequena'''
for k in range(0, 7):
    model.R3.add(expr=(quicksum(model.Rota_NP[i, j, k] for j in range(0, 4) for i in range(0, 4)) <= 2))

'''Define o número máximo de bases percorridas pela sonda grande'''
for k in range(0, 5):
    model.R3.add(expr=(quicksum(model.Rota_NG[i, j, k] for j in range(0, 4) for i in range(0, 4)) <= 3))


'''Limita a quantidade transportada em cada navio'''
for k in range(0, 7):
    model.R3.add(expr=(model.QTD_transp_NP[k] <= 200))

for k in range(0, 5):
    model.R3.add(expr=(model.QTD_transp_NG[k] <= 500))

'''Limita as rotas para o mesmo destino'''
for k in range(0, 7):
    for i in range(0, 4):
        model.R3.add(expr=(model.Rota_NP[i, i, k] == 0))

for k in range(0, 5):
    for i in range(0, 4):
        model.R3.add(expr=(model.Rota_NG[i, i, k] == 0))

'''Limita o transporte para o mesmo destino'''
for k in range(0, 7):
    for i in range(0, 4):
        model.R3.add(expr=(model.QTD_NP[i, i, k] == 0))

for k in range(0, 5):
    for i in range(0, 4):
        model.R3.add(expr=(model.QTD_NG[i, i, k] == 0))

'''Limita as rotas proibidas'''
for k in range(0, 7):
    model.R3.add(expr=(model.Rota_NP[2, 1, k] == 0))
    model.R3.add(expr=(model.Rota_NP[3, 1, k] == 0))
    model.R3.add(expr=(model.Rota_NP[3, 2, k] == 0))

for k in range(0, 5):
    model.R3.add(expr=(model.Rota_NG[2, 1, k] == 0))
    model.R3.add(expr=(model.Rota_NG[3, 1, k] == 0))
    model.R3.add(expr=(model.Rota_NG[3, 2, k] == 0))

'''Limita o transporte para as rotas proibidas'''
for k in range(0, 7):
    model.R3.add(expr=(model.QTD_NP[2, 1, k] == 0))
    model.R3.add(expr=(model.QTD_NP[3, 1, k] == 0))
    model.R3.add(expr=(model.QTD_NP[3, 2, k] == 0))

for k in range(0, 5):
    model.R3.add(expr=(model.QTD_NG[2, 1, k] == 0))
    model.R3.add(expr=(model.QTD_NG[3, 1, k] == 0))
    model.R3.add(expr=(model.QTD_NG[3, 2, k] == 0))

'''Identifica a quantidade transportada por base de cada navio pequeno'''
model.R3.add(expr=(model.QTD_transp_por_base_NP[0] == quicksum(
    model.QTD_NP[i, j, k] for i in range(0, 4) for j in range(1, 2) for k in range(0, 7))))

model.R3.add(expr=(model.QTD_transp_por_base_NP[1] == quicksum(
    model.QTD_NP[i, j, k] for i in range(0, 4) for j in range(2, 3) for k in range(0, 7))))

model.R3.add(expr=(model.QTD_transp_por_base_NP[2] == quicksum(
    model.QTD_NP[i, j, k] for i in range(0, 4) for j in range(3, 4) for k in range(0, 7))))

'''Identifica a quantidade transportada por base de cada navio grande'''
model.R3.add(expr=(model.QTD_transp_por_base_NG[0] == quicksum(
    model.QTD_NG[i, j, k] for i in range(0, 4) for j in range(1, 2) for k in range(0, 5))))

model.R3.add(expr=(model.QTD_transp_por_base_NG[1] == quicksum(
    model.QTD_NG[i, j, k] for i in range(0, 4) for j in range(2, 3) for k in range(0, 5))))

model.R3.add(expr=(model.QTD_transp_por_base_NG[2] == quicksum(
    model.QTD_NG[i, j, k] for i in range(0, 4) for j in range(3, 4) for k in range(0, 5))))

'''Garante que todos os treinados por base sejam transportados pelos navios'''
model.R4 = ConstraintList()
model.R4.add(expr=(model.Q_Treinada_Por_Base[0] == model.QTD_transp_por_base_NP[0] + model.QTD_transp_por_base_NG[0]))
model.R4.add(expr=(model.Q_Treinada_Por_Base[1] == model.QTD_transp_por_base_NP[1] + model.QTD_transp_por_base_NG[1]))
model.R4.add(expr=(model.Q_Treinada_Por_Base[2] == model.QTD_transp_por_base_NP[2] + model.QTD_transp_por_base_NG[2]))

'''Identifica a quantidade total transportada das 3 bases'''
model.R4.add(expr=(model.QTD_transp_total == quicksum(model.QTD_transp_NP[k] for k in range(0, 7)) + quicksum(
    model.QTD_transp_NG[k] for k in range(0, 5))))

'''Garante que a quantidade total transportada seja igual a quantidade total treinada por base'''
model.R4.add(expr=(model.QTD_transp_total == model.Q_Treinada_Por_Base[0] + model.Q_Treinada_Por_Base[1] + model.Q_Treinada_Por_Base[2]))

'''Calcula a distancia percorrida em cada rota pelos navios pequenos'''
model.R5 = ConstraintList()
model.R5.add(expr=(model.Distancia_rota_NP[0] == (370 / 2) * quicksum(model.Rota_NP[0, 1, k] + model.Rota_NP[1, 0, k] for k in range(0, 7))))
model.R5.add(expr=(model.Distancia_rota_NP[1] == (515 / 3) * quicksum(model.Rota_NP[0, 1, k] + model.Rota_NP[1, 2, k] + model.Rota_NP[2, 0, k] for k in range(0, 7))))


model.R5.add(expr=(model.Distancia_rota_NP[2] == (665 / 3) * quicksum(model.Rota_NP[0, 2, k] + model.Rota_NP[2, 3, k] + model.Rota_NP[3, 0, k] for k in range(0, 7))))

model.R5.add(expr=(model.Distancia_rota_NP[3] == (460 / 2) * quicksum(model.Rota_NP[0, 2, k] + model.Rota_NP[2, 0, k] for k in range(0, 7))))

model.R5.add(expr=(model.Distancia_rota_NP[4] == (600 / 2) * quicksum(model.Rota_NP[0, 3, k] + model.Rota_NP[3, 0, k] for k in range(0, 7))))

model.R5.add(expr=(model.Distancia_rota_NP[5] == (640 / 3) * quicksum(model.Rota_NP[0, 1, k] + model.Rota_NP[1, 3, k] + model.Rota_NP[3, 0, k] for k in range(0, 7))))

'''Calcula a distancia percorrida em cada rota pelos navios grandes'''
model.R6 = ConstraintList()

model.R6.add(expr=(model.Distancia_rota_NG[0] == (370 / 2) * quicksum(model.Rota_NG[0, 1, k] + model.Rota_NG[1, 0, k] for k in range(0, 5))))

model.R6.add(expr=(model.Distancia_rota_NG[1] == (515 / 3) * quicksum(model.Rota_NG[0, 1, k] + model.Rota_NG[1, 2, k] + model.Rota_NG[2, 0, k] for k in range(0, 5))))

model.R6.add(expr=(model.Distancia_rota_NG[2] == (665 / 3) * quicksum(model.Rota_NG[0, 2, k] + model.Rota_NG[2, 3, k] + model.Rota_NG[3, 0, k] for k in range(0, 5))))

model.R6.add(expr=(model.Distancia_rota_NG[3] == (460 / 2) * quicksum(model.Rota_NG[0, 2, k] + model.Rota_NG[2, 0, k] for k in range(0, 5))))

model.R6.add(expr=(model.Distancia_rota_NG[4] == (600 / 2) * quicksum(model.Rota_NG[0, 3, k] + model.Rota_NG[3, 0, k] for k in range(0, 5))))

model.R6.add(expr=(model.Distancia_rota_NG[5] == (640 / 3) * quicksum(model.Rota_NG[0, 1, k] + model.Rota_NG[1, 3, k] + model.Rota_NG[3, 0, k] for k in range(0, 5))))

model.R6.add(expr=(model.Distancia_rota_NG[6] == (720 / 4) * quicksum(model.Rota_NG[0, 1, k] + model.Rota_NG[1, 2, k] + model.Rota_NG[2, 3, k] + model.Rota_NG[3, 3, k] for k in range(0, 5))))

'''Calcula a distancia total percorrida pelas navios pequenos'''
model.R7 = ConstraintList()

model.R7.add(expr=(model.Distancia_percorrida_NP == quicksum(model.Distancia_rota_NP[k] for k in range(0, 7))))

'''Calcula a distancia total percorrida pelas navios grandes'''
model.R7.add(expr=(model.Distancia_percorrida_NG == quicksum(model.Distancia_rota_NG[k] for k in range(0, 5))))

'''Calcula o custo fixo total das navios pequenos utilizadas'''
model.R7.add(expr=(model.Custo_fixo_NP == 5000 * quicksum(model.NP_utilizado[k] for k in range(0, 7))))

'''Calcula o custo fixo total das navios grandes utilizadas'''
model.R7.add(expr=(model.Custo_fixo_NG == 10000 * quicksum(model.NG_utilizado[k] for k in range(0, 5))))

'''Garante que caso um navio pequeno seja utilizado, seja corretamente identificado'''
model.r8 = ConstraintList()
for i in range(0, 7):
    model.r8.add(expr=(model.NP_utilizado[i]*M>=model.QTD_transp_NP[i]))

'''Garante que caso um navio grande seja utilizado, seja corretamente identificado'''
model.r9 = ConstraintList()
for i in range(0, 5):
    model.r9.add(expr=(model.NG_utilizado[i]*M>=model.QTD_transp_NG[i]))

'''Define a função objetivo'''
model.obj = Objective(expr=(model.custo_de_treinamento + model.Distancia_percorrida_NP * 2 +
                    model.Distancia_percorrida_NG * 3 +
                    model.Custo_fixo_NP +
                    model.Custo_fixo_NG),
                      sense = minimize
                      )
'''Otimiza o modelo
Aqui é possível alterar entre o otimizador que será utilizado, 
bastando apenas habilitar e desabilitar as linhas a seguir.
Obs: o solver GLPK só irá funcionar se vc tiver instalado corretamente em seu computador!
'''
optimizer = Highs()
# optimizer = SolverFactory('glpk')
# optimizer = SolverFactory('gurobi')

'''Otimiza o modelo'''
results = optimizer.solve(model)

valor_final_fo = model.obj.expr()
print(f'Valor final da função objetivo: {valor_final_fo}')

for k in range(0, 7):
    valor = model.NP_utilizado[k].value
    if valor>0.0:
        print(f'NP_utilizado[{k}] = {valor}')

for k in range(0, 5):
    valor = model.NG_utilizado[k].value
    if valor>0.0:
        print(f'NG_utilizado[{k}] = {valor}')
     
for k in range(0, 7):
    valor = model.QTD_transp_NP[k].value
    if valor>0.0:
        print(f'QTD_transp_NP[{k}] = {valor}')

for k in range(0, 5):
    valor = model.QTD_transp_NG[k].value
    if valor>0.0:
        print(f'QTD_transp_NG[{k}] = {valor}')

for k in range(0, 3):
    valor = model.QTD_transp_por_base_NG[k].value
    if valor>0.0:
        print(f'QTD_transp_por_base_NG[{k}] = {valor}')

for k in range(0, 3):
    valor = model.QTD_transp_por_base_NP[k].value
    if valor>0.0:
        print(f'QTD_transp_por_base_NP[{k}] = {valor}')

for k in range(0, 3):
    valor = model.Q_Treinada_Por_Base[k].value
    if valor>0.0:
        print(f'Q_Treinada_Por_Base[{k}] = {valor}')
