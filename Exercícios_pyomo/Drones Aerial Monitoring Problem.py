'''Importa as bibliotecas'''
import random
import time
# import gurobipy as grb
import numpy as np
import pandas as pd
import numpy as np

from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.contrib.appsi.solvers.highs import Highs

def carrega_instancia(numero_instancia):
    '''Função utilizada para carregar a instância'''

    if numero_instancia == 1:
        '''Define os nós no espaço por quais os drones podem se movimentar'''
        N = {0: (0, 0), 1: (0, 25), 2: (0, 50), 3: (0, 75), 4: (0, 100),
             5: (1, 0), 6: (1, 25), 7: (1, 50), 8: (1, 75), 9: (1, 100),
             10: (2, 0), 11: (2, 25), 12: (2, 50), 13: (2, 75), 14: (2, 100),
             15: (3, 0), 16: (3, 25), 17: (3, 50), 18: (3, 75), 19: (3, 100)
             }

        N_sem_zero = {n: N[n] for n in range(1, len(N))}

        H = {0: 50, 1: 100, 2: 300}
        H_sem_zero = {n: N[n] for n in range(1, len(H))}

        '''Áreas monitoradas'''
        A = {0: (43, 17), 1: (80, 82), 2: (26, 27), 3: (36, 71), 4: (26, 35), 5: (22, 33), 6: (62, 87), 7: (26, 49)}

        u = {i: {r: 90 for r in range(len(H))} for i in range(len(N))} #random.randint(10, 30)

        b = {a: {i: {r: 1 for r in range(len(H))} for i in range(len(N))} for a in range(len(A))}

        '''Torna os pontos com altura de 300m com acurácia insuficiente e portanto recebe o valor 0'''
        for a in range(len(A)):
            for i in range(len(N)):
                b[a][i][2] = 0

        return N, N_sem_zero, H, H_sem_zero, A, u, b

    if numero_instancia == 2:

        '''Define os nós no espaço por quais os drones podem se movimentar'''
        N = {
            0: (0, 0), 1: (0, 25), 2: (0, 50), 3: (0, 75), 4: (0, 100),
             5: (1, 0), 6: (1, 25), 7: (1, 50), 8: (1, 75), 9: (1, 100),
             10: (2, 0), 11: (2, 25), 12: (2, 50), 13: (2, 75), 14: (2, 100),
             15: (3, 0), 16: (3, 25), 17: (3, 50), 18: (3, 75), 19: (3, 100),
             20: (34, 71), 21: (8, 28), 22: (62, 72), 23: (63, 17), 24: (62, 53),
            25: (22, 33), 26: (26, 78), 27: (64, 54), 28: (25, 54), 29: (55, 34),
            30: (35, 67), 31: (4, 71), 32: (90, 4)
             }

        '''Define os índices N a partir do índice 1'''
        N_sem_zero = {n: N[n] for n in range(1, len(N))}

        '''Define os índices H'''
        H = {0: 0, 1: 50, 2: 100, 3: 300}
        H_sem_zero = {n: N[n] for n in range(1, len(H))}

        '''Áreas monitoradas'''
        A = {0: (34, 71), 1: (8, 28), 2: (62, 72), 3: (63, 17), 4: (62, 53), 5: (22, 33),
              6: (26, 78), 7: (64, 54), 8: (25, 54), 9: (55, 34), 10: (35, 67)}

        '''Define os tempos de monitoramento em 90 segundos'''
        u = {i: {r: 90 for r in H_sem_zero} for i in N_sem_zero}

        '''Define inicialmente que todos os pontos possuem acurária igual a 1, ou seja, são viáveis'''
        b = {a: {i: {r: 1 for r in H} for i in N} for a in A}

        '''Posteriormente escolhe aleatoriamente pontos onde a acurácia é baixa e que não podem ser utilizados
        para monitorar uma área'''
        for a in range(len(A)):
            for i in range(0, len(N), 2):
                for r in range(0, len(H), 2):
                    b[a][i][r] = random.randint(0, 2)

        return N, N_sem_zero, H, H_sem_zero, A, u, b

def calcula_distancia(i, r, j, s):
    '''Calcula a distancia entre 2 pontos (i,r) -> (j,s) e retorna a distância'''

    dist = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2 + (r-s)**2)
    return dist

'''Cria o dicionário para salvar os resultados'''
resultados = {'Tempos Máximos': [], 'Quantidade_de_drones': [], 'Z': [], 'Tempo Computacional': [], 'Quantidade de restrições': [], 'Quantidade de variáveis': []}

'''Define parâmetros que serão variados para avaliar o modelo'''
total_de_drones = [1, 2, 3, 4, 5]
tempos_maximos = [100, 120, 140, 160, 180, 200, 300, 1000, 2000]

'''Carrega a instância'''
N, N_sem_zero, H, H_sem_zero, A, u, b = carrega_instancia(2)

'''Varia a quantidade de drones e o tempo máximo para avaliar os resultados para a instância'''
for qtd_drones in total_de_drones:
    for Tmax in tempos_maximos:
        inicio = time.time()

        '''Declara o modelo'''
        # modelo = grb.Model(name="Drones")
        model = ConcreteModel()

        '''Indices dos drones'''
        K = range(0, qtd_drones)

        '''==========================Adicionar as variáveis=========================='''
        # x = modelo.addVars(N, H, N, H, K, vtype=grb.GRB.BINARY, lb=0, name='x')
        # y = modelo.addVars(N_sem_zero, H_sem_zero, K, vtype=grb.GRB.BINARY, lb=0, name='y')
        # u_ = modelo.addVars(N, H, K, vtype=grb.GRB.CONTINUOUS, lb=0, name='u_')

        model.x = Var(N, H, N, H, K, within = NonNegativeIntegers)
        model.y = Var(N_sem_zero, H_sem_zero, K, within = Binary)
        model.u_ = Var(N, H, K, within = NonNegativeReals)

        '''==========================Adiciona Parametros=========================='''
        t = {(i, r, j, s): calcula_distancia(N[i], H[r], N[j], H[s]) for i in N for r in H for j in N for s in H}

        '''==========================Adiciona Restricoes=========================='''

        # modelo.addConstrs((x[i, r, j, s, k] == 0 for i in N for j in N for r in H for s in H for k in K if (i, r) == (j, s)), name='R0')
        model.R1 = ConstraintList()
        for i in N:
            for j in N:
                for r in H:
                    for s in H: 
                        for k in K:
                            if (i, r) == (j, s):
                                model.R1.add(expr=(model.x[i, r, j, s, k] == 0))

        
        model.R2 = ConstraintList()
        # modelo.addConstrs((grb.quicksum(b[a][i][r]*y[i, r, k] for k in K for r in H_sem_zero for i in N_sem_zero) >= 1 for a in A), name='R2')
        
        '''Esta restrição garante que ao menos 1 sonda vigie cada área a em A'''
        for a in A:
            model.R2.add(expr=(quicksum(b[a][i][r]*model.y[i, r, k] for k in K for r in H_sem_zero for i in N_sem_zero) >= 1 ))
        
        

        # modelo.addConstrs((grb.quicksum(y[i, r, k] for k in K) <= 1 for i in N_sem_zero for r in H_sem_zero), name='R3')
        model.R3 = ConstraintList()
        for i in N_sem_zero:
            for r in H_sem_zero:
                model.R3.add(expr=(quicksum(model.y[i, r, k] for k in K) <= 1))
        
        
        
        '''Esta restrição garante que apenas 1 sonda pare em um nó (i, r)'''

        # modelo.addConstrs((grb.quicksum(x[i, r, j, s, k] for s in H for j in N if i != j and r != s) == y[i, r, k] for i in N_sem_zero for r in H_sem_zero for k in K), name='R4')

        # modelo.addConstrs((grb.quicksum(x[0, 0, j, s, k] for s in H for j in N) == 1 for k in K), name='R5')

        # modelo.addConstrs((grb.quicksum(x[i, r, 0, 0, k] for i in N for r in H) == 1 for k in K), name='R6')

        # modelo.addConstrs((grb.quicksum(x[i, r, j, s, k] for s in H for j in N if i != j and r != s) == grb.quicksum(x[j, s, i, r, k] for s in H for j in N if i != j and r != s) for i in N for r in H for k in K), name='R7')

        # modelo.addConstrs(((grb.quicksum(t[i, r, j, s]*x[i, r, j, s, k] for r in H for s in H for i in N for j in N) + grb.quicksum(u[i][r]*y[i, r, k] for r in H_sem_zero for i in N_sem_zero)) <= Tmax for k in K), name='R9')
        
        model.R4 = ConstraintList()
        
        for i in N_sem_zero:
            for r in H_sem_zero:
                for k in K:
                    model.R4.add(expr=(quicksum(model.x[i, r, j, s, k] for s in H for j in N if i != j and r != s) == model.y[i, r, k]))
        
        for k in K:
            model.R4.add(expr=(quicksum(model.x[0, 0, j, s, k] for s in H for j in N) == 1))
            model.R4.add(expr=(quicksum(model.x[i, r, 0, 0, k] for i in N for r in H) == 1))

        for i in N:
            for r in H:
                for k in K:
                    model.R4.add(expr=(quicksum(model.x[i, r, j, s, k] for s in H for j in N if i != j and r != s) == quicksum(model.x[j, s, i, r, k] for s in H for j in N if i != j and r != s)))
        
        for k in K:
            model.R4.add(expr=(quicksum(t[i, r, j, s]*model.x[i, r, j, s, k] for r in H for s in H for i in N for j in N) + quicksum(u[i][r]*model.y[i, r, k] for r in H_sem_zero for i in N_sem_zero) <= Tmax))
        
        
        '''Esta restrição garante que o tempo que a sonda percorre somado ao tempo que ela fica parada em uma área não exceda o tempo máximo que o drone pode voar'''

        model.R5 = ConstraintList()
        
        # modelo.addConstrs((u_[0, 0, k] == 0 for k in K), name='R12')

        for k in K:
            model.R5.add(expr=(model.u_[0, 0, k] == 0))


        model.R6 = ConstraintList()
        # modelo.addConstrs((u_[j, s, k] >= u_[i, r, k] +1 -len(N)*len(H)*(1-x[i, r, j, s, k]) for i in N for r in H for j in N_sem_zero for s in H_sem_zero for k in K if i != j and r != s), name='R13')

        
        for i in N:
            for r in H:
                for j in N_sem_zero:
                    for s in H_sem_zero:
                        for k in K:
                            if i != j and r != s:
                                model.R6.add(expr=(model.u_[j, s, k] >= model.u_[i, r, k] +1 -len(N)*len(H)*(1-model.x[i, r, j, s, k])))

        
        
        model.R7 = ConstraintList()
        # modelo.addConstrs((u_[i, r, k] <= len(N)*len(H)-1 for i in N for r in H for k in K), name='R14')

        
        for i in N:
            for r in H:
                for k in K:
                    model.R7.add(expr=(model.u_[i, r, k] <= len(N)*len(H)-1))

        # breakpoint()
        '''Declara a função objetivo'''
        # modelo.setObjective((grb.quicksum(t[i, r, j, s]*x[i, r, j, s, k] for r in H for s in H for i in N for j in N for k in K) + grb.quicksum(u[i][r]*y[i, r, k] for r in H_sem_zero for i in N_sem_zero for k in K)))

        
        '''Define a função objetivo'''
        model.obj = Objective(expr=(quicksum(t[i, r, j, s]*model.x[i, r, j, s, k] for r in H for s in H for i in N for j in N for k in K) + quicksum(u[i][r]*model.y[i, r, k] for r in H_sem_zero for i in N_sem_zero for k in K)),
                            sense = minimize
                            )
        
        
        '''==========================Define a função Objetivo=========================='''
        # modelo.ModelSense = grb.GRB.MINIMIZE
        # modelo.update()
        # modelo.write("instancia_spam.lp")

        '''==========================Limita o tempo de execução=========================='''
        # modelo.Params.timeLimit = 100

        '''==========================Otimiza o modelo=========================='''
        # modelo.optimize()

        '''Otimiza o modelo'''
        # optimizer = SolverFactory('ipopt')
        optimizer = SolverFactory('glpk')
        # optimizer = Highs()
        # optimizer = SolverFactory('gurobi')

        '''Otimiza o modelo'''
        results = optimizer.solve(model)

        '''Identifica o tempo final'''
        final = time.time()

        valor_final_fo = model.obj.expr()
        print(f'Valor final da função objetivo: {valor_final_fo}')

        # '''Salva os resultados'''
        # resultados['Tempos Máximos'].append(round(Tmax, 3))
        # resultados['Quantidade_de_drones'].append(round(qtd_drones, 3))
        # resultados['Tempo Computacional'].append(round(final-inicio, 3))
        # resultados['Quantidade de variáveis'].append(round(modelo.NumVars, 3))
        # resultados['Quantidade de restrições'].append(round(modelo.NumConstrs, 3))

        # if modelo.SolCount > 0:
        #     resultados['Z'].append(round(modelo.ObjVal, 3))
        # else:
        #     resultados['Z'].append('inf')

print('====================================')
print(resultados)
'''Salva os resultados gerados em um arquivo .csv'''
# pd.DataFrame(resultados).to_csv('resultados_drones.csv', index=False, sep=';')