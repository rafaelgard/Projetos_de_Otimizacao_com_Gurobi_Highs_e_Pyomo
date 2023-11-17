'''Importa as bibliotecas'''
import random
import time
import os
import numpy as np
import pandas as pd
import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.contrib.appsi.solvers.highs import Highs
from pyomo.opt import SolverStatus, TerminationCondition
from ..utils.utilidades import highs_solver

np.random.seed(0)
#, 'glpk':SolverFactory('glpk')
# optimizers = {'highs':Highs(), 'gurobi':SolverFactory('gurobi')}
optimizers = {'highs':Highs()}
# optimizers = {'gurobi':SolverFactory('gurobi')}

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
# resultados = {'id_instancia':[], 'Optimizer':[], 'Tempos Máximos': [], 'Quantidade_de_drones': [], 'Z': [], 'Tempo Computacional': [], 'Quantidade de restrições': [], 'Quantidade de variáveis': [], 't_max_solver':[]}
resultados = {'id_instancia':[], 'Optimizer':[], 'Tempos Máximos': [], 'Quantidade_de_drones': [], 'Z': [], 'Tempo Computacional': [], 't_max_solver':[]}

'''Define parâmetros que serão variados para avaliar o modelo'''
total_de_drones = [1, 2, 3, 4, 5]
tempos_maximos = [100, 120, 140, 160, 180, 200, 300, 1000, 2000]
tempos_maximos_disponiveis_solver = [100, 300, 600]

for t_max_solver in tempos_maximos_disponiveis_solver:
    for optimizer_name in optimizers.keys():
        '''Varia a quantidade de drones e o tempo máximo para avaliar os resultados para a instância'''
        for numero_instancia in [1,2]:
            '''Carrega a instância'''
            N, N_sem_zero, H, H_sem_zero, A, u, b = carrega_instancia(numero_instancia)

            for qtd_drones in total_de_drones:
                for Tmax in tempos_maximos:
                    inicio = time.time()

                    '''Declara o modelo'''
                    # modelo = grb.Model(name="Drones")
                    model = ConcreteModel()

                    '''Indices dos drones'''
                    K = range(0, qtd_drones)

                    '''==========================Adicionar as variáveis=========================='''
                    model.x = Var(N, H, N, H, K, within = Binary)
                    model.y = Var(N_sem_zero, H_sem_zero, K, within = Binary)
                    model.u_ = Var(N, H, K, within = PositiveReals, bounds=(0, None))
                    model.funcao_objetivo = Var(within = PositiveReals, bounds=(0, None))

                    '''==========================Adiciona Parametros=========================='''
                    t = {(i, r, j, s): calcula_distancia(N[i], H[r], N[j], H[s]) for i in N for r in H for j in N for s in H}

                    '''==========================Adiciona Restricoes=========================='''

                    model.R1 = ConstraintList()
                    for i in N:
                        for j in N:
                            for r in H:
                                for s in H: 
                                    for k in K:
                                        if (i, r) == (j, s):
                                            model.R1.add(expr=(model.x[i, r, j, s, k] == 0))

                    
                    '''Esta restrição garante que ao menos 1 sonda vigie cada área a em A'''
                    model.R2 = ConstraintList()
                    for a in A:
                        model.R2.add(expr=(quicksum(b[a][i][r]*model.y[i, r, k] for k in K for r in H_sem_zero for i in N_sem_zero) >= 1 ))
                    
                    

                    model.R3 = ConstraintList()
                    for i in N_sem_zero:
                        for r in H_sem_zero:
                            model.R3.add(expr=(quicksum(model.y[i, r, k] for k in K) <= 1))
                    
                    
                    
                    '''Esta restrição garante que apenas 1 sonda pare em um nó (i, r)'''
                    model.R4 = ConstraintList()
                    
                    for i in N_sem_zero:
                        for r in H_sem_zero:
                            for k in K:
                                model.R4.add(expr=(quicksum(model.x[i, r, j, s, k] for s in H for j in N if i != j and r != s) == model.y[i, r, k]))
                                # modelo.addConstrs((grb.quicksum(x[i, r, j, s, k] for s in H for j in N if i != j and r != s) == y[i, r, k] for i in N_sem_zero for r in H_sem_zero for k in K), name='R4')
                    
                    model.R5 = ConstraintList()
                    model.R6 = ConstraintList()
                    
                    for k in K:
                        model.R5.add(expr=(quicksum(model.x[0, 0, j, s, k] for s in H for j in N) == 1))                        
                        model.R6.add(expr=(quicksum(model.x[i, r, 0, 0, k] for i in N for r in H) == 1))

                    model.R7 = ConstraintList()
                    for i in N:
                        for r in H:
                            for k in K:
                                model.R7.add(expr=(quicksum(model.x[i, r, j, s, k] for s in H for j in N if i != j and r != s) == quicksum(model.x[j, s, i, r, k] for s in H for j in N if i != j and r != s)))
                    
                    model.R8 = ConstraintList()
                    for k in K:
                        model.R8.add(expr=(quicksum(t[i, r, j, s]*model.x[i, r, j, s, k] for r in H for s in H for i in N for j in N) + quicksum(u[i][r]*model.y[i, r, k] for r in H_sem_zero for i in N_sem_zero) <= Tmax))
                    
                    '''Esta restrição garante que o tempo que a sonda percorre somado ao tempo que ela fica parada em uma área não exceda o tempo máximo que o drone pode voar'''
                    model.R9 = ConstraintList()
                    
                    for k in K:
                        model.R9.add(expr=(model.u_[0, 0, k] == 0))

                    model.R10 = ConstraintList()
                    for i in N:
                        for r in H:
                            for j in N_sem_zero:
                                for s in H_sem_zero:
                                    for k in K:
                                        if i != j and r != s:
                                            model.R10.add(expr=(model.u_[j, s, k] >= model.u_[i, r, k] +1 -len(N)*len(H)*(1-model.x[i, r, j, s, k])))
                    
                    model.R11 = ConstraintList()                    
                    for i in N:
                        for r in H:
                            for k in K:
                                model.R11.add(expr=(model.u_[i, r, k] <= len(N)*len(H)-1))

                    '''==========================Define a função Objetivo=========================='''
                    model.R12 = ConstraintList()
                    model.R12.add(expr=(model.funcao_objetivo == quicksum(t[i, r, j, s]*model.x[i, r, j, s, k] for r in H for s in H for i in N for j in N for k in K) + quicksum(u[i][r]*model.y[i, r, k] for r in H_sem_zero for i in N_sem_zero for k in K))) 

                    model.obj = Objective(expr=(model.funcao_objetivo),
                                        sense = minimize
                                        )

                    '''==========================Otimiza o modelo=========================='''
                    optimizer = optimizers[optimizer_name]
                    
                    '''Otimiza o modelo'''
                    print('================================================')
                    if optimizer_name != 'highs' and optimizer_name !='gurobi':
                        results = optimizer.solve(model, tee=True, timelimit=t_max_solver)
                    
                    elif optimizer_name == 'gurobi':
                        optimizer.options['TimeLimit'] = t_max_solver
                        results = optimizer.solve(model, tee=True)
                    
                    elif optimizer_name == 'highs':
                        # try:
                        path = "Exercícios_pyomo\drone.mps"
                        model.write(path)
                        h_model, solution, model_status, model_status_name, valor_final_fo = highs_solver(path, t_max_solver)
                        os.remove(path)

                    '''Identifica o tempo final''' 
                    final = time.time()
                    print(f'Tempo final: {round(final-inicio, 3)}')

                    if optimizer_name == 'gurobi' and (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
                        valor_final_fo = results.obj.expr()
                        print(f'Valor final da função objetivo: {valor_final_fo}')

                        '''Salva os resultados'''
                        resultados['id_instancia'].append(numero_instancia)
                        resultados['Optimizer'].append(optimizer_name)
                        resultados['t_max_solver'].append(t_max_solver)
                        resultados['Tempos Máximos'].append(round(Tmax, 3))
                        resultados['Quantidade_de_drones'].append(round(qtd_drones, 3))
                        resultados['Tempo Computacional'].append(round(final-inicio, 3))
                        # resultados['Quantidade de variáveis'].append(round(model.NumVars, 3))
                        # resultados['Quantidade de restrições'].append(round(model.NumConstrs, 3))

                        if isinstance(model.obj.expr(), float):
                            resultados['Z'].append(round(model.obj.expr(), 3))
                        else:
                            resultados['Z'].append('inf')
                    
                    elif optimizer_name =='highs':

                        if model_status_name =='kInfeasible':
                            valor_final_fo == 'inf'
                        

                        '''Salva os resultados'''
                        resultados['id_instancia'].append(numero_instancia)
                        resultados['Optimizer'].append(optimizer_name)
                        resultados['t_max_solver'].append(t_max_solver)
                        resultados['Tempos Máximos'].append(round(Tmax, 3))
                        resultados['Quantidade_de_drones'].append(round(qtd_drones, 3))
                        resultados['Tempo Computacional'].append(round(final-inicio, 3))
                        # resultados['Quantidade de variáveis'].append(round(model.NumVars, 3))
                        # resultados['Quantidade de restrições'].append(round(model.NumConstrs, 3))

                        if isinstance(valor_final_fo, float):
                            resultados['Z'].append(round(valor_final_fo, 3))
                        else:
                            resultados['Z'].append(valor_final_fo)
        
                        print(f'Valor final da função objetivo: {valor_final_fo}')
                    
                    else:
                        valor_final_fo = 'inf'
                        print(f'Valor final da função objetivo: {valor_final_fo}')
                        
                        '''Salva os resultados'''
                        resultados['id_instancia'].append(numero_instancia)
                        resultados['t_max_solver'].append(t_max_solver)
                        resultados['Optimizer'].append(optimizer_name)
                        resultados['Tempos Máximos'].append(round(Tmax, 3))
                        resultados['Quantidade_de_drones'].append(round(qtd_drones, 3))
                        resultados['Tempo Computacional'].append(round(final-inicio, 3))
                        resultados['Z'].append('inf')

print('====================================')
print(resultados)

resultados = pd.DataFrame(resultados).fillna('-')

'''Salva os resultados gerados em um arquivo .csv'''
try:
    resultados.to_csv('Exercícios_pyomo\resultados_drones_completo.csv', index=False, sep=';')

except:
    print('Erro ao salvar resultados!')
    breakpoint()