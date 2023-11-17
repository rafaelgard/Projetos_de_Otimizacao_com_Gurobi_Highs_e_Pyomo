import highspy
import pandas as pd
import os

def imprime_resultados_geral_gurobipy(modelo):
    '''Imprime os resultados de um modelo criado no gurobipy'''

    print('=========================================')
    print('Quantidade de restrições: ', modelo.NumConstrs)
    print('Quantidade de variáveis: ', modelo.NumVars)
    print('=========================================')
    print('Status do modelo: ', modelo.Status)
    print('=========================================')
    print('Valor da função objetivo: ', modelo.ObjVal)
    print('=========================================')
    for v in modelo.getVars():
        # é possível alterar a linha a seguir caso seu modelo
        # possua valores negativos nas variáveis
        if v.x != 0:
            print('%s %g' % (v.varName, v.x))
            print('=========================================')


def salva_resultados_gurobipy(modelo):
    '''Esta função salva os resultados do modelo. 
    A chamada dessa função deve ficar ao final do código do modelo 
    logo depois de modelo.optimize()

    modelo: modelo do gurobipy
    '''

    # Cria o dicionário para salvar os resultados
    resultados = {'Tempos Máximos': [], 'Z': [], 'Tempo Computacional': [
    ], 'Quantidade de restrições': [], 'Quantidade de variáveis': []}

    '''Recupera informações do modelo'''
    param_info = modelo.getParamInfo('TimeLimit')
    Tmax = param_info.get('Value')

    '''Salva os resultados'''
    resultados['Tempos Máximos'].append(round(Tmax, 3))
    resultados['Tempo Computacional'].append(round(modelo.Runtime, 3))
    resultados['Quantidade de variáveis'].append(round(modelo.NumVars, 3))
    resultados['Quantidade de restrições'].append(round(modelo.NumConstrs, 3))

    '''Salva os resultados gerados em um arquivo .csv'''
    pd.DataFrame(resultados).to_csv('resultados.csv', index=False, sep=';')

def highs_solver(model, t_max_solver: float):
    '''Esta função lê um modelo criado no pyomo e utiliza o solver highs para otimizar o modelo
    
    model: modelo criado no pyomo

    t_max_solver: tempo máximo em mínutos que o solver tem disponível para otimizar o modelo
    '''

    # cria um modelo temporário que será lido pelo highs
    temp_path = "temp_model.mps"
    model.write(temp_path)

    h = highspy.Highs()
    status = h.readModel(temp_path)
    print(f'Lendo arquivo do modelo{temp_path}, returns a status of {status}')

    ### aqui é possível modificar diversos hiperparâmetros do solver

    # define o tempo limite máximo de execução do otimizador
    h.setOptionValue("time_limit", t_max_solver)

    # modifique conforme o número de threads do seu processador
    # h.setOptionValue("threads", 12)

    # define o gap considerado pelo solver
    # h.setOptionValue("mip_rel_gap", 0.0001)

    # define a porcentagem de tempo gasto na heurística
    # h.setOptionValue("mip_heuristic_effort", 0.3)

    h.run()
    solution = h.getSolution()
    info = h.getInfo()
    model_status = h.getModelStatus()

    print('Model status = ', h.modelStatusToString(model_status))
    print('Optimal objective = ', info.objective_function_value)
    print('Iteration count = ', info.simplex_iteration_count)
    print('Primal solution status = ',
          h.solutionStatusToString(info.primal_solution_status))
    print('Dual solution status = ',
          h.solutionStatusToString(info.dual_solution_status))
    print('Basis validity = ', h.basisValidityToString(info.basis_validity))

    # cria um modelo temporário
    os.remove(temp_path)

    return h, solution, model_status.value, model_status.name, info.objective_function_value
