from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.contrib.appsi.solvers.highs import Highs

'''======================= Define os Parâmetros utilizados no modelo ======================='''
V_max_tanque = [2700, 2800, 1100, 1800, 3400]

Demanda_de_combustivel = [2900, 4000, 4900]

Demanda_maxima_n_atendida = [500, 500, 500]

Penalidade = [10, 8, 6]

'''Define o BIG M'''
M = 1000000

'''Cria o modelo'''
model = ConcreteModel()

'''Define os índices'''
combustiveis = range(0, 5)
tanques = range(0, 3)

'''Bin_tipo representa se um combustível j será armazenado em um tanque i'''
model.Bin_tipo = Var(combustiveis, tanques, within = Binary, bounds=(0, 1))

'''======================= Define as variáveis utilizados no modelo ======================='''
'''Define a quantidade de combustível j armazenado em um tanque i'''
model.Vol = Var(combustiveis, tanques, within = Integers, bounds=(0, None))

'''Identifica a quantidade de combustível j que foi armazenado em todos os tanques i'''
model.Vol_atend = Var(tanques, within = Integers, bounds=(0, None))

'''Identifica a quantidade total de combustível j não atendida'''
model.Vol_n_atendido = Var(tanques, within = Integers, bounds=(0, None))

'''Variável binária que identifica se a quantidade de combustível j não atendida é menor ou igual a 100 ou maior do que 100'''
model.K = Var(tanques, range(0, 2), within = Binary, bounds=(0, 1))

model.Vol_n_atendido_sum = Var(tanques, within = Integers, bounds=(0, None))

'''R1: Garante que a quantidade de combustível j não atendida seja identificada corretamente e que apenas 1 opção seja atendida'''
model.R1 = ConstraintList()
for i in combustiveis:
    model.R1.add(expr=(sum(model.Bin_tipo[i,j] for j in tanques) == 1))

'''R2: Identifica o volume não atendido'''
model.R2 = ConstraintList()
for j in tanques:
    model.R2.add(expr=(model.Vol_atend[j] == sum(model.Vol[i, j]  for i in combustiveis)))

'''R3: Garante que Bin_tipo[i, j] seja igual a 1 quando Vol[i, j]>0 e 0 caso contrário'''
model.R3 = ConstraintList()
for j in tanques:
    for i in combustiveis:
        model.R3.add(expr=(model.Bin_tipo[i, j]*M >= model.Vol[i, j]))

'''R4: Limita a quantidade de combustível a capacidade máxima do tanque'''
model.R4 = ConstraintList()
for i in combustiveis:
    model.R4.add(expr=(sum(model.Vol[i, j] - M*(model.Bin_tipo[i, j]) for j in tanques) <= V_max_tanque[i]))

'''R5: Limita a quantidade máxima não atendida'''
model.R5 = ConstraintList()
for j in tanques:
    model.R5.add(expr=(model.Vol_n_atendido[j] <= Demanda_maxima_n_atendida[j]))
   
'''R6: Identifica o volume de combustível j não atendido'''
model.R6 = ConstraintList()
for j in tanques:
    model.R6.add(expr=(model.Vol_n_atendido[j] == Demanda_de_combustivel[j] - model.Vol_atend[j]))

'''R6: Se o volume não atendido<=100 => k0=1 e k1 =0'''
model.R7 = ConstraintList()
for j in tanques:
    model.R7.add(expr=(model.Vol_n_atendido[j] + M*(1-model.K[j, 0]-model.K[j, 1]) <= 100))

'''R8: Se o volume não atendido>=100 => k0= 0 e k1= 1'''
model.R8 = ConstraintList()
for j in tanques:
    model.R8.add(expr=(model.Vol_n_atendido[j] +M*(1-model.K[j, 1]-model.K[j, 0])) >= 100)

'''R9: Garante que K0 ou K1 seja 1, que ao menos 1 seja selecionado e que os 2 não sejam selecionados ao mesmo tempo'''
model.R9 = ConstraintList()
for j in tanques:
    model.R9.add(expr=(model.K[j, 0] + model.K[j, 1] == 1))

model.R10 = ConstraintList()
for j in tanques:
    model.R10.add(expr=(model.Vol_n_atendido_sum[j] >= model.Vol_n_atendido[j] * Penalidade[j] + M*(1-model.K[j, 0]-model.K[j, 1])))
    model.R10.add(expr=(model.Vol_n_atendido_sum[j] >= model.Vol_n_atendido[j] * 3 * Penalidade[j] + M*(1-model.K[j, 1]-model.K[j, 0])))


model.R11 = ConstraintList()
for j in tanques:
    model.R11.add(expr=(model.Vol_n_atendido[j] >= 0))


'''Define a função objetivo'''
model.obj = Objective(expr=(sum(model.Vol_n_atendido_sum[j] for j in tanques)),
                      sense = minimize
                      )

model.pprint()

optimizer = Highs()
# optimizer = SolverFactory('glpk')
# optimizer = SolverFactory('gurobi')

# habilite caso utilize o gplk para verificar o processo de otimização
# results = optimizer.solve(model, tee=True)
results = optimizer.solve(model)

valor_final_fo = model.obj.expr()
print(f'Valor final da função objetivo: {valor_final_fo}')

'''Acessando os valores das variáveis após a otimização'''
for i in combustiveis:
    for j in tanques:
        valor_bin_tipo = model.Bin_tipo[i, j].value
        if valor_bin_tipo>0:
            print(f'Bin_tipo[{i}, {j}] = {valor_bin_tipo}')

for i in combustiveis:
    for j in tanques:
        valor_vol = model.Vol[i, j].value
        if valor_vol>0:
            print(f'Vol[{i}, {j}] = {valor_vol}')

for j in tanques:
    valor_vol_atend = model.Vol_atend[j].value
    print(f'Vol_atend[{j}] = {valor_vol_atend}')

for j in tanques:
    valor_vol_n_atendido = model.Vol_n_atendido[j].value
    print(f'Vol_n_atendido[{j}] = {valor_vol_n_atendido}')

for j in tanques:
    for k in range(2):
        valor_k = model.K[j, k].value
        if valor_k>0:
            print(f'K[{j}, {k}] = {valor_k}')
