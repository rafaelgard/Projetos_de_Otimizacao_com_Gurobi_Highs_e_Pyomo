def imprime_resultados_geral(modelo):
    m = modelo
    print('=========================================')
    print('Quantidade de restrições: ', m.NumConstrs)
    print('Quantidade de variáveis: ', m.NumVars)
    print('=========================================')
    print('Status do modelo: ', m.Status)
    print('=========================================')
    print('Valor da função objetivo: ', m.ObjVal)
    print('=========================================')
    for v in m.getVars():  # m.getVars() retorna as variáveis do modelo
        if v.x != 0:
            print('%s %g' % (v.varName, v.x))
    print('=========================================')