\ Model problemadamarinha
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  200 XCB[0,0] + 200 XCB[0,1] + 300 XCB[0,2] + 300 XCB[1,0] + 400 XCB[1,1]
   + 220 XCB[1,2] + 300 XCB[2,0] + 400 XCB[2,1] + 250 XCB[2,2]
   + 2 Distancia_rota_NP[0] + 2 Distancia_rota_NP[1]
   + 2 Distancia_rota_NP[2] + 2 Distancia_rota_NP[3]
   + 2 Distancia_rota_NP[4] + 2 Distancia_rota_NP[5]
   + 2 Distancia_rota_NP[6] + 3 Distancia_rota_NG[0]
   + 3 Distancia_rota_NG[1] + 3 Distancia_rota_NG[2]
   + 3 Distancia_rota_NG[3] + 3 Distancia_rota_NG[4] + 5000 NP_utilizado[0]
   + 5000 NP_utilizado[1] + 5000 NP_utilizado[2] + 5000 NP_utilizado[3]
   + 5000 NP_utilizado[4] + 5000 NP_utilizado[5] + 5000 NP_utilizado[6]
   + 10000 NG_utilizado[0] + 10000 NG_utilizado[1] + 10000 NG_utilizado[2]
   + 10000 NG_utilizado[3] + 10000 NG_utilizado[4]
Subject To
 R0: - XCB[0,0] - XCB[1,0] - XCB[2,0] + Q_Treinada_Por_Base[0] = 0
 R1: - XCB[0,1] - XCB[1,1] - XCB[2,1] + Q_Treinada_Por_Base[1] = 0
 R2: - XCB[0,2] - XCB[1,2] - XCB[2,2] + Q_Treinada_Por_Base[2] = 0
 restricao_1: XCB[0,0] + XCB[0,1] + XCB[0,2] = 1000
 restricao_2: XCB[1,0] + XCB[1,1] + XCB[1,2] = 600
 restricao_3: XCB[2,0] + XCB[2,1] + XCB[2,2] = 700
 restricao_4: XCB[0,0] + XCB[1,0] + XCB[2,0] <= 1000
 restricao_5: XCB[0,1] + XCB[1,1] + XCB[2,1] <= 800
 restricao_6: XCB[0,2] + XCB[1,2] + XCB[2,2] <= 700
 R9: - XCB[0,0] - XCB[1,0] - XCB[2,0] + Q_Treinada_Por_Base[0] = 0
 R10: - XCB[0,1] - XCB[1,1] - XCB[2,1] + Q_Treinada_Por_Base[1] = 0
 R11: - XCB[0,2] - XCB[1,2] - XCB[2,2] + Q_Treinada_Por_Base[2] = 0
 R12: - QTD_NP[0,0,0] - QTD_NP[0,1,0] - QTD_NP[0,2,0] - QTD_NP[0,3,0]
   - QTD_NP[1,0,0] - QTD_NP[1,1,0] - QTD_NP[1,2,0] - QTD_NP[1,3,0]
   - QTD_NP[2,0,0] - QTD_NP[2,1,0] - QTD_NP[2,2,0] - QTD_NP[2,3,0]
   - QTD_NP[3,0,0] - QTD_NP[3,1,0] - QTD_NP[3,2,0] - QTD_NP[3,3,0]
   + QTD_transp_NP[0] = 0
 R13: - QTD_NP[0,0,1] - QTD_NP[0,1,1] - QTD_NP[0,2,1] - QTD_NP[0,3,1]
   - QTD_NP[1,0,1] - QTD_NP[1,1,1] - QTD_NP[1,2,1] - QTD_NP[1,3,1]
   - QTD_NP[2,0,1] - QTD_NP[2,1,1] - QTD_NP[2,2,1] - QTD_NP[2,3,1]
   - QTD_NP[3,0,1] - QTD_NP[3,1,1] - QTD_NP[3,2,1] - QTD_NP[3,3,1]
   + QTD_transp_NP[1] = 0
 R14: - QTD_NP[0,0,2] - QTD_NP[0,1,2] - QTD_NP[0,2,2] - QTD_NP[0,3,2]
   - QTD_NP[1,0,2] - QTD_NP[1,1,2] - QTD_NP[1,2,2] - QTD_NP[1,3,2]
   - QTD_NP[2,0,2] - QTD_NP[2,1,2] - QTD_NP[2,2,2] - QTD_NP[2,3,2]
   - QTD_NP[3,0,2] - QTD_NP[3,1,2] - QTD_NP[3,2,2] - QTD_NP[3,3,2]
   + QTD_transp_NP[2] = 0
 R15: - QTD_NP[0,0,3] - QTD_NP[0,1,3] - QTD_NP[0,2,3] - QTD_NP[0,3,3]
   - QTD_NP[1,0,3] - QTD_NP[1,1,3] - QTD_NP[1,2,3] - QTD_NP[1,3,3]
   - QTD_NP[2,0,3] - QTD_NP[2,1,3] - QTD_NP[2,2,3] - QTD_NP[2,3,3]
   - QTD_NP[3,0,3] - QTD_NP[3,1,3] - QTD_NP[3,2,3] - QTD_NP[3,3,3]
   + QTD_transp_NP[3] = 0
 R16: - QTD_NP[0,0,4] - QTD_NP[0,1,4] - QTD_NP[0,2,4] - QTD_NP[0,3,4]
   - QTD_NP[1,0,4] - QTD_NP[1,1,4] - QTD_NP[1,2,4] - QTD_NP[1,3,4]
   - QTD_NP[2,0,4] - QTD_NP[2,1,4] - QTD_NP[2,2,4] - QTD_NP[2,3,4]
   - QTD_NP[3,0,4] - QTD_NP[3,1,4] - QTD_NP[3,2,4] - QTD_NP[3,3,4]
   + QTD_transp_NP[4] = 0
 R17: - QTD_NP[0,0,5] - QTD_NP[0,1,5] - QTD_NP[0,2,5] - QTD_NP[0,3,5]
   - QTD_NP[1,0,5] - QTD_NP[1,1,5] - QTD_NP[1,2,5] - QTD_NP[1,3,5]
   - QTD_NP[2,0,5] - QTD_NP[2,1,5] - QTD_NP[2,2,5] - QTD_NP[2,3,5]
   - QTD_NP[3,0,5] - QTD_NP[3,1,5] - QTD_NP[3,2,5] - QTD_NP[3,3,5]
   + QTD_transp_NP[5] = 0
 R18: - QTD_NP[0,0,6] - QTD_NP[0,1,6] - QTD_NP[0,2,6] - QTD_NP[0,3,6]
   - QTD_NP[1,0,6] - QTD_NP[1,1,6] - QTD_NP[1,2,6] - QTD_NP[1,3,6]
   - QTD_NP[2,0,6] - QTD_NP[2,1,6] - QTD_NP[2,2,6] - QTD_NP[2,3,6]
   - QTD_NP[3,0,6] - QTD_NP[3,1,6] - QTD_NP[3,2,6] - QTD_NP[3,3,6]
   + QTD_transp_NP[6] = 0
 R19: - QTD_NG[0,0,0] - QTD_NG[0,1,0] - QTD_NG[0,2,0] - QTD_NG[0,3,0]
   - QTD_NG[1,0,0] - QTD_NG[1,1,0] - QTD_NG[1,2,0] - QTD_NG[1,3,0]
   - QTD_NG[2,0,0] - QTD_NG[2,1,0] - QTD_NG[2,2,0] - QTD_NG[2,3,0]
   - QTD_NG[3,0,0] - QTD_NG[3,1,0] - QTD_NG[3,2,0] - QTD_NG[3,3,0]
   + QTD_transp_NG[0] = 0
 R20: - QTD_NG[0,0,1] - QTD_NG[0,1,1] - QTD_NG[0,2,1] - QTD_NG[0,3,1]
   - QTD_NG[1,0,1] - QTD_NG[1,1,1] - QTD_NG[1,2,1] - QTD_NG[1,3,1]
   - QTD_NG[2,0,1] - QTD_NG[2,1,1] - QTD_NG[2,2,1] - QTD_NG[2,3,1]
   - QTD_NG[3,0,1] - QTD_NG[3,1,1] - QTD_NG[3,2,1] - QTD_NG[3,3,1]
   + QTD_transp_NG[1] = 0
 R21: - QTD_NG[0,0,2] - QTD_NG[0,1,2] - QTD_NG[0,2,2] - QTD_NG[0,3,2]
   - QTD_NG[1,0,2] - QTD_NG[1,1,2] - QTD_NG[1,2,2] - QTD_NG[1,3,2]
   - QTD_NG[2,0,2] - QTD_NG[2,1,2] - QTD_NG[2,2,2] - QTD_NG[2,3,2]
   - QTD_NG[3,0,2] - QTD_NG[3,1,2] - QTD_NG[3,2,2] - QTD_NG[3,3,2]
   + QTD_transp_NG[2] = 0
 R22: - QTD_NG[0,0,3] - QTD_NG[0,1,3] - QTD_NG[0,2,3] - QTD_NG[0,3,3]
   - QTD_NG[1,0,3] - QTD_NG[1,1,3] - QTD_NG[1,2,3] - QTD_NG[1,3,3]
   - QTD_NG[2,0,3] - QTD_NG[2,1,3] - QTD_NG[2,2,3] - QTD_NG[2,3,3]
   - QTD_NG[3,0,3] - QTD_NG[3,1,3] - QTD_NG[3,2,3] - QTD_NG[3,3,3]
   + QTD_transp_NG[3] = 0
 R23: - QTD_NG[0,0,4] - QTD_NG[0,1,4] - QTD_NG[0,2,4] - QTD_NG[0,3,4]
   - QTD_NG[1,0,4] - QTD_NG[1,1,4] - QTD_NG[1,2,4] - QTD_NG[1,3,4]
   - QTD_NG[2,0,4] - QTD_NG[2,1,4] - QTD_NG[2,2,4] - QTD_NG[2,3,4]
   - QTD_NG[3,0,4] - QTD_NG[3,1,4] - QTD_NG[3,2,4] - QTD_NG[3,3,4]
   + QTD_transp_NG[4] = 0
 R24: Rota_NP[0,0,0] + Rota_NP[0,1,0] + Rota_NP[0,2,0] + Rota_NP[0,3,0]
   + Rota_NP[1,0,0] + Rota_NP[1,1,0] + Rota_NP[1,2,0] + Rota_NP[1,3,0]
   + Rota_NP[2,0,0] + Rota_NP[2,1,0] + Rota_NP[2,2,0] + Rota_NP[2,3,0]
   + Rota_NP[3,0,0] + Rota_NP[3,1,0] + Rota_NP[3,2,0] + Rota_NP[3,3,0]
   <= 2
 R25: Rota_NP[0,0,1] + Rota_NP[0,1,1] + Rota_NP[0,2,1] + Rota_NP[0,3,1]
   + Rota_NP[1,0,1] + Rota_NP[1,1,1] + Rota_NP[1,2,1] + Rota_NP[1,3,1]
   + Rota_NP[2,0,1] + Rota_NP[2,1,1] + Rota_NP[2,2,1] + Rota_NP[2,3,1]
   + Rota_NP[3,0,1] + Rota_NP[3,1,1] + Rota_NP[3,2,1] + Rota_NP[3,3,1]
   <= 2
 R26: Rota_NP[0,0,2] + Rota_NP[0,1,2] + Rota_NP[0,2,2] + Rota_NP[0,3,2]
   + Rota_NP[1,0,2] + Rota_NP[1,1,2] + Rota_NP[1,2,2] + Rota_NP[1,3,2]
   + Rota_NP[2,0,2] + Rota_NP[2,1,2] + Rota_NP[2,2,2] + Rota_NP[2,3,2]
   + Rota_NP[3,0,2] + Rota_NP[3,1,2] + Rota_NP[3,2,2] + Rota_NP[3,3,2]
   <= 2
 R27: Rota_NP[0,0,3] + Rota_NP[0,1,3] + Rota_NP[0,2,3] + Rota_NP[0,3,3]
   + Rota_NP[1,0,3] + Rota_NP[1,1,3] + Rota_NP[1,2,3] + Rota_NP[1,3,3]
   + Rota_NP[2,0,3] + Rota_NP[2,1,3] + Rota_NP[2,2,3] + Rota_NP[2,3,3]
   + Rota_NP[3,0,3] + Rota_NP[3,1,3] + Rota_NP[3,2,3] + Rota_NP[3,3,3]
   <= 2
 R28: Rota_NP[0,0,4] + Rota_NP[0,1,4] + Rota_NP[0,2,4] + Rota_NP[0,3,4]
   + Rota_NP[1,0,4] + Rota_NP[1,1,4] + Rota_NP[1,2,4] + Rota_NP[1,3,4]
   + Rota_NP[2,0,4] + Rota_NP[2,1,4] + Rota_NP[2,2,4] + Rota_NP[2,3,4]
   + Rota_NP[3,0,4] + Rota_NP[3,1,4] + Rota_NP[3,2,4] + Rota_NP[3,3,4]
   <= 2
 R29: Rota_NP[0,0,5] + Rota_NP[0,1,5] + Rota_NP[0,2,5] + Rota_NP[0,3,5]
   + Rota_NP[1,0,5] + Rota_NP[1,1,5] + Rota_NP[1,2,5] + Rota_NP[1,3,5]
   + Rota_NP[2,0,5] + Rota_NP[2,1,5] + Rota_NP[2,2,5] + Rota_NP[2,3,5]
   + Rota_NP[3,0,5] + Rota_NP[3,1,5] + Rota_NP[3,2,5] + Rota_NP[3,3,5]
   <= 2
 R30: Rota_NP[0,0,6] + Rota_NP[0,1,6] + Rota_NP[0,2,6] + Rota_NP[0,3,6]
   + Rota_NP[1,0,6] + Rota_NP[1,1,6] + Rota_NP[1,2,6] + Rota_NP[1,3,6]
   + Rota_NP[2,0,6] + Rota_NP[2,1,6] + Rota_NP[2,2,6] + Rota_NP[2,3,6]
   + Rota_NP[3,0,6] + Rota_NP[3,1,6] + Rota_NP[3,2,6] + Rota_NP[3,3,6]
   <= 2
 R31: Rota_NG[0,0,0] + Rota_NG[0,1,0] + Rota_NG[0,2,0] + Rota_NG[0,3,0]
   + Rota_NG[1,0,0] + Rota_NG[1,1,0] + Rota_NG[1,2,0] + Rota_NG[1,3,0]
   + Rota_NG[2,0,0] + Rota_NG[2,1,0] + Rota_NG[2,2,0] + Rota_NG[2,3,0]
   + Rota_NG[3,0,0] + Rota_NG[3,1,0] + Rota_NG[3,2,0] + Rota_NG[3,3,0]
   <= 3
 R32: Rota_NG[0,0,1] + Rota_NG[0,1,1] + Rota_NG[0,2,1] + Rota_NG[0,3,1]
   + Rota_NG[1,0,1] + Rota_NG[1,1,1] + Rota_NG[1,2,1] + Rota_NG[1,3,1]
   + Rota_NG[2,0,1] + Rota_NG[2,1,1] + Rota_NG[2,2,1] + Rota_NG[2,3,1]
   + Rota_NG[3,0,1] + Rota_NG[3,1,1] + Rota_NG[3,2,1] + Rota_NG[3,3,1]
   <= 3
 R33: Rota_NG[0,0,2] + Rota_NG[0,1,2] + Rota_NG[0,2,2] + Rota_NG[0,3,2]
   + Rota_NG[1,0,2] + Rota_NG[1,1,2] + Rota_NG[1,2,2] + Rota_NG[1,3,2]
   + Rota_NG[2,0,2] + Rota_NG[2,1,2] + Rota_NG[2,2,2] + Rota_NG[2,3,2]
   + Rota_NG[3,0,2] + Rota_NG[3,1,2] + Rota_NG[3,2,2] + Rota_NG[3,3,2]
   <= 3
 R34: Rota_NG[0,0,3] + Rota_NG[0,1,3] + Rota_NG[0,2,3] + Rota_NG[0,3,3]
   + Rota_NG[1,0,3] + Rota_NG[1,1,3] + Rota_NG[1,2,3] + Rota_NG[1,3,3]
   + Rota_NG[2,0,3] + Rota_NG[2,1,3] + Rota_NG[2,2,3] + Rota_NG[2,3,3]
   + Rota_NG[3,0,3] + Rota_NG[3,1,3] + Rota_NG[3,2,3] + Rota_NG[3,3,3]
   <= 3
 R35: Rota_NG[0,0,4] + Rota_NG[0,1,4] + Rota_NG[0,2,4] + Rota_NG[0,3,4]
   + Rota_NG[1,0,4] + Rota_NG[1,1,4] + Rota_NG[1,2,4] + Rota_NG[1,3,4]
   + Rota_NG[2,0,4] + Rota_NG[2,1,4] + Rota_NG[2,2,4] + Rota_NG[2,3,4]
   + Rota_NG[3,0,4] + Rota_NG[3,1,4] + Rota_NG[3,2,4] + Rota_NG[3,3,4]
   <= 3
 R36: QTD_transp_NP[0] <= 200
 R37: QTD_transp_NP[1] <= 200
 R38: QTD_transp_NP[2] <= 200
 R39: QTD_transp_NP[3] <= 200
 R40: QTD_transp_NP[4] <= 200
 R41: QTD_transp_NP[5] <= 200
 R42: QTD_transp_NP[6] <= 200
 R43: QTD_transp_NG[0] <= 500
 R44: QTD_transp_NG[1] <= 500
 R45: QTD_transp_NG[2] <= 500
 R46: QTD_transp_NG[3] <= 500
 R47: QTD_transp_NG[4] <= 500
 R48: Rota_NP[0,0,0] = 0
 R49: Rota_NP[0,0,1] = 0
 R50: Rota_NP[0,0,2] = 0
 R51: Rota_NP[0,0,3] = 0
 R52: Rota_NP[0,0,4] = 0
 R53: Rota_NP[0,0,5] = 0
 R54: Rota_NP[0,0,6] = 0
 R55: Rota_NP[1,1,0] = 0
 R56: Rota_NP[1,1,1] = 0
 R57: Rota_NP[1,1,2] = 0
 R58: Rota_NP[1,1,3] = 0
 R59: Rota_NP[1,1,4] = 0
 R60: Rota_NP[1,1,5] = 0
 R61: Rota_NP[1,1,6] = 0
 R62: Rota_NP[2,2,0] = 0
 R63: Rota_NP[2,2,1] = 0
 R64: Rota_NP[2,2,2] = 0
 R65: Rota_NP[2,2,3] = 0
 R66: Rota_NP[2,2,4] = 0
 R67: Rota_NP[2,2,5] = 0
 R68: Rota_NP[2,2,6] = 0
 R69: Rota_NP[3,3,0] = 0
 R70: Rota_NP[3,3,1] = 0
 R71: Rota_NP[3,3,2] = 0
 R72: Rota_NP[3,3,3] = 0
 R73: Rota_NP[3,3,4] = 0
 R74: Rota_NP[3,3,5] = 0
 R75: Rota_NP[3,3,6] = 0
 R76: Rota_NG[0,0,0] = 0
 R77: Rota_NG[0,0,1] = 0
 R78: Rota_NG[0,0,2] = 0
 R79: Rota_NG[0,0,3] = 0
 R80: Rota_NG[0,0,4] = 0
 R81: Rota_NG[1,1,0] = 0
 R82: Rota_NG[1,1,1] = 0
 R83: Rota_NG[1,1,2] = 0
 R84: Rota_NG[1,1,3] = 0
 R85: Rota_NG[1,1,4] = 0
 R86: Rota_NG[2,2,0] = 0
 R87: Rota_NG[2,2,1] = 0
 R88: Rota_NG[2,2,2] = 0
 R89: Rota_NG[2,2,3] = 0
 R90: Rota_NG[2,2,4] = 0
 R91: Rota_NG[3,3,0] = 0
 R92: Rota_NG[3,3,1] = 0
 R93: Rota_NG[3,3,2] = 0
 R94: Rota_NG[3,3,3] = 0
 R95: Rota_NG[3,3,4] = 0
 R96: QTD_NP[0,0,0] = 0
 R97: QTD_NP[0,0,1] = 0
 R98: QTD_NP[0,0,2] = 0
 R99: QTD_NP[0,0,3] = 0
 R100: QTD_NP[0,0,4] = 0
 R101: QTD_NP[0,0,5] = 0
 R102: QTD_NP[0,0,6] = 0
 R103: QTD_NP[1,1,0] = 0
 R104: QTD_NP[1,1,1] = 0
 R105: QTD_NP[1,1,2] = 0
 R106: QTD_NP[1,1,3] = 0
 R107: QTD_NP[1,1,4] = 0
 R108: QTD_NP[1,1,5] = 0
 R109: QTD_NP[1,1,6] = 0
 R110: QTD_NP[2,2,0] = 0
 R111: QTD_NP[2,2,1] = 0
 R112: QTD_NP[2,2,2] = 0
 R113: QTD_NP[2,2,3] = 0
 R114: QTD_NP[2,2,4] = 0
 R115: QTD_NP[2,2,5] = 0
 R116: QTD_NP[2,2,6] = 0
 R117: QTD_NP[3,3,0] = 0
 R118: QTD_NP[3,3,1] = 0
 R119: QTD_NP[3,3,2] = 0
 R120: QTD_NP[3,3,3] = 0
 R121: QTD_NP[3,3,4] = 0
 R122: QTD_NP[3,3,5] = 0
 R123: QTD_NP[3,3,6] = 0
 R124: QTD_NG[0,0,0] = 0
 R125: QTD_NG[0,0,1] = 0
 R126: QTD_NG[0,0,2] = 0
 R127: QTD_NG[0,0,3] = 0
 R128: QTD_NG[0,0,4] = 0
 R129: QTD_NG[1,1,0] = 0
 R130: QTD_NG[1,1,1] = 0
 R131: QTD_NG[1,1,2] = 0
 R132: QTD_NG[1,1,3] = 0
 R133: QTD_NG[1,1,4] = 0
 R134: QTD_NG[2,2,0] = 0
 R135: QTD_NG[2,2,1] = 0
 R136: QTD_NG[2,2,2] = 0
 R137: QTD_NG[2,2,3] = 0
 R138: QTD_NG[2,2,4] = 0
 R139: QTD_NG[3,3,0] = 0
 R140: QTD_NG[3,3,1] = 0
 R141: QTD_NG[3,3,2] = 0
 R142: QTD_NG[3,3,3] = 0
 R143: QTD_NG[3,3,4] = 0
 R144: Rota_NP[2,1,0] = 0
 R145: Rota_NP[2,1,1] = 0
 R146: Rota_NP[2,1,2] = 0
 R147: Rota_NP[2,1,3] = 0
 R148: Rota_NP[2,1,4] = 0
 R149: Rota_NP[2,1,5] = 0
 R150: Rota_NP[2,1,6] = 0
 R151: Rota_NP[3,1,0] = 0
 R152: Rota_NP[3,1,1] = 0
 R153: Rota_NP[3,1,2] = 0
 R154: Rota_NP[3,1,3] = 0
 R155: Rota_NP[3,1,4] = 0
 R156: Rota_NP[3,1,5] = 0
 R157: Rota_NP[3,1,6] = 0
 R158: Rota_NP[3,2,0] = 0
 R159: Rota_NP[3,2,1] = 0
 R160: Rota_NP[3,2,2] = 0
 R161: Rota_NP[3,2,3] = 0
 R162: Rota_NP[3,2,4] = 0
 R163: Rota_NP[3,2,5] = 0
 R164: Rota_NP[3,2,6] = 0
 R165: Rota_NG[2,1,0] = 0
 R166: Rota_NG[2,1,1] = 0
 R167: Rota_NG[2,1,2] = 0
 R168: Rota_NG[2,1,3] = 0
 R169: Rota_NG[2,1,4] = 0
 R170: Rota_NG[3,1,0] = 0
 R171: Rota_NG[3,1,1] = 0
 R172: Rota_NG[3,1,2] = 0
 R173: Rota_NG[3,1,3] = 0
 R174: Rota_NG[3,1,4] = 0
 R175: Rota_NG[3,2,0] = 0
 R176: Rota_NG[3,2,1] = 0
 R177: Rota_NG[3,2,2] = 0
 R178: Rota_NG[3,2,3] = 0
 R179: Rota_NG[3,2,4] = 0
 R180: QTD_NP[2,1,0] = 0
 R181: QTD_NP[2,1,1] = 0
 R182: QTD_NP[2,1,2] = 0
 R183: QTD_NP[2,1,3] = 0
 R184: QTD_NP[2,1,4] = 0
 R185: QTD_NP[2,1,5] = 0
 R186: QTD_NP[2,1,6] = 0
 R187: QTD_NP[3,1,0] = 0
 R188: QTD_NP[3,1,1] = 0
 R189: QTD_NP[3,1,2] = 0
 R190: QTD_NP[3,1,3] = 0
 R191: QTD_NP[3,1,4] = 0
 R192: QTD_NP[3,1,5] = 0
 R193: QTD_NP[3,1,6] = 0
 R194: QTD_NP[3,2,0] = 0
 R195: QTD_NP[3,2,1] = 0
 R196: QTD_NP[3,2,2] = 0
 R197: QTD_NP[3,2,3] = 0
 R198: QTD_NP[3,2,4] = 0
 R199: QTD_NP[3,2,5] = 0
 R200: QTD_NP[3,2,6] = 0
 R201: QTD_NG[2,1,0] = 0
 R202: QTD_NG[2,1,1] = 0
 R203: QTD_NG[2,1,2] = 0
 R204: QTD_NG[2,1,3] = 0
 R205: QTD_NG[2,1,4] = 0
 R206: QTD_NG[3,1,0] = 0
 R207: QTD_NG[3,1,1] = 0
 R208: QTD_NG[3,1,2] = 0
 R209: QTD_NG[3,1,3] = 0
 R210: QTD_NG[3,1,4] = 0
 R211: QTD_NG[3,2,0] = 0
 R212: QTD_NG[3,2,1] = 0
 R213: QTD_NG[3,2,2] = 0
 R214: QTD_NG[3,2,3] = 0
 R215: QTD_NG[3,2,4] = 0
 R216: Q_Treinada_Por_Base[0] - QTD_transp_por_base_NP[0]
   - QTD_transp_por_base_NG[0] = 0
 R217: Q_Treinada_Por_Base[1] - QTD_transp_por_base_NP[1]
   - QTD_transp_por_base_NG[1] = 0
 R218: Q_Treinada_Por_Base[2] - QTD_transp_por_base_NP[2]
   - QTD_transp_por_base_NG[2] = 0
 R219: - 185 Rota_NP[0,1,0] - 185 Rota_NP[0,1,1] - 185 Rota_NP[0,1,2]
   - 185 Rota_NP[0,1,3] - 185 Rota_NP[0,1,4] - 185 Rota_NP[0,1,5]
   - 185 Rota_NP[0,1,6] - 185 Rota_NP[1,0,0] - 185 Rota_NP[1,0,1]
   - 185 Rota_NP[1,0,2] - 185 Rota_NP[1,0,3] - 185 Rota_NP[1,0,4]
   - 185 Rota_NP[1,0,5] - 185 Rota_NP[1,0,6] + Distancia_rota_NP[0] = 0
 R220: - 171.6666666666667 Rota_NP[0,1,0]
   - 171.6666666666667 Rota_NP[0,1,1] - 171.6666666666667 Rota_NP[0,1,2]
   - 171.6666666666667 Rota_NP[0,1,3] - 171.6666666666667 Rota_NP[0,1,4]
   - 171.6666666666667 Rota_NP[0,1,5] - 171.6666666666667 Rota_NP[0,1,6]
   - 171.6666666666667 Rota_NP[1,2,0] - 171.6666666666667 Rota_NP[1,2,1]
   - 171.6666666666667 Rota_NP[1,2,2] - 171.6666666666667 Rota_NP[1,2,3]
   - 171.6666666666667 Rota_NP[1,2,4] - 171.6666666666667 Rota_NP[1,2,5]
   - 171.6666666666667 Rota_NP[1,2,6] - 171.6666666666667 Rota_NP[2,0,0]
   - 171.6666666666667 Rota_NP[2,0,1] - 171.6666666666667 Rota_NP[2,0,2]
   - 171.6666666666667 Rota_NP[2,0,3] - 171.6666666666667 Rota_NP[2,0,4]
   - 171.6666666666667 Rota_NP[2,0,5] - 171.6666666666667 Rota_NP[2,0,6]
   + Distancia_rota_NP[1] = 0
 R221: - 221.6666666666667 Rota_NP[0,2,0]
   - 221.6666666666667 Rota_NP[0,2,1] - 221.6666666666667 Rota_NP[0,2,2]
   - 221.6666666666667 Rota_NP[0,2,3] - 221.6666666666667 Rota_NP[0,2,4]
   - 221.6666666666667 Rota_NP[0,2,5] - 221.6666666666667 Rota_NP[0,2,6]
   - 221.6666666666667 Rota_NP[2,3,0] - 221.6666666666667 Rota_NP[2,3,1]
   - 221.6666666666667 Rota_NP[2,3,2] - 221.6666666666667 Rota_NP[2,3,3]
   - 221.6666666666667 Rota_NP[2,3,4] - 221.6666666666667 Rota_NP[2,3,5]
   - 221.6666666666667 Rota_NP[2,3,6] - 221.6666666666667 Rota_NP[3,0,0]
   - 221.6666666666667 Rota_NP[3,0,1] - 221.6666666666667 Rota_NP[3,0,2]
   - 221.6666666666667 Rota_NP[3,0,3] - 221.6666666666667 Rota_NP[3,0,4]
   - 221.6666666666667 Rota_NP[3,0,5] - 221.6666666666667 Rota_NP[3,0,6]
   + Distancia_rota_NP[2] = 0
 R222: - 230 Rota_NP[0,2,0] - 230 Rota_NP[0,2,1] - 230 Rota_NP[0,2,2]
   - 230 Rota_NP[0,2,3] - 230 Rota_NP[0,2,4] - 230 Rota_NP[0,2,5]
   - 230 Rota_NP[0,2,6] - 230 Rota_NP[2,0,0] - 230 Rota_NP[2,0,1]
   - 230 Rota_NP[2,0,2] - 230 Rota_NP[2,0,3] - 230 Rota_NP[2,0,4]
   - 230 Rota_NP[2,0,5] - 230 Rota_NP[2,0,6] + Distancia_rota_NP[3] = 0
 R223: - 300 Rota_NP[0,3,0] - 300 Rota_NP[0,3,1] - 300 Rota_NP[0,3,2]
   - 300 Rota_NP[0,3,3] - 300 Rota_NP[0,3,4] - 300 Rota_NP[0,3,5]
   - 300 Rota_NP[0,3,6] - 300 Rota_NP[3,0,0] - 300 Rota_NP[3,0,1]
   - 300 Rota_NP[3,0,2] - 300 Rota_NP[3,0,3] - 300 Rota_NP[3,0,4]
   - 300 Rota_NP[3,0,5] - 300 Rota_NP[3,0,6] + Distancia_rota_NP[4] = 0
 R224: - 213.3333333333333 Rota_NP[0,1,0]
   - 213.3333333333333 Rota_NP[0,1,1] - 213.3333333333333 Rota_NP[0,1,2]
   - 213.3333333333333 Rota_NP[0,1,3] - 213.3333333333333 Rota_NP[0,1,4]
   - 213.3333333333333 Rota_NP[0,1,5] - 213.3333333333333 Rota_NP[0,1,6]
   - 213.3333333333333 Rota_NP[1,3,0] - 213.3333333333333 Rota_NP[1,3,1]
   - 213.3333333333333 Rota_NP[1,3,2] - 213.3333333333333 Rota_NP[1,3,3]
   - 213.3333333333333 Rota_NP[1,3,4] - 213.3333333333333 Rota_NP[1,3,5]
   - 213.3333333333333 Rota_NP[1,3,6] - 213.3333333333333 Rota_NP[3,0,0]
   - 213.3333333333333 Rota_NP[3,0,1] - 213.3333333333333 Rota_NP[3,0,2]
   - 213.3333333333333 Rota_NP[3,0,3] - 213.3333333333333 Rota_NP[3,0,4]
   - 213.3333333333333 Rota_NP[3,0,5] - 213.3333333333333 Rota_NP[3,0,6]
   + Distancia_rota_NP[5] = 0
 R225: - 185 Rota_NG[0,1,0] - 185 Rota_NG[0,1,1] - 185 Rota_NG[0,1,2]
   - 185 Rota_NG[0,1,3] - 185 Rota_NG[0,1,4] - 185 Rota_NG[1,0,0]
   - 185 Rota_NG[1,0,1] - 185 Rota_NG[1,0,2] - 185 Rota_NG[1,0,3]
   - 185 Rota_NG[1,0,4] + Distancia_rota_NG[0] = 0
 R226: - 171.6666666666667 Rota_NG[0,1,0]
   - 171.6666666666667 Rota_NG[0,1,1] - 171.6666666666667 Rota_NG[0,1,2]
   - 171.6666666666667 Rota_NG[0,1,3] - 171.6666666666667 Rota_NG[0,1,4]
   - 171.6666666666667 Rota_NG[1,2,0] - 171.6666666666667 Rota_NG[1,2,1]
   - 171.6666666666667 Rota_NG[1,2,2] - 171.6666666666667 Rota_NG[1,2,3]
   - 171.6666666666667 Rota_NG[1,2,4] - 171.6666666666667 Rota_NG[2,0,0]
   - 171.6666666666667 Rota_NG[2,0,1] - 171.6666666666667 Rota_NG[2,0,2]
   - 171.6666666666667 Rota_NG[2,0,3] - 171.6666666666667 Rota_NG[2,0,4]
   + Distancia_rota_NG[1] = 0
 R227: - 221.6666666666667 Rota_NG[0,2,0]
   - 221.6666666666667 Rota_NG[0,2,1] - 221.6666666666667 Rota_NG[0,2,2]
   - 221.6666666666667 Rota_NG[0,2,3] - 221.6666666666667 Rota_NG[0,2,4]
   - 221.6666666666667 Rota_NG[2,3,0] - 221.6666666666667 Rota_NG[2,3,1]
   - 221.6666666666667 Rota_NG[2,3,2] - 221.6666666666667 Rota_NG[2,3,3]
   - 221.6666666666667 Rota_NG[2,3,4] - 221.6666666666667 Rota_NG[3,0,0]
   - 221.6666666666667 Rota_NG[3,0,1] - 221.6666666666667 Rota_NG[3,0,2]
   - 221.6666666666667 Rota_NG[3,0,3] - 221.6666666666667 Rota_NG[3,0,4]
   + Distancia_rota_NG[2] = 0
 R228: - 230 Rota_NG[0,2,0] - 230 Rota_NG[0,2,1] - 230 Rota_NG[0,2,2]
   - 230 Rota_NG[0,2,3] - 230 Rota_NG[0,2,4] - 230 Rota_NG[2,0,0]
   - 230 Rota_NG[2,0,1] - 230 Rota_NG[2,0,2] - 230 Rota_NG[2,0,3]
   - 230 Rota_NG[2,0,4] + Distancia_rota_NG[3] = 0
 R229: - 300 Rota_NG[0,3,0] - 300 Rota_NG[0,3,1] - 300 Rota_NG[0,3,2]
   - 300 Rota_NG[0,3,3] - 300 Rota_NG[0,3,4] - 300 Rota_NG[3,0,0]
   - 300 Rota_NG[3,0,1] - 300 Rota_NG[3,0,2] - 300 Rota_NG[3,0,3]
   - 300 Rota_NG[3,0,4] + Distancia_rota_NG[4] = 0
 R230: - 213.3333333333333 Rota_NG[0,1,0]
   - 213.3333333333333 Rota_NG[0,1,1] - 213.3333333333333 Rota_NG[0,1,2]
   - 213.3333333333333 Rota_NG[0,1,3] - 213.3333333333333 Rota_NG[0,1,4]
   - 213.3333333333333 Rota_NG[1,3,0] - 213.3333333333333 Rota_NG[1,3,1]
   - 213.3333333333333 Rota_NG[1,3,2] - 213.3333333333333 Rota_NG[1,3,3]
   - 213.3333333333333 Rota_NG[1,3,4] - 213.3333333333333 Rota_NG[3,0,0]
   - 213.3333333333333 Rota_NG[3,0,1] - 213.3333333333333 Rota_NG[3,0,2]
   - 213.3333333333333 Rota_NG[3,0,3] - 213.3333333333333 Rota_NG[3,0,4]
   + Distancia_rota_NG[5] = 0
 R231: - 180 Rota_NG[0,1,0] - 180 Rota_NG[0,1,1] - 180 Rota_NG[0,1,2]
   - 180 Rota_NG[0,1,3] - 180 Rota_NG[0,1,4] - 180 Rota_NG[1,2,0]
   - 180 Rota_NG[1,2,1] - 180 Rota_NG[1,2,2] - 180 Rota_NG[1,2,3]
   - 180 Rota_NG[1,2,4] - 180 Rota_NG[2,3,0] - 180 Rota_NG[2,3,1]
   - 180 Rota_NG[2,3,2] - 180 Rota_NG[2,3,3] - 180 Rota_NG[2,3,4]
   - 180 Rota_NG[3,3,0] - 180 Rota_NG[3,3,1] - 180 Rota_NG[3,3,2]
   - 180 Rota_NG[3,3,3] - 180 Rota_NG[3,3,4] + Distancia_rota_NG[6] = 0
 R232: - QTD_transp_NP[0] + 1e+06 NP_utilizado[0] >= 0
 R233: - QTD_transp_NP[1] + 1e+06 NP_utilizado[1] >= 0
 R234: - QTD_transp_NP[2] + 1e+06 NP_utilizado[2] >= 0
 R235: - QTD_transp_NP[3] + 1e+06 NP_utilizado[3] >= 0
 R236: - QTD_transp_NP[4] + 1e+06 NP_utilizado[4] >= 0
 R237: - QTD_transp_NP[5] + 1e+06 NP_utilizado[5] >= 0
 R238: - QTD_transp_NP[6] + 1e+06 NP_utilizado[6] >= 0
 R239: - QTD_transp_NG[0] + 1e+06 NG_utilizado[0] >= 0
 R240: - QTD_transp_NG[1] + 1e+06 NG_utilizado[1] >= 0
 R241: - QTD_transp_NG[2] + 1e+06 NG_utilizado[2] >= 0
 R242: - QTD_transp_NG[3] + 1e+06 NG_utilizado[3] >= 0
 R243: - QTD_transp_NG[4] + 1e+06 NG_utilizado[4] >= 0
 qc0: QTD_transp_por_base_NP[0] + [ - Rota_NP[0,1,0] * QTD_NP[0,1,0]
   - Rota_NP[0,1,1] * QTD_NP[0,1,1] - Rota_NP[0,1,2] * QTD_NP[0,1,2]
   - Rota_NP[0,1,3] * QTD_NP[0,1,3] - Rota_NP[0,1,4] * QTD_NP[0,1,4]
   - Rota_NP[0,1,5] * QTD_NP[0,1,5] - Rota_NP[0,1,6] * QTD_NP[0,1,6]
   - Rota_NP[1,1,0] * QTD_NP[1,1,0] - Rota_NP[1,1,1] * QTD_NP[1,1,1]
   - Rota_NP[1,1,2] * QTD_NP[1,1,2] - Rota_NP[1,1,3] * QTD_NP[1,1,3]
   - Rota_NP[1,1,4] * QTD_NP[1,1,4] - Rota_NP[1,1,5] * QTD_NP[1,1,5]
   - Rota_NP[1,1,6] * QTD_NP[1,1,6] - Rota_NP[2,1,0] * QTD_NP[2,1,0]
   - Rota_NP[2,1,1] * QTD_NP[2,1,1] - Rota_NP[2,1,2] * QTD_NP[2,1,2]
   - Rota_NP[2,1,3] * QTD_NP[2,1,3] - Rota_NP[2,1,4] * QTD_NP[2,1,4]
   - Rota_NP[2,1,5] * QTD_NP[2,1,5] - Rota_NP[2,1,6] * QTD_NP[2,1,6]
   - Rota_NP[3,1,0] * QTD_NP[3,1,0] - Rota_NP[3,1,1] * QTD_NP[3,1,1]
   - Rota_NP[3,1,2] * QTD_NP[3,1,2] - Rota_NP[3,1,3] * QTD_NP[3,1,3]
   - Rota_NP[3,1,4] * QTD_NP[3,1,4] - Rota_NP[3,1,5] * QTD_NP[3,1,5]
   - Rota_NP[3,1,6] * QTD_NP[3,1,6] ] = 0
 qc1: QTD_transp_por_base_NP[1] + [ - Rota_NP[0,2,0] * QTD_NP[0,2,0]
   - Rota_NP[0,2,1] * QTD_NP[0,2,1] - Rota_NP[0,2,2] * QTD_NP[0,2,2]
   - Rota_NP[0,2,3] * QTD_NP[0,2,3] - Rota_NP[0,2,4] * QTD_NP[0,2,4]
   - Rota_NP[0,2,5] * QTD_NP[0,2,5] - Rota_NP[0,2,6] * QTD_NP[0,2,6]
   - Rota_NP[1,2,0] * QTD_NP[1,2,0] - Rota_NP[1,2,1] * QTD_NP[1,2,1]
   - Rota_NP[1,2,2] * QTD_NP[1,2,2] - Rota_NP[1,2,3] * QTD_NP[1,2,3]
   - Rota_NP[1,2,4] * QTD_NP[1,2,4] - Rota_NP[1,2,5] * QTD_NP[1,2,5]
   - Rota_NP[1,2,6] * QTD_NP[1,2,6] - Rota_NP[2,2,0] * QTD_NP[2,2,0]
   - Rota_NP[2,2,1] * QTD_NP[2,2,1] - Rota_NP[2,2,2] * QTD_NP[2,2,2]
   - Rota_NP[2,2,3] * QTD_NP[2,2,3] - Rota_NP[2,2,4] * QTD_NP[2,2,4]
   - Rota_NP[2,2,5] * QTD_NP[2,2,5] - Rota_NP[2,2,6] * QTD_NP[2,2,6]
   - Rota_NP[3,2,0] * QTD_NP[3,2,0] - Rota_NP[3,2,1] * QTD_NP[3,2,1]
   - Rota_NP[3,2,2] * QTD_NP[3,2,2] - Rota_NP[3,2,3] * QTD_NP[3,2,3]
   - Rota_NP[3,2,4] * QTD_NP[3,2,4] - Rota_NP[3,2,5] * QTD_NP[3,2,5]
   - Rota_NP[3,2,6] * QTD_NP[3,2,6] ] = 0
 qc2: QTD_transp_por_base_NP[2] + [ - Rota_NP[0,3,0] * QTD_NP[0,3,0]
   - Rota_NP[0,3,1] * QTD_NP[0,3,1] - Rota_NP[0,3,2] * QTD_NP[0,3,2]
   - Rota_NP[0,3,3] * QTD_NP[0,3,3] - Rota_NP[0,3,4] * QTD_NP[0,3,4]
   - Rota_NP[0,3,5] * QTD_NP[0,3,5] - Rota_NP[0,3,6] * QTD_NP[0,3,6]
   - Rota_NP[1,3,0] * QTD_NP[1,3,0] - Rota_NP[1,3,1] * QTD_NP[1,3,1]
   - Rota_NP[1,3,2] * QTD_NP[1,3,2] - Rota_NP[1,3,3] * QTD_NP[1,3,3]
   - Rota_NP[1,3,4] * QTD_NP[1,3,4] - Rota_NP[1,3,5] * QTD_NP[1,3,5]
   - Rota_NP[1,3,6] * QTD_NP[1,3,6] - Rota_NP[2,3,0] * QTD_NP[2,3,0]
   - Rota_NP[2,3,1] * QTD_NP[2,3,1] - Rota_NP[2,3,2] * QTD_NP[2,3,2]
   - Rota_NP[2,3,3] * QTD_NP[2,3,3] - Rota_NP[2,3,4] * QTD_NP[2,3,4]
   - Rota_NP[2,3,5] * QTD_NP[2,3,5] - Rota_NP[2,3,6] * QTD_NP[2,3,6]
   - Rota_NP[3,3,0] * QTD_NP[3,3,0] - Rota_NP[3,3,1] * QTD_NP[3,3,1]
   - Rota_NP[3,3,2] * QTD_NP[3,3,2] - Rota_NP[3,3,3] * QTD_NP[3,3,3]
   - Rota_NP[3,3,4] * QTD_NP[3,3,4] - Rota_NP[3,3,5] * QTD_NP[3,3,5]
   - Rota_NP[3,3,6] * QTD_NP[3,3,6] ] = 0
 qc3: QTD_transp_por_base_NG[0] + [ - Rota_NG[0,1,0] * QTD_NG[0,1,0]
   - Rota_NG[0,1,1] * QTD_NG[0,1,1] - Rota_NG[0,1,2] * QTD_NG[0,1,2]
   - Rota_NG[0,1,3] * QTD_NG[0,1,3] - Rota_NG[0,1,4] * QTD_NG[0,1,4]
   - Rota_NG[1,1,0] * QTD_NG[1,1,0] - Rota_NG[1,1,1] * QTD_NG[1,1,1]
   - Rota_NG[1,1,2] * QTD_NG[1,1,2] - Rota_NG[1,1,3] * QTD_NG[1,1,3]
   - Rota_NG[1,1,4] * QTD_NG[1,1,4] - Rota_NG[2,1,0] * QTD_NG[2,1,0]
   - Rota_NG[2,1,1] * QTD_NG[2,1,1] - Rota_NG[2,1,2] * QTD_NG[2,1,2]
   - Rota_NG[2,1,3] * QTD_NG[2,1,3] - Rota_NG[2,1,4] * QTD_NG[2,1,4]
   - Rota_NG[3,1,0] * QTD_NG[3,1,0] - Rota_NG[3,1,1] * QTD_NG[3,1,1]
   - Rota_NG[3,1,2] * QTD_NG[3,1,2] - Rota_NG[3,1,3] * QTD_NG[3,1,3]
   - Rota_NG[3,1,4] * QTD_NG[3,1,4] ] = 0
 qc4: QTD_transp_por_base_NG[1] + [ - Rota_NG[0,2,0] * QTD_NG[0,2,0]
   - Rota_NG[0,2,1] * QTD_NG[0,2,1] - Rota_NG[0,2,2] * QTD_NG[0,2,2]
   - Rota_NG[0,2,3] * QTD_NG[0,2,3] - Rota_NG[0,2,4] * QTD_NG[0,2,4]
   - Rota_NG[1,2,0] * QTD_NG[1,2,0] - Rota_NG[1,2,1] * QTD_NG[1,2,1]
   - Rota_NG[1,2,2] * QTD_NG[1,2,2] - Rota_NG[1,2,3] * QTD_NG[1,2,3]
   - Rota_NG[1,2,4] * QTD_NG[1,2,4] - Rota_NG[2,2,0] * QTD_NG[2,2,0]
   - Rota_NG[2,2,1] * QTD_NG[2,2,1] - Rota_NG[2,2,2] * QTD_NG[2,2,2]
   - Rota_NG[2,2,3] * QTD_NG[2,2,3] - Rota_NG[2,2,4] * QTD_NG[2,2,4]
   - Rota_NG[3,2,0] * QTD_NG[3,2,0] - Rota_NG[3,2,1] * QTD_NG[3,2,1]
   - Rota_NG[3,2,2] * QTD_NG[3,2,2] - Rota_NG[3,2,3] * QTD_NG[3,2,3]
   - Rota_NG[3,2,4] * QTD_NG[3,2,4] ] = 0
 qc5: QTD_transp_por_base_NG[2] + [ - Rota_NG[0,3,0] * QTD_NG[0,3,0]
   - Rota_NG[0,3,1] * QTD_NG[0,3,1] - Rota_NG[0,3,2] * QTD_NG[0,3,2]
   - Rota_NG[0,3,3] * QTD_NG[0,3,3] - Rota_NG[0,3,4] * QTD_NG[0,3,4]
   - Rota_NG[1,3,0] * QTD_NG[1,3,0] - Rota_NG[1,3,1] * QTD_NG[1,3,1]
   - Rota_NG[1,3,2] * QTD_NG[1,3,2] - Rota_NG[1,3,3] * QTD_NG[1,3,3]
   - Rota_NG[1,3,4] * QTD_NG[1,3,4] - Rota_NG[2,3,0] * QTD_NG[2,3,0]
   - Rota_NG[2,3,1] * QTD_NG[2,3,1] - Rota_NG[2,3,2] * QTD_NG[2,3,2]
   - Rota_NG[2,3,3] * QTD_NG[2,3,3] - Rota_NG[2,3,4] * QTD_NG[2,3,4]
   - Rota_NG[3,3,0] * QTD_NG[3,3,0] - Rota_NG[3,3,1] * QTD_NG[3,3,1]
   - Rota_NG[3,3,2] * QTD_NG[3,3,2] - Rota_NG[3,3,3] * QTD_NG[3,3,3]
   - Rota_NG[3,3,4] * QTD_NG[3,3,4] ] = 0
 qc6: - Q_Treinada_Por_Base[0] - Q_Treinada_Por_Base[1]
   - Q_Treinada_Por_Base[2] + [ QTD_transp_NP[0] * NP_utilizado[0]
   + QTD_transp_NP[1] * NP_utilizado[1]
   + QTD_transp_NP[2] * NP_utilizado[2]
   + QTD_transp_NP[3] * NP_utilizado[3]
   + QTD_transp_NP[4] * NP_utilizado[4]
   + QTD_transp_NP[5] * NP_utilizado[5]
   + QTD_transp_NP[6] * NP_utilizado[6]
   + QTD_transp_NG[0] * NG_utilizado[0]
   + QTD_transp_NG[1] * NG_utilizado[1]
   + QTD_transp_NG[2] * NG_utilizado[2]
   + QTD_transp_NG[3] * NG_utilizado[3]
   + QTD_transp_NG[4] * NG_utilizado[4] ] = 0
Bounds
 QTD_transp_NP[0] <= 200
 QTD_transp_NP[1] <= 200
 QTD_transp_NP[2] <= 200
 QTD_transp_NP[3] <= 200
 QTD_transp_NP[4] <= 200
 QTD_transp_NP[5] <= 200
 QTD_transp_NP[6] <= 200
 QTD_transp_NG[0] <= 500
 QTD_transp_NG[1] <= 500
 QTD_transp_NG[2] <= 500
 QTD_transp_NG[3] <= 500
 QTD_transp_NG[4] <= 500
Binaries
 Rota_NP[0,0,0] Rota_NP[0,0,1] Rota_NP[0,0,2] Rota_NP[0,0,3] Rota_NP[0,0,4]
 Rota_NP[0,0,5] Rota_NP[0,0,6] Rota_NP[0,1,0] Rota_NP[0,1,1] Rota_NP[0,1,2]
 Rota_NP[0,1,3] Rota_NP[0,1,4] Rota_NP[0,1,5] Rota_NP[0,1,6] Rota_NP[0,2,0]
 Rota_NP[0,2,1] Rota_NP[0,2,2] Rota_NP[0,2,3] Rota_NP[0,2,4] Rota_NP[0,2,5]
 Rota_NP[0,2,6] Rota_NP[0,3,0] Rota_NP[0,3,1] Rota_NP[0,3,2] Rota_NP[0,3,3]
 Rota_NP[0,3,4] Rota_NP[0,3,5] Rota_NP[0,3,6] Rota_NP[1,0,0] Rota_NP[1,0,1]
 Rota_NP[1,0,2] Rota_NP[1,0,3] Rota_NP[1,0,4] Rota_NP[1,0,5] Rota_NP[1,0,6]
 Rota_NP[1,1,0] Rota_NP[1,1,1] Rota_NP[1,1,2] Rota_NP[1,1,3] Rota_NP[1,1,4]
 Rota_NP[1,1,5] Rota_NP[1,1,6] Rota_NP[1,2,0] Rota_NP[1,2,1] Rota_NP[1,2,2]
 Rota_NP[1,2,3] Rota_NP[1,2,4] Rota_NP[1,2,5] Rota_NP[1,2,6] Rota_NP[1,3,0]
 Rota_NP[1,3,1] Rota_NP[1,3,2] Rota_NP[1,3,3] Rota_NP[1,3,4] Rota_NP[1,3,5]
 Rota_NP[1,3,6] Rota_NP[2,0,0] Rota_NP[2,0,1] Rota_NP[2,0,2] Rota_NP[2,0,3]
 Rota_NP[2,0,4] Rota_NP[2,0,5] Rota_NP[2,0,6] Rota_NP[2,1,0] Rota_NP[2,1,1]
 Rota_NP[2,1,2] Rota_NP[2,1,3] Rota_NP[2,1,4] Rota_NP[2,1,5] Rota_NP[2,1,6]
 Rota_NP[2,2,0] Rota_NP[2,2,1] Rota_NP[2,2,2] Rota_NP[2,2,3] Rota_NP[2,2,4]
 Rota_NP[2,2,5] Rota_NP[2,2,6] Rota_NP[2,3,0] Rota_NP[2,3,1] Rota_NP[2,3,2]
 Rota_NP[2,3,3] Rota_NP[2,3,4] Rota_NP[2,3,5] Rota_NP[2,3,6] Rota_NP[3,0,0]
 Rota_NP[3,0,1] Rota_NP[3,0,2] Rota_NP[3,0,3] Rota_NP[3,0,4] Rota_NP[3,0,5]
 Rota_NP[3,0,6] Rota_NP[3,1,0] Rota_NP[3,1,1] Rota_NP[3,1,2] Rota_NP[3,1,3]
 Rota_NP[3,1,4] Rota_NP[3,1,5] Rota_NP[3,1,6] Rota_NP[3,2,0] Rota_NP[3,2,1]
 Rota_NP[3,2,2] Rota_NP[3,2,3] Rota_NP[3,2,4] Rota_NP[3,2,5] Rota_NP[3,2,6]
 Rota_NP[3,3,0] Rota_NP[3,3,1] Rota_NP[3,3,2] Rota_NP[3,3,3] Rota_NP[3,3,4]
 Rota_NP[3,3,5] Rota_NP[3,3,6] Rota_NG[0,0,0] Rota_NG[0,0,1] Rota_NG[0,0,2]
 Rota_NG[0,0,3] Rota_NG[0,0,4] Rota_NG[0,1,0] Rota_NG[0,1,1] Rota_NG[0,1,2]
 Rota_NG[0,1,3] Rota_NG[0,1,4] Rota_NG[0,2,0] Rota_NG[0,2,1] Rota_NG[0,2,2]
 Rota_NG[0,2,3] Rota_NG[0,2,4] Rota_NG[0,3,0] Rota_NG[0,3,1] Rota_NG[0,3,2]
 Rota_NG[0,3,3] Rota_NG[0,3,4] Rota_NG[1,0,0] Rota_NG[1,0,1] Rota_NG[1,0,2]
 Rota_NG[1,0,3] Rota_NG[1,0,4] Rota_NG[1,1,0] Rota_NG[1,1,1] Rota_NG[1,1,2]
 Rota_NG[1,1,3] Rota_NG[1,1,4] Rota_NG[1,2,0] Rota_NG[1,2,1] Rota_NG[1,2,2]
 Rota_NG[1,2,3] Rota_NG[1,2,4] Rota_NG[1,3,0] Rota_NG[1,3,1] Rota_NG[1,3,2]
 Rota_NG[1,3,3] Rota_NG[1,3,4] Rota_NG[2,0,0] Rota_NG[2,0,1] Rota_NG[2,0,2]
 Rota_NG[2,0,3] Rota_NG[2,0,4] Rota_NG[2,1,0] Rota_NG[2,1,1] Rota_NG[2,1,2]
 Rota_NG[2,1,3] Rota_NG[2,1,4] Rota_NG[2,2,0] Rota_NG[2,2,1] Rota_NG[2,2,2]
 Rota_NG[2,2,3] Rota_NG[2,2,4] Rota_NG[2,3,0] Rota_NG[2,3,1] Rota_NG[2,3,2]
 Rota_NG[2,3,3] Rota_NG[2,3,4] Rota_NG[3,0,0] Rota_NG[3,0,1] Rota_NG[3,0,2]
 Rota_NG[3,0,3] Rota_NG[3,0,4] Rota_NG[3,1,0] Rota_NG[3,1,1] Rota_NG[3,1,2]
 Rota_NG[3,1,3] Rota_NG[3,1,4] Rota_NG[3,2,0] Rota_NG[3,2,1] Rota_NG[3,2,2]
 Rota_NG[3,2,3] Rota_NG[3,2,4] Rota_NG[3,3,0] Rota_NG[3,3,1] Rota_NG[3,3,2]
 Rota_NG[3,3,3] Rota_NG[3,3,4] NP_utilizado[0] NP_utilizado[1]
 NP_utilizado[2] NP_utilizado[3] NP_utilizado[4] NP_utilizado[5]
 NP_utilizado[6] NG_utilizado[0] NG_utilizado[1] NG_utilizado[2]
 NG_utilizado[3] NG_utilizado[4]
Generals
 XCB[0,0] XCB[0,1] XCB[0,2] XCB[1,0] XCB[1,1] XCB[1,2] XCB[2,0] XCB[2,1]
 XCB[2,2] Q_Treinada_Por_Base[0] Q_Treinada_Por_Base[1]
 Q_Treinada_Por_Base[2] QTD_NP[0,0,0] QTD_NP[0,0,1] QTD_NP[0,0,2]
 QTD_NP[0,0,3] QTD_NP[0,0,4] QTD_NP[0,0,5] QTD_NP[0,0,6] QTD_NP[0,1,0]
 QTD_NP[0,1,1] QTD_NP[0,1,2] QTD_NP[0,1,3] QTD_NP[0,1,4] QTD_NP[0,1,5]
 QTD_NP[0,1,6] QTD_NP[0,2,0] QTD_NP[0,2,1] QTD_NP[0,2,2] QTD_NP[0,2,3]
 QTD_NP[0,2,4] QTD_NP[0,2,5] QTD_NP[0,2,6] QTD_NP[0,3,0] QTD_NP[0,3,1]
 QTD_NP[0,3,2] QTD_NP[0,3,3] QTD_NP[0,3,4] QTD_NP[0,3,5] QTD_NP[0,3,6]
 QTD_NP[1,0,0] QTD_NP[1,0,1] QTD_NP[1,0,2] QTD_NP[1,0,3] QTD_NP[1,0,4]
 QTD_NP[1,0,5] QTD_NP[1,0,6] QTD_NP[1,1,0] QTD_NP[1,1,1] QTD_NP[1,1,2]
 QTD_NP[1,1,3] QTD_NP[1,1,4] QTD_NP[1,1,5] QTD_NP[1,1,6] QTD_NP[1,2,0]
 QTD_NP[1,2,1] QTD_NP[1,2,2] QTD_NP[1,2,3] QTD_NP[1,2,4] QTD_NP[1,2,5]
 QTD_NP[1,2,6] QTD_NP[1,3,0] QTD_NP[1,3,1] QTD_NP[1,3,2] QTD_NP[1,3,3]
 QTD_NP[1,3,4] QTD_NP[1,3,5] QTD_NP[1,3,6] QTD_NP[2,0,0] QTD_NP[2,0,1]
 QTD_NP[2,0,2] QTD_NP[2,0,3] QTD_NP[2,0,4] QTD_NP[2,0,5] QTD_NP[2,0,6]
 QTD_NP[2,1,0] QTD_NP[2,1,1] QTD_NP[2,1,2] QTD_NP[2,1,3] QTD_NP[2,1,4]
 QTD_NP[2,1,5] QTD_NP[2,1,6] QTD_NP[2,2,0] QTD_NP[2,2,1] QTD_NP[2,2,2]
 QTD_NP[2,2,3] QTD_NP[2,2,4] QTD_NP[2,2,5] QTD_NP[2,2,6] QTD_NP[2,3,0]
 QTD_NP[2,3,1] QTD_NP[2,3,2] QTD_NP[2,3,3] QTD_NP[2,3,4] QTD_NP[2,3,5]
 QTD_NP[2,3,6] QTD_NP[3,0,0] QTD_NP[3,0,1] QTD_NP[3,0,2] QTD_NP[3,0,3]
 QTD_NP[3,0,4] QTD_NP[3,0,5] QTD_NP[3,0,6] QTD_NP[3,1,0] QTD_NP[3,1,1]
 QTD_NP[3,1,2] QTD_NP[3,1,3] QTD_NP[3,1,4] QTD_NP[3,1,5] QTD_NP[3,1,6]
 QTD_NP[3,2,0] QTD_NP[3,2,1] QTD_NP[3,2,2] QTD_NP[3,2,3] QTD_NP[3,2,4]
 QTD_NP[3,2,5] QTD_NP[3,2,6] QTD_NP[3,3,0] QTD_NP[3,3,1] QTD_NP[3,3,2]
 QTD_NP[3,3,3] QTD_NP[3,3,4] QTD_NP[3,3,5] QTD_NP[3,3,6] QTD_NG[0,0,0]
 QTD_NG[0,0,1] QTD_NG[0,0,2] QTD_NG[0,0,3] QTD_NG[0,0,4] QTD_NG[0,1,0]
 QTD_NG[0,1,1] QTD_NG[0,1,2] QTD_NG[0,1,3] QTD_NG[0,1,4] QTD_NG[0,2,0]
 QTD_NG[0,2,1] QTD_NG[0,2,2] QTD_NG[0,2,3] QTD_NG[0,2,4] QTD_NG[0,3,0]
 QTD_NG[0,3,1] QTD_NG[0,3,2] QTD_NG[0,3,3] QTD_NG[0,3,4] QTD_NG[1,0,0]
 QTD_NG[1,0,1] QTD_NG[1,0,2] QTD_NG[1,0,3] QTD_NG[1,0,4] QTD_NG[1,1,0]
 QTD_NG[1,1,1] QTD_NG[1,1,2] QTD_NG[1,1,3] QTD_NG[1,1,4] QTD_NG[1,2,0]
 QTD_NG[1,2,1] QTD_NG[1,2,2] QTD_NG[1,2,3] QTD_NG[1,2,4] QTD_NG[1,3,0]
 QTD_NG[1,3,1] QTD_NG[1,3,2] QTD_NG[1,3,3] QTD_NG[1,3,4] QTD_NG[2,0,0]
 QTD_NG[2,0,1] QTD_NG[2,0,2] QTD_NG[2,0,3] QTD_NG[2,0,4] QTD_NG[2,1,0]
 QTD_NG[2,1,1] QTD_NG[2,1,2] QTD_NG[2,1,3] QTD_NG[2,1,4] QTD_NG[2,2,0]
 QTD_NG[2,2,1] QTD_NG[2,2,2] QTD_NG[2,2,3] QTD_NG[2,2,4] QTD_NG[2,3,0]
 QTD_NG[2,3,1] QTD_NG[2,3,2] QTD_NG[2,3,3] QTD_NG[2,3,4] QTD_NG[3,0,0]
 QTD_NG[3,0,1] QTD_NG[3,0,2] QTD_NG[3,0,3] QTD_NG[3,0,4] QTD_NG[3,1,0]
 QTD_NG[3,1,1] QTD_NG[3,1,2] QTD_NG[3,1,3] QTD_NG[3,1,4] QTD_NG[3,2,0]
 QTD_NG[3,2,1] QTD_NG[3,2,2] QTD_NG[3,2,3] QTD_NG[3,2,4] QTD_NG[3,3,0]
 QTD_NG[3,3,1] QTD_NG[3,3,2] QTD_NG[3,3,3] QTD_NG[3,3,4] QTD_transp_NP[0]
 QTD_transp_NP[1] QTD_transp_NP[2] QTD_transp_NP[3] QTD_transp_NP[4]
 QTD_transp_NP[5] QTD_transp_NP[6] QTD_transp_NG[0] QTD_transp_NG[1]
 QTD_transp_NG[2] QTD_transp_NG[3] QTD_transp_NG[4]
 QTD_transp_por_base_NP[0] QTD_transp_por_base_NP[1]
 QTD_transp_por_base_NP[2] QTD_transp_por_base_NG[0]
 QTD_transp_por_base_NG[1] QTD_transp_por_base_NG[2] Distancia_rota_NP[0]
 Distancia_rota_NP[1] Distancia_rota_NP[2] Distancia_rota_NP[3]
 Distancia_rota_NP[4] Distancia_rota_NP[5] Distancia_rota_NP[6]
 Distancia_rota_NG[0] Distancia_rota_NG[1] Distancia_rota_NG[2]
 Distancia_rota_NG[3] Distancia_rota_NG[4] Distancia_rota_NG[5]
 Distancia_rota_NG[6]
End
