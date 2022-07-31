import copy

import MathLibrary as Ml
import Linear
import pandas as pd
import numpy as np
import Matrix

dfp1 = pd.read_excel('p1.xlsx', header=None)
dfp2 = pd.read_excel('p2.xlsx', header=None)
dfc = pd.read_excel('c.xlsx', header=None)

dfp_count = dfp1[0].apply(str).apply(lambda x: x.split('.')[0])
dfp_count = dfp_count.value_counts().sort_index()

rules = [1, 1, 1, 1, 1, 1]

dfp1_spl = pd.DataFrame()
for i in range(len(rules)):
    splitdfp1 = dfp1.loc[(dfp1[0].apply(str).apply(lambda x: x.split('.')[0]) == str(i + 1)) & (
                dfp1[0].apply(str).apply(lambda x: x.split('.')[1]) == str(rules[i]))]
    dfp1_spl = dfp1_spl.append(splitdfp1)
dfp1_spl = dfp1_spl.reset_index().drop(columns=[0,'index'])
p1 = dfp1_spl.to_numpy()

dfp2_spl = pd.DataFrame()
for i in range(len(rules)):
    splitdfp2 = dfp2.loc[(dfp2[0].apply(str).apply(lambda x: x.split('.')[0]) == str(i + 1)) & (
                dfp2[0].apply(str).apply(lambda x: x.split('.')[1]) == str(rules[i]))]
    dfp2_spl = dfp2_spl.append(splitdfp2)
dfp2_spl = dfp2_spl.reset_index().drop(columns=[0, 'index'])
p2 = dfp2_spl.to_numpy()
c = dfc.to_numpy()
c = c.transpose()
p = np.zeros([len(p1),len(p1)])
for i in range(len(p1)):
    a = (p1[i], p2[i])
    p_itog, F = Linear.solver(c, a, 'max')
    p[i] = p_itog
q = np.dot(p,c)
print(q)
df = pd.DataFrame(p)
# Найдем матрицу ПИ
dataframe, gates, boxes = Matrix.find_boxes(df)
print('Проходные состояния:' + '\n', gates)
print('Ящики:' + '\n', *boxes)
LinResult = Matrix.linear_matrix(dataframe, boxes)
Pi_matrix = Matrix.pi_matrix(dataframe, gates, boxes, LinResult)
print('Рассчитанная матрица Пи:' + '\n', Pi_matrix)

# найдем r(P) формула в задаче 3.2
r_P = copy.deepcopy(Pi_matrix)
r_P = np.dot(Pi_matrix, q)
print(r_P)
# Ml.ergo_solver('PrimerAV.xlsx')
