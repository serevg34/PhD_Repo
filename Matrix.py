import copy
import pandas as pd
import numpy as np
import functools


# Считываем данные из файла Excel
df = pd.read_excel('Primer5.xlsx', sheet_name='Лист1', header=None)
# Инициализируем списки для переходов и ящиков
step, box, final = [], [], []
# Создаем матрицу заполненную нулями
R = np.zeros((len(df), len(df)))
for i in range(len(df)):
    Ri, Rj, R_obj = [], [], []
    step.append(i)
    for j in range(len(df)):
        if df[j][i] > 0:
            Ri.append(j)
    if Ri == [i]:
        # Во всей строке только пересечение с собой получается ящик из одного элемента
        print(i, '= Bad')
        # Одиночный ящик добавляем в финальный ответ
        final.append([i+1])
        # Одиночный ящик исключаем из списка проходных состояний
        step.remove(i)
    for j in Ri:
        for k in range(len(df)):
            if df[k][j] > 0:
                Rj.append(k)
    Ri = Ri + Rj
    for item in Ri:
        if item not in R_obj:
            R_obj.append(item)
    R_obj.sort()
    for j in range(int(len(df) - len(R_obj))):
        R_obj.append(0)
    R[i] = R_obj
print('Получившаяся матрица переходов '+'\n', R)
# Внутри цикла ищем ящики
g = np.arange(0, len(R))
for i in range(len(df)):
    case = []
    for j in range(len(df)):
        x, y = R[i], R[j]
        if i != j:
            if functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, x, y)):
                if all(R[j] != g.astype(np.float64)):
                    case.append(j)
    if case:
        # Удаляем найденные ящики из списка проходных состояний
        for k in case:
            if k in step:
                step.remove(k)
            case = []
# Вторая итерация поиска ящиков
for i in range(len(df)):
    case = []
    for j in range(len(df)):
        x, y = R[i], R[j]
        if functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, x, y), True):
            case.append(j+1)
    box.append(case)
ways = [h + 1 for h in step]
print('Проходные состояния' + '\n', ways)
# Исключаем из списка ящиков повторения и проходные состояния
for z in box:
    if len(z) > 1 and (z not in final) and (z[0] not in ways):
        final.append(z)
        final.sort()
print('Ящики' + '\n', *final)
print()
for i in range(len(final)):
    final[i] = [h - 1 for h in final[i]]
    LinMatrix = df.iloc[final[i], final[i]]
    m = np.array(LinMatrix.values.tolist()).transpose()
    for j in range(len(m)):
        m[j][j] = m[j][j] - 1
    m[len(m)-1] = [1 for _ in range(len(m))]
    V = np.zeros(len(m))
    V[len(m)-1] = 1
    otvet = np.linalg.solve(m, V)
    print(otvet)

# Создание матрицы без проходных состояний
df1 = copy.deepcopy(df)
df1 = df1.drop(index=step, columns=step)
df1 = df1.reset_index(drop=True)
df1 = pd.DataFrame(df1)
# Изменить имя столбцов
# Надо сделать проверки на значения в матрице
# Проверка внутри ящика что сумма равна 1
