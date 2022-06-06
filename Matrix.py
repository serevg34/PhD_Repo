import os
import pandas as pd
import numpy as np
import functools


def linear_matrix(dataframe, boxes):
    # Решение системы линейных уравнений
    linResult = []
    for i in range(len(boxes)):
        boxes[i] = [h - 1 for h in boxes[i]]
        linMatrix = dataframe.iloc[boxes[i], boxes[i]]
        box_values = np.array(linMatrix.values.tolist()).transpose()
        for j in range(len(box_values)):
            box_values[j][j] = box_values[j][j] - 1
        box_values[len(box_values) - 1] = [1 for _ in range(len(box_values))]
        V = np.zeros(len(box_values))
        V[len(box_values) - 1] = 1
        answer = np.linalg.solve(box_values, V)
        linResult.append(answer)
    return linResult


def pi_matrix(dataframe, gates, boxes, linear_result):
    # Решение линейной системы и создание матрицы Пи
    matrix_PI = np.zeros([len(dataframe), len(dataframe)])
    right = np.zeros([len(gates), len(dataframe)])
    left = np.zeros([len(gates), len(gates)])
    edin = np.zeros([len(gates), len(gates)])
    for i in range(len(boxes)):
        for j in boxes[i]:
            r = 0
            for k in boxes[i]:
                matrix_PI[j, k] = linear_result[i][r]
                r += 1
    table = np.asarray(dataframe.values.tolist())
    k = 0
    for i in gates:
        right[k] = np.matmul(table[i - 1], matrix_PI)
        k += 1
    for i in range(len(gates)):
        for j in range(len(gates)):
            left[i][j] = table[gates[i] - 1][gates[j] - 1]
            edin[i][i] = 1
    left = edin - left
    answer = np.linalg.solve(left, right)
    for i in range(len(gates)):
        matrix_PI[gates[i] - 1] = answer[i]
    result = (np.around(matrix_PI, 2))
    return result


def find_boxes(file):
    # Считываем данные из файла Excel
    df = pd.read_excel(file, sheet_name='Лист1', header=None)
    # Инициализируем списки для переходов и ящиков
    step, box, bins = [], [], []
    # Создаем матрицу, заполненную нулями
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
            bins.append([i+1])
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
    # print('Получившаяся матрица переходов '+'\n', R)
    # Внутри цикла ищем ящики
    g = np.arange(0, len(R))
    for i in range(len(df)):
        case = []
        for j in range(len(df)):
            x, y = R[i], R[j]
            if i != j:
                if functools.reduce(lambda a, b: a and b, map(lambda p, q: p == q, x, y)):
                    if all(R[j] != g.astype(np.float64)):
                        case.append(j)
        if case:
            # Удаляем найденные ящики из списка проходных состояний
            for k in case:
                if k in step:
                    step.remove(k)
    # Вторая итерация поиска ящиков
    for i in range(len(df)):
        case = []
        for j in range(len(df)):
            x, y = R[i], R[j]
            if functools.reduce(lambda a, b: a and b, map(lambda p, q: p == q, x, y), True):
                case.append(j+1)
        box.append(case)
    ways = [h + 1 for h in step]
    # Исключаем из списка ящиков повторения и проходные состояния
    for z in box:
        if len(z) > 1 and (z not in bins) and (z[0] not in ways):
            bins.append(z)
            bins.sort()
    return df, ways, bins
    # Надо сделать проверки на значения в матрице
    # Проверка внутри ящика, что сумма равна 1


def ergo_solver(file):
    dataframe, gates, boxes = find_boxes(file)
    print('Проходные состояния:' + '\n', gates)
    print('Ящики:' + '\n', *boxes)
    LinResult = linear_matrix(dataframe, boxes)
    Pi_matrix = pi_matrix(dataframe, gates, boxes, LinResult)
    print('Рассчитанная матрица Пи:' + '\n', Pi_matrix)


# Начало работы программы
print(os.getcwd())
ergo_solver('PrimerAV.xlsx')
print('Нажмите Enter, чтобы закрыть окно')
input()
