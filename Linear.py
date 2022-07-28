import os
import numpy as np
import copy
import pandas


def solver(c, a, method):
    sum_a1 = 0
    sum_a2 = 0
    k = 0
    for i in range(len(a[0])):
        sum_a1 += a[0][i]
        sum_a2 += a[1][i]
    if sum_a1 > 1 or sum_a2 < 1:
        print("Ошибка ограничений, множество пустое!")
        return
    for i in range(len(a[0])):
        if a[0][i] == a[1][i]:
            k += 1
        elif a[0][i] < 0 or a[1][i] < 0:
            print("Ограничения Отрицательные!")
            return
        elif a[0][i] > 1 or a[1][i] > 1:
            print("Ограничения больше единицы!")
            return
        elif a[0][i] > a[1][i]:
            print("Нижнее ограничение больше верхнего!")
            return
    if k == len(a[0]) or sum_a1 == 1 or sum_a2 == 1:
        print("Ошибка ограничений, множество состоит из одного элемента!")
        return
    k = 0
    for i in range(len(c) - 1):
        if c[i] == c[i + 1]:
            k = k + 1
    if k == len(c) - 1:
        print("Любой вектор из множества является решением задачи!")
        return
    r = [[0] * len(c) for _ in range(4)]
    for i in range(len(c)):
        r[0][i] = c[i]
        r[1][i] = i + 1
        r[2][i] = a[0][i]
        r[3][i] = a[1][i]
    heap_sort(copy.deepcopy(r[0]), r)
    if method == 'max':
        for b in r:
            b.reverse()
    x, y, z, sum_for_alpha, alpha, r2 = [], [], [], 0, 0, copy.deepcopy(r[2])
    if sum_a1 < 1:
        if sum_a2 > 1:
            for _ in range(len(a[0])):
                x.append(r2)
            for i in range(len(a[0])):
                for j in range(i + 1):
                    x[j][i] = r[3][j]
                y.append(0)
                for j in range(len(a[0])):
                    y[i] += x[i][j]
                if y[i] >= 1:
                    for j in range(len(a[0])):
                        if i == j:
                            sum_for_alpha = sum_for_alpha
                        else:
                            sum_for_alpha += x[i][j]
                    alpha = (1 - sum_for_alpha) / r[3][i]
                    for j in range(len(a[0])):
                        z.append(0)
                        z[j] = x[i][j]
                        if i == j:
                            z[j] = x[i][j] * alpha
                    break
            r.append(z)
            # Обратная пирамидальная сортировка
            heap_sort(copy.deepcopy(r[1]), r)
    # Производим округление результата, что бы избежать проблемы цифрового нуля
    x = np.around(r[4], 2)
    # Находим значение целевой функции
    f = (c * x).sum()
    return x, f


def heapify(nums, heap_size, root_index, m):
    # Предположим, что индекс самого большого элемента является корневым индексом
    largest = root_index
    left_child = (2 * root_index) + 1
    right_child = (2 * root_index) + 2
    # Если левый потомок корня является допустимым индексом, а элемент больше
    # чем текущий самый большой элемент, то обновляем самый большой элемент
    if left_child < heap_size and nums[left_child] > nums[largest]:
        largest = left_child
    # Делаем то же самое для right_child
    if right_child < heap_size and nums[right_child] > nums[largest]:
        largest = right_child
    # Если самый большой элемент больше не является корневым элементом, меняем их местами
    if largest != root_index:
        nums[root_index], nums[largest] = nums[largest], nums[root_index]
        for j in range(len(m)):
            m[j][root_index], m[j][largest] = m[j][largest], m[j][root_index]
        # Еще раз проходим функцией, чтобы проверить, что новый узел максимальный по значению
        heapify(nums, heap_size, largest, m)


def heap_sort(nums, m):
    n = len(nums)
    # Создаем Max Heap из списка
    # Второй аргумент означает, что мы останавливаемся на элементе перед -1, то есть на первом элементе списка.
    # Третий аргумент означает, что мы повторяем в обратном направлении, уменьшая количество i на 1
    for i in range(n, -1, -1):
        heapify(nums, n, i, m)
    # Перемещаем корень max heat в конец
    for i in range(n - 1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]
        for j in range(len(m)):
            m[j][i], m[j][0] = m[j][0], m[j][i]
        heapify(nums, i, 0, m)


def prepare_data(raw_data):
    # Объявляем имена столбцов в файле Excel чтобы по ним разделять данные
    c_title, a1_title, a2_title = 'Cтоимости', 'Нижние границы интервалов вероятности A1', \
                                  'Верхние границы интервалов вероятности A2'
    # Создаем кортежи нужного размера заполненные нулями, затем заполняем их данными из DataFrame
    c_raw, a1_raw, a2_raw = [0] * len(raw_data[c_title]), [0] * len(raw_data[c_title]), [0] * len(raw_data[c_title])
    for I in range(len(raw_data[c_title])):
        c_raw[I] = raw_data[c_title][I]
        a1_raw[I] = raw_data[a1_title][I]
        a2_raw[I] = raw_data[a2_title][I]
    return c_raw, a1_raw, a2_raw


def universal_solver(input_data, method):
    # Объявляем имя столбцов содержащих решение задачи
    X, F = 'Минимизирующее решение X' if method == 'min' else 'Максимизирующее решение Х', 'Значение  F'
    if '.csv' in input_data:
        # Читаем данные из файла и создаем объект DataFrame
        data = pandas.read_csv('in.csv', sep=';')
        # Используем функцию подготовки данных, для получения списков значений
        C, A1, A2 = prepare_data(data)
        # Дополняем DataFrame столбцом со значениями х-сов полученных из функции solver
        data[X], data[F] = solver(C, [A1, A2], method)
        data[F][1:] = None
        # Записываем данные DataFrame в файл csv
        data.to_csv('out.csv', index=False, sep=';')
        print('Решение записано в файл out.csv')
    elif '.xlsx' in input_data:
        # Читаем данные из файла и создаем объект DataFrame
        data = pandas.read_excel('Data.xlsx', engine='openpyxl')
        # Используем функцию подготовки данных, для получения списков значений
        C, A1, A2 = prepare_data(data)
        # Дополняем DataFrame столбцом со значениями х-сов полученных из функции solver
        data[X], data[F] = solver(C, [A1, A2], method)
        data[F][1:] = None
        # Записываем данные DataFrame в исходный файл Excel
        writer = pandas.ExcelWriter('Data.xlsx')
        data.to_excel(writer, 'Test', index=False)
        writer.save()
        print('Решение записано в файл Excel')
    elif len(input_data) == 2 and len(input_data[0]) == len(input_data[1][0]):
        # Входные данные передаются напрямую в функцию
        x_final, f_final = solver(input_data[0], input_data[1], method)
        # Печатаем решение в консоль
        print('X =', x_final, ' F max =' if method == 'max' else ' F min =', f_final)
    else:
        print('Ошибка! Некорректные входные данные.')
    return


# Начало работы программы
# Введем входные данные с помощью переменных
# print(os.getcwd())
# costs = [2, 4, 10, 8, 6]
# low_a = [0.1, 0.2, 0.1, 0.05, 0.05]
# high_a = [0.3, 0.4, 0.5, 0.1, 0.3]
# Data = [costs, [low_a, high_a]]
# # На вход данной функции можно передать или переменную Data, или файлы csv и xlsx
# # print("Приветствую! Задайте входные данные. Для этого запишите имя файла Excel или csv или напишите Data для "
# #       "использования демонстрационных данных")
# # file = input()
# # print("Укажите метод записав min (Минимум) или max (Максимум)")
# # algorithm = input()
# if file == "Data":
#     print("C=", costs)
#     print("a1=", low_a)
#     print("a2=", high_a)
#     universal_solver(Data, algorithm)
# else:
#     universal_solver(file, algorithm)
# print('Нажмите Enter, чтобы закрыть окно')
# input()
