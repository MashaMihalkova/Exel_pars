import re
import os
import pandas as pd
import numpy as np
from glob import glob

# xlsx parser - так как xlsx имеет другую структуру
def parse_data(df):
    data = []
    year = int(df.iloc[0, 2][14:18])
    month = int(df.iloc[0, 2][11:13])
    dict_OOO = dict()  # подрядчик
    dict_stage = dict()  # проект
    dict_res_name = dict()  # ресурс

    stage_opened = False  # служебное
    res_type_opened = False  # служебное

    for ind, i in enumerate(df.iloc[7:, 0], start=7):
        if not pd.isna(i):

            # Условие выхода из цикла: Нашли "Итого"
            if (re.findall(r'итого', i.lower())) or (ind == df.shape[0]):
                # append в data
                for k, v in dict_res_name.items():
                    data.extend([{'year': year, 'month': month, **dict_OOO, **dict_stage, **dict_res_type,
                                  'res_name': k, 'hours': v}])

                # проверка на сумму машиночасов из словаря и из Экселя
                assert round(sum(dict_res_name.values()), 0) == round(df.iloc[last_res_type_index, 13],
                                                                      0), f'sum of machines = {sum(dict_res_name.values())} should be the same in Excel = {df.iloc[last_res_type_index, 13]}, last_resource_index ={last_res_type_index},current_index={ind}'
                # break

                print('Parsing finished')
                break

            # ищем OOO
            if re.findall(r'ооо', i.lower()):
                # нашли OOO

                # тут я записываю все предыдущие 'res_name' и их 'hours'
                if res_type_opened:  # если до этого 'res_type' был открыт
                    # тут записывается [ООО: 'ххх' , stage: 'xxx', res_type: 'xxx', res_name: 'xxx', 'hours': 'xxx'] и аппендит в data
                    for k, v in dict_res_name.items():
                        data.extend(
                            [{'year': year, 'month': month, **dict_OOO, **dict_stage, **dict_res_type, 'res_name': k,
                              'hours': v}])
                    dict_res_name = dict()  # обнуляем словарь так как начинается другой ООО

                    res_type_opened = False
                    stage_opened = False

                dict_OOO = {'contractor': i}
                continue

            # ищем проект  - в названии есть "этап"
            if re.findall(r'этап', i.lower()):

                # тут я записываю все предыдущие 'res_name' и их 'hours'
                if res_type_opened:  # если до этого 'res_type' был открыт
                    # тут записывается [ООО: 'ххх' , stage: 'xxx', res_type: 'xxx', res_name: 'xxx', 'hours': 'xxx'] и аппендит в data
                    for k, v in dict_res_name.items():
                        data.extend(
                            [{'year': year, 'month': month, **dict_OOO, **dict_stage, **dict_res_type, 'res_name': k,
                              'hours': v}])
                    dict_res_name = dict()  # обнуляем словарь так как начинается другой проект

                    res_type_opened = False

                dict_stage = {'stage': i}
                stage_opened = True
                continue

            # ищем группу ресурсов или res_type [например Трубоукладчики, Тягачи ...] - на 8 столбце должно быть число
            # еще вариант такой type(df.iloc[ind,8]) == int было до этого toBeDeleted:  not pd.isna(df.iloc[ind,8])
            if (stage_opened) & (pd.isna(df.iloc[ind, 3])) & (not re.findall(r'\(', i.lower())):
                if (pd.isna(df.iloc[ind, 8])): continue

                # тут я записываю все предыдущие 'res_name' и их 'hours'
                if res_type_opened:  # если до этого 'res_type' был открыт
                    # тут записывается [ООО: 'ххх' , stage: 'xxx', res_type: 'xxx', res_name: 'xxx', 'hours': 'xxx'] и аппендит в data
                    for k, v in dict_res_name.items():
                        data.extend(
                            [{'year': year, 'month': month, **dict_OOO, **dict_stage, **dict_res_type, 'res_name': k,
                              'hours': v}])

                    # проверка на сумму машиночасов из словаря и из Экселя
                    if pd.isna(df.iloc[last_res_type_index, 13]):
                        print(
                            f' Missed data in file (sum of resources): row = {last_res_type_index}, column= {13}: {df.iloc[last_res_type_index, 13]}')
                    else:
                        assert round(sum(dict_res_name.values()), 0) == round(df.iloc[last_res_type_index, 13],
                                                                              0), f'sum of machines = {sum(dict_res_name.values())} should be the same in Excel = {df.iloc[last_res_type_index, 13]}, {last_res_type_index},{ind}'

                    dict_res_name = dict()  # обнуляем словарь так как начинается другой тип машин

                # тут я объявляю новый 'res_type'
                dict_res_type = {'res_type': i}
                res_type_opened = True
                last_res_type_index = ind  # чтобы потом проверять сумму машиночасов в assert

                continue

            # ищем группу имя ресурса или res_name [Тягач лесовозный, Тягач седельный ... ]  - на 3 столбце должно быть число
            ## как правило на 3 столбце должно быть число -->  not pd.isna(df.iloc[ind, 3])
            ## но если числа нет, то
            ## может присутствовать скобка в названии -->  ( len(re.findall(r'\(', i.lower()))>0 )
            ## и в названии больше одного слова
            if (stage_opened) & (res_type_opened) & (
                    (not pd.isna(df.iloc[ind, 3])) or (len(re.findall(r'\(', i.lower())) > 0) or (len(i.split()) > 1)):
                if pd.isna(df.iloc[ind, 13]): continue  # На случай если данных по машине нет

                if i not in dict_res_name.keys():  # 'res_name':'hours'
                    dict_res_name[i] = df.iloc[ind, 13]
                else:
                    # если такое значение уже есть в списке
                    dict_res_name[i] += df.iloc[ind, 13]
    return data

# xls parser - так как xls имеет другую структуру
def parse_data_xls(df):
    year = int(df.iloc[0, 2][14:18])
    month = int(df.iloc[0, 2][11:13])
    data = []
    dict_OOO = dict()  # подрядчик
    dict_stage = dict()  # проект
    dict_res_name = dict()  # ресурс

    stage_opened = False  # служебное
    res_type_opened = False  # служебное

    for ind, i in enumerate(df.iloc[7:, 0], start=7):
        if not pd.isna(i):

            # Условие выхода из цикла: Нашли "Итого"
            if re.findall(r'итого', i.lower()):
                # append в data
                for k, v in dict_res_name.items():
                    data.extend([{'year': year, 'month': month, **dict_OOO, **dict_stage, **dict_res_type,
                                  'res_name': k, 'hours': v}])

                # проверка на сумму машиночасов из словаря и из Экселя
                assert round(sum(dict_res_name.values()), 0) == round(df.iloc[last_res_type_index, 13],
                                                                      0), f'sum of machines = {sum(dict_res_name.values())} should be the same in Excel = {df.iloc[last_res_type_index, 13]}, last_resource_index ={last_res_type_index},current_index={ind}'

                print('Parsing finished')
                break

                # ищем OOO
            if re.findall(r'ооо*|есин|стройтранс*|газстройпро*', i.lower()):
                # нашли OOO

                # так как ООО может начаться новый в рамках одного ресурса (res_type и dict_res_name)
                # если следующая строка после ООО будет техника - то нужно append-ить в data
                ## ищем следующую технику:
                if (not pd.isna(df.iloc[ind + 1, 3]) and (ind > 10) and (not pd.isna(df.iloc[
                                                                                         ind - 1, 3]))):  # еще одно условие  если первое слишком слабое or ( not pd.isna(df.iloc[ind+1,8])
                    # следующяя строка будет техника - тогда заносим аппенд в data

                    for k, v in dict_res_name.items():
                        data.extend([{'year': year, 'month': month, **dict_OOO, **dict_stage, **dict_res_type,
                                      'res_name': k, 'hours': v}])

                    # проверка на сумму машиночасов из словаря и из Экселя
                    assert round(sum(dict_res_name.values()), 0) == round(df.iloc[last_res_type_index, 13],
                                                                          0), f'sum of machines = {sum(dict_res_name.values())} should be the same in Excel = {df.iloc[last_res_type_index, 13]}, last_resource_index ={last_res_type_index},current_index={ind}'

                    dict_res_name = dict()  # обнуляем словарь так как начинается другой ООО

                dict_OOO = {'contractor': i}
                # проверка если следующий элемент - это ресурс - то обновляем last_res_type_index
                if (not pd.isna(df.iloc[ind + 1, 3])):
                    last_res_type_index = ind  # чтобы потом проверять сумму машиночасов в assert
                continue

            # ищем проект  - в названии есть "этап"
            if re.findall(r'этап', i.lower()):
                #  ---  тут мы должны обновить data
                if res_type_opened:

                    # тут записывается [ООО: 'ххх' , stage: 'xxx', res_type: 'xxx', res_name: 'xxx', 'hours': 'xxx'] и аппендит в data
                    for k, v in dict_res_name.items():
                        data.extend([{'year': year, 'month': month, **dict_OOO, **dict_stage, **dict_res_type,
                                      'res_name': k, 'hours': v}])

                    # проверка на сумму машиночасов из словаря и из Экселя
                    assert round(sum(dict_res_name.values()), 0) == round(df.iloc[last_res_type_index, 13],
                                                                          0), f'sum of machines = {sum(dict_res_name.values())} should be the same in Excel = {df.iloc[last_res_type_index, 13]}, last_resource_index ={last_res_type_index},current_index={ind}'

                    dict_res_name = dict()  # обнуляем словарь так как начинается другой тип машин

                # --- Закончили обновлять data

                dict_stage = {'stage': i}
                res_type_opened = False  # это нужно закрывать на случай, если после этапа будет res_type и в data занесется еще не заполненный res_type
                stage_opened = True
                continue

            # ищем группу ресурсов или res_type [например Трубоукладчики, Тягачи ...] - на 8 столбце должно быть число
            if (stage_opened) & (pd.isna(df.iloc[ind, 3])) & (not re.findall(r'\(',
                                                                             i.lower())):  # еще вариант такой type(df.iloc[ind,8]) == int было до этого toBeDeleted:  not pd.isna(df.iloc[ind,8])

                if (pd.isna(df.iloc[ind, 8])) & (pd.isna(df.iloc[ind, 13])):
                    print('в строке нет ни часов ни вида техники. Значение этого ряда не будет учитываться', i,
                          df.iloc[ind, :])
                    continue

                # поскольку res_type может смениться у одного подрядчика - записываем в data
                # тут я записываю все предыдущие 'res_name' и их 'hours'
                if res_type_opened:

                    # тут записывается [ООО: 'ххх' , stage: 'xxx', res_type: 'xxx', res_name: 'xxx', 'hours': 'xxx'] и аппендит в data
                    for k, v in dict_res_name.items():
                        data.extend([{'year': year, 'month': month, **dict_OOO, **dict_stage, **dict_res_type,
                                      'res_name': k, 'hours': v}])

                    # проверка на сумму машиночасов из словаря и из Экселя
                    assert round(sum(dict_res_name.values()), 0) == round(df.iloc[last_res_type_index, 13],
                                                                          0), f'sum of machines = {sum(dict_res_name.values())} should be the same in Excel = {df.iloc[last_res_type_index, 13]}, last_resource_index ={last_res_type_index},current_index={ind}'
                    # break

                    dict_res_name = dict()  # обнуляем словарь так как начинается другой тип машин

                # тут я объявляю новый 'res_type'
                dict_res_type = {'res_type': i}
                res_type_opened = True

                continue

            # ищем группу имя ресурса или res_name [Тягач лесовозный, Тягач седельный ... ]  - на 3 столбце должно быть число
            if (stage_opened) & (res_type_opened) & (
                    (not pd.isna(df.iloc[ind, 3])) or (len(re.findall(r'\(', i.lower())) > 0) or (
                    len(i.split()) > 1)):  # старое условие - его убрать ( not pd.isna(df.iloc[ind,3]) )
                if pd.isna(df.iloc[ind, 13]): continue  # На случай если данных по машине нет

                if i not in dict_res_name.keys():  # 'res_name':'hours'
                    dict_res_name[i] = df.iloc[ind, 13]
                else:
                    dict_res_name[i] += df.iloc[ind, 13]  # если такое значение уже есть в списке

    return data




