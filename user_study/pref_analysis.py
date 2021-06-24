# -*- coding: utf-8 -*-
# @Author: TomLotze
# @Date:   2021-06-23 13:21
# @Last Modified by:   TomLotze
# @Last Modified time: 2021-06-23 13:24


preferences = [[3, 0, 0], [2, 1, 0],[0, 0, 3],
 [0, 2, 1], [3, 0, 0], [0, 3, 0], [0, 2, 1], [1, 2, 0],
 [0, 1, 2], [2, 0, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3],
 [0, 0, 3], [0, 2, 1], [0, 3, 0], [2, 1, 0], [3, 0, 0],
 [3, 0, 0], [0, 2, 1], [2, 1, 0], [0, 2, 1], [1, 0, 2],
 [0, 3, 0], [0, 3, 0], [3, 0, 0], [0, 3, 0], [3, 0, 0],
 [0, 0, 3], [1, 0, 2], [0, 0, 3], [0, 2, 1], [0, 1, 2],
 [0, 2, 1], [0, 3, 0], [0, 3, 0], [2, 1, 0], [0, 2, 1],
 [1, 1, 1], [3, 0, 0], [2, 0, 1], [0, 0, 3], [1, 1, 1],
 [1, 2, 0], [1, 0, 2], [2, 0, 1], [3, 0, 0], [1, 0, 2],
 [3, 0, 0], [2, 0, 1], [1, 1, 1], [2, 0, 1], [0, 0, 3],
 [2, 0, 1], [1, 1, 1], [2, 0, 1], [0, 2, 1], [3, 0, 0],
 [0, 1, 2], [0, 3, 0]]

pref_matrix = [[(0, 0, 0) for i in range(4)] for j in range(4)]
for i, pref in enumerate(preferences):
    true_class1, true_class2 = GT[i]

    if true_class1 < true_class2:
        old_tuple = pref_matrix[true_class2][true_class1]
        new_tuple = (pref[1]+ old_tuple[0], pref[0] + old_tuple[1], old_tuple[2]+pref[2])
        pref_matrix[true_class2][true_class1] = new_tuple
    else:
        old_tuple = pref_matrix[true_class1][true_class2]
        new_tuple = tuple([i + j for i, j in zip(old_tuple, pref)])
        pref_matrix[true_class1][true_class2] = new_tuple
for i, row in enumerate(pref_matrix):
    for j, tup in enumerate(row):
        if tup == (0, 0, 0):
            pref_matrix[i][j]=0



print(pref_matrix)