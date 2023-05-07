from numpy import mat, zeros
import numpy as np
from sim import cal_sim
from hungarian import Hungarian
import copy


def dif_matrix(graph):
    polygon1 = graph[0]
    polygon2 = graph[1]

    len1 = len(polygon1)
    len2 = len(polygon2)

    data = []

    for m in range(len1):
        vertex1 = polygon1[m]
        col = []
        for n in range(len2):
            vertex2 = polygon2[n]
            #print(vertex1)
            dif = 1 - cal_sim(vertex1, vertex2)
            col.append(dif)
        data.append(col)
    #print(data)
    return data

def nei_matrix(graph):
    polygon1 = graph[0]
    polygon2 = graph[1]

    len1 = len(polygon1)
    len2 = len(polygon2)

    data = dif_matrix(graph)
    da = data
    rl = len(da)
    cl = len(da[0])
    crl = np.abs(rl - cl)

    if rl > cl:
        for i in da:
            for n in range(crl):
                i.append(999999)
    if rl < cl:
        for i in range(crl):
            da.append([0]*len(da[0]))
    nei_m = []
    nei_result = []
    #print(data)
    for m in range(len1):
        col = []
        col1 = []
        for n in range(len2):
            d = data[m][n]
            data1 = copy.deepcopy(data)
            data1[m][n] = -99999
            hungarian = Hungarian(data1)
            hungarian.calculate()
            if rl > cl:
                res = (hungarian.get_total_potential() + d - 999999 * crl) / min(len1, len2)
            else:
                res = (hungarian.get_total_potential() + d + 99999) / min(len1, len2)
            #print(res)
            col1.append(1-res)
            col.append(hungarian.get_results())
        nei_result.append(col)
        nei_m.append(col1)
    if rl < cl:
        for i in range(rl):
            for n in range(cl):
                nei_result[i][n] = copy.deepcopy(nei_result[i][n][0: cl-crl])
    nr = copy.deepcopy(nei_result)
    if rl > cl:
        for i in range(rl):
            for n in range(cl):
                for x in range(len(nei_result[i][n])):
                    a, b = nei_result[i][n][x]
                    #nrr = copy.deepcopy(nei_result[i][n])
                    if b > (cl - 1):
                        #nrr.pop(x)
                        nr[i][n].remove((a, b))
    nei_result = copy.copy(nr)
    #print(nei_result)
    return nei_m, nei_result


if __name__ == '__main__':
    data = [1,2,2,6,5,5,4,8]
    print(data)
    data.remove(2)
    print(data)
