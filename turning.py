import matplotlib.pyplot as plt
import numpy as np
import math
import copy


def draw_turning(p):
    #p[[1,1],[1,3],[2,3],[2,1],[1,1]]

    #print(p)
    lenp = len(p)
    totallen = 0
    vectorlen = np.zeros(lenp)
    vectorlen[0] = 0
    angle = np.zeros(lenp - 1)
    xvalue = np.zeros(lenp - 1)
    for i in range(lenp-1):
        vectorlen[i+1] = math.sqrt((p[i+1][0] - p[i][0])**2 + (p[i+1][1] - p[i][1])**2)
        totallen += vectorlen[i+1]
    abx1 = 1 - p[0][0]
    aby1 = -p[0][1]
    abx2 = p[1][0] - p[0][0]
    aby2 = p[1][1] - p[0][1]
    a = [abx1, aby1, 0]
    b = [abx2, aby2, 0]
    c = abx1 * aby2 - abx2 * aby1
    pointcheng = (abx1 * abx2) + (aby1 * aby2)
    changducheng = math.sqrt(abx1 ** 2 + aby1 ** 2) * math.sqrt(abx2 ** 2 + aby2 ** 2)
    pos = pointcheng / changducheng
    pp = math.acos(pos)
    ''''''
    if c < 0:
        angle[0] = -pp
    elif c > 0:
        angle[0] = pp
    else:
        angle[0] = 0.785

    angle[0] = pp
    xvalue[0] = 0

    for i in range(1, lenp - 1):
        xvalue[i] = xvalue[i-1] + vectorlen[i] / totallen
        abx1 = p[i-1][0] - p[i][0]
        aby1 = p[i-1][1] - p[i][1]
        abx2 = p[i+1][0] - p[i][0]
        aby2 = p[i+1][1] - p[i][1]
        a = [abx1, aby1, 0]
        b = [abx2, aby2, 0]
        c = abx1 * aby2 - abx2 * aby1
        pointcheng = (abx1 * abx2) + (aby1 * aby2)
        changducheng = math.sqrt(abx1 ** 2 + aby1 ** 2) * math.sqrt(abx2 ** 2 + aby2 ** 2)
        pos = pointcheng / changducheng;
        pp = math.acos(pos)

        if c < 0:
            angle[i] = angle[i-1] - pp
        elif c > 0:
            angle[i] = angle[i-1] + pp
        else:
            angle[i] = angle[i-1]

    angle = angle / np.pi
    sum = 0

    return xvalue, angle

