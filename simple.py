import numpy as np
import copy


def simple_polygon(polygon, n):
    l1 = [polygon[0]]
    for i in range(1, len(polygon)):
        if i % n == 0:
            l1.append(polygon[i])
    if (len(polygon) - 1) % n != 0:
        l1.append(polygon[-1])
    return l1


def bt(lev):
    for x in range(lev):
        for y in range(2**x):
            print("send " + str(y) + " to " + str(y) + " lev: " + str(lev - x - 1))
            print("send " + str(y) + " to " + str(y+2**x) + " lev: " + str(lev - x - 1))
            print("recv " + str(y) + " from " + str(y) + " lev: " + str(lev - x - 1))
            print("recv " + str(y+2**x) + " from " + str(y) + " lev: " + str(lev - x - 1))
    return lev


if __name__ == '__main__':
    '''
    p = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
    print(p[0:-1])
    p = copy.deepcopy(simple_polygon(p, 1))
    print(p)
    '''
    bt(4)
