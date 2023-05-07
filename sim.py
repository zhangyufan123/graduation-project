# from numpy import fft, ifft
import shapefile
import numpy as np
from simple import simple_polygon
import copy
import time


# 计算多边形的中心点坐标
def centerpoint(vertex):
    vlen = len(vertex)
    px = 0
    py = 0
    for i in range(vlen - 1):
        x, y = vertex[i]
        px = px + x
        py = py + y
    px = px / (vlen - 1)
    py = py / (vlen - 1)
    return px, py


def cp(vertex):
    lx = []
    ly = []
    for i in vertex:
        xx, yy = i
        lx.append(xx)
        ly.append(yy)
    px = np.mean(lx)
    py = np.mean(ly)
    return px, py


def centroid(vertex):
    area = 0
    cx = 0
    cy = 0
    vlen = len(vertex)
    x0, y0 = vertex[0]

    for i in range(1, vlen - 1):
        x1, y1 = vertex[i]
        x2, y2 = vertex[i + 1]

        v1x = x1 - x0
        v1y = y1 - y0
        v2x = x2 - x0
        v2y = y2 - y0

        A = (v1x * v2y - v2x * v1y) / 2
        area += A

        x = v1x + v2x
        y = v1y + v2y
        cx += A * x
        cy += A * y

    cenx = cx / area / 3 + x0
    ceny = cy / area / 3 + y0
    return cenx, ceny


# 求最远点距离
def farthestPoint(cx, cy, px, py, vertex):
    vlen = len(vertex)
    dis = 0
    ptx = 0
    pty = 0
    for i in vertex:
        x, y = i
        d = (x - cx) * (x - cx) + (y - cy) * (y - cy)
        if d > dis:
            dis = d
            ptx = x
            pty = y
    d1 = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
    d2 = np.sqrt((ptx - px) ** 2 + (pty - py) ** 2)
    return d1 + d2


# 傅里叶变换
def cal_fourier(vertex):
    cenx, ceny = centroid(vertex)
    inp = []
    for i in vertex:
        x, y = i
        # print(x,y)
        dis = farthestPoint(x, y, cenx, ceny, vertex)
        inp.append(dis)
    # print(inp)
    # print(inp)
    inp_f = np.fft.ifft(inp, 20)
    inp_f = np.abs(inp_f)
    # print(inp_f)
    return np.array(inp_f)


# 计算相似度
def cal_sim(vertex1, vertex2):
    f1 = cal_fourier(vertex1[0:-1])
    f2 = cal_fourier(vertex2[0:-1])
    f1m = []
    f2m = []
    c0 = 0
    c1 = 0
    for i in f1:
        c0 += i ** 2
    for i in f2:
        c1 += i ** 2
    for i in f1:
        f1m.append(i / np.sqrt(c0))
    for i in f2:
        f2m.append(i / np.sqrt(c1))
    dif = 0
    m = min(len(f1m), len(f2m))
    for i in range(m):
        res = (f1m[i] - f2m[i]) * (f1m[i] - f2m[i])
        dif = dif + res
    return 1 - np.sqrt(dif)

def scale_dif(vertex1, vertex2):
    c1x, c1y = centerpoint(vertex1)
    c2x, c2y = centerpoint(vertex2)
    dis1 = 0
    dis2 = 0
    for v1 in vertex1:
        v1x, v1y = v1
        dis1 += np.sqrt((v1x - c1x)**2 + (v1y - c1y)**2)
    for v2 in vertex2:
        v2x, v2y = v2
        dis2 += np.sqrt((v2x - c2x)**2 + (v2y - c2y)**2)
    sim = min(dis1/len(vertex1), dis2/len(vertex2)) / max(dis1/len(vertex1), dis2/len(vertex2))
    #print(dis1/len(vertex1), dis2/len(vertex2))
    return sim


if __name__ == '__main__':

    time_start = time.time()

    # sf = shapefile.Reader('test/t1/EDU/EDU1/DEU_adm3.shp')
    sf = shapefile.Reader('test/polygon/43.shp')
    # sf = shapefile.Reader('test/simple-polygon/p2/layer.shp')
    # sf = shapefile.Reader('test/position-graph/po-po/huanwei/layer.shp')
    shapes = sf.shapes()
    # sf1 = shapefile.Reader('test/t1/EDU/EDU2/DEU_adm3.shp')
    sf1 = shapefile.Reader('test/polygon/42.shp')
    # sf1 = shapefile.Reader('test/simple-polygon/p2/layer.shp')
    # sf1 = shapefile.Reader('test/position-graph/po-po/huanwei/layer1.shp')
    shapes1 = sf1.shapes()
    mp = []
    for p in shapes:
        pts = p.points
        x, y = zip(*pts)
        polygon = []
        for i in range(len(x)):
            vertex = [x[i], y[i]]
            polygon.append(vertex)
        mp.append(polygon)
    mp1 = []
    for p in shapes1:
        pts = p.points
        x, y = zip(*pts)
        polygon = []
        for i in range(len(x)):
            vertex = [x[i], y[i]]
            polygon.append(vertex)
        mp1.append(polygon)
    sn = 1
    numl = [1, 2, 3, 5, 10, 15, 20, 25, 30]
    mp[0] = copy.deepcopy(simple_polygon(mp[0], 1))
    mp1[0] = copy.deepcopy(simple_polygon(mp1[0], sn))
    print(len(mp[0]))
    print(len(mp1[0]))
    graph = [mp, mp1]

    #print(cal_sim(graph[0][0], graph[1][0]))
    print(scale_dif(graph[0][0], graph[1][0]))

    time_end = time.time()
    time_sum = time_end - time_start
    print('xxx {:.5f} s'.format(time_sum))
