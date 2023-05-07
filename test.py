import time

import numpy as np
import shapefile
from simple import simple_polygon
from neighbor_support import nei_matrix
from position import cal
from poufen import work
from sim import cal_sim

time_start = time.time()
# crs_WGS84 = CRS.from_epsg(4326)
# crs_WebMercator = CRS.from_epsg(3857)

# sf = shapefile.Reader('test/t1/EDU/EDU1/DEU_adm3.shp')
# sf = shapefile.Reader('test/test1.shp')
# sf = shapefile.Reader('test/position-graph/huanwei/layer2.shp')
# sf = shapefile.Reader('test/position-graph/xuanzhuan/layer3.shp')
sf = shapefile.Reader('test/small-polygon/p5/layer.shp')
# sf = shapefile.Reader('test/general/1/layer.shp')
shapes = sf.shapes()
# sf1 = shapefile.Reader('test/t1/EDU/EDU2/DEU_adm3.shp')
# sf1 = shapefile.Reader('test/test3.shp')
# sf1 = shapefile.Reader('test/position-graph/huanwei/layer1.shp')
# sf1 = shapefile.Reader('test/position-graph/xuanzhuan/layer4.shp')
sf1 = shapefile.Reader('test/small-polygon/p5/DEU_adm3.shp')
# sf1 = shapefile.Reader('test/general/1/layer1.shp')
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
mps1 = []
mps2 = []
sn = 10
for i in mp:
    mps1.append(simple_polygon(i, sn))
for i in mp1:
    mps2.append(simple_polygon(i, sn))
graph = [mps1, mps2]
len1 = len(mp)
len2 = len(mp1)

# NS矩阵
ns_data, ns_result = nei_matrix(graph)
# 复杂度矩阵
comp = []
c0 = []
c1 = []
for m in range(len1):
    p1 = np.array(graph[0][m])
    c0.append(work(p1, m))
for n in range(len2):
    p2 = np.array(graph[1][n])
    c1.append(work(p2, n))
for m in range(len1):
    col = []
    for n in range(len2):
        data = (c0[m] + c1[n]) / 2
        col.append(data)
    comp.append(col)
'''
for m in range(len1):
    col = []
    p1 = np.array(graph[0][m])
    for n in range(len2):
        p2 = np.array(graph[1][n])
        data = (work(p1) + work(p2)) / 2
        col.append(data)
    comp.append(col)
'''
# 相似度矩阵
sim = []
for m in range(len1):
    col = []
    p1 = np.array(graph[0][m])
    for n in range(len2):
        p2 = np.array(graph[1][n])
        data = cal_sim(p1, p2)
        col.append(data)
    sim.append(col)

im = []
ns = 0
test = 0
ix, iy = 0, 0
for m in range(len1):
    col = []
    for n in range(len2):
        i = ns_data[m][n] * comp[m][n] * sim[m][n]
        ns = ns_data[m][n]
        if ns >= test:
            test = ns
            ix = m
            iy = n
        col.append(i)
    im.append(col)

# 多边形对应关系
p_coo = ns_result[ix][iy]
print(p_coo)

# 位置图相似度
psim = cal(graph, p_coo)
# print(p_coo)

tsim = 0
for i in range(len(p_coo)):
    x, y = p_coo[i]
    data = cal_sim(mp[x], mp1[y])
    tsim += data

# wp, ws 需要自己设置
wp = 0.5
ws = 0.5
total_sim = wp * psim + ws * tsim / min(len1, len2)
print('Neighbor support matrix:')
print(ns_data)
print('Similarity matrix:')
print(sim)
print('Complexity matrix:')
print(comp)
print('Importance matrix:')
print(im)

print('position sim = ' + str(psim))
print('mean value of polygon sim = ' + str(tsim / min(len1, len2)))
print('the sim = ' + str(total_sim))
time_end = time.time()
time_sum = time_end - time_start
print('xxx {:.5f} s'.format(time_sum))
