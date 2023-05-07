import matplotlib.pyplot as plt
import numpy as np
import random
from shapely.geometry import Polygon, LinearRing  # 多边形模型，和线性环模型
from shapely.geometry.polygon import LinearRing

def IsSimplePoly(poly):
    poly_ring = poly.boundary
    if poly_ring.is_ring and list(poly.interiors) == []:
        return True
    else:
        return False


def GetPolyVex(poly):
    return np.asarray(poly.exterior)


def VexCCW(poly):
    return 1 if LinearRing(poly.exterior).is_ccw else -1


def GetDivideVexIdx(poly):
    dividevex_idx_li = []
    dividevex_arg_li = []
    vex_arr = GetPolyVex(poly)
    vex_arr = vex_arr.tolist()
    nums = len(vex_arr.coords) - 1
    vex_arr = list(vex_arr.coords)
    vex_arr = np.array(vex_arr)
    vex_arr = vex_arr[0:-1]
    if nums <= 3:
        return vex_arr, dividevex_idx_li, dividevex_arg_li

    pm = VexCCW(poly)
    for i in range(nums):
        v = vex_arr[i, :]
        l = vex_arr[i - 1, :]
        r = vex_arr[(i + 1) % nums, :]
        fir_vector = v - l
        sec_vector = r - v
        A = np.array([fir_vector, sec_vector])
        if pm * np.linalg.det(A) > 0:
            remainvex_arr = np.concatenate([vex_arr[:i, :], vex_arr[i + 1:, :]], axis=0)
            remain_poly = Polygon(remainvex_arr)
            tri = Polygon([l, v, r])
            if (remain_poly.is_valid
                    and remain_poly.intersection(tri).area < 1e-8
                    and poly.equals(remain_poly.union(tri))):

                dividevex_idx_li.append(i)
                arc = np.arccos(
                    -np.dot(fir_vector, sec_vector) / np.linalg.norm(fir_vector) / np.linalg.norm(sec_vector))
                dividevex_arg_li.append(arc)
    return vex_arr, dividevex_idx_li, dividevex_arg_li


def GetDivTri(poly, tris=[]):
    vex_arr, dv_idx_li, dv_arc_li = GetDivideVexIdx(poly)
    nums = len(vex_arr)
    if nums <= 3:
        tris.append(poly)
        return tris
    idx = dv_idx_li[np.argmin(np.array(dv_arc_li))]
    v = vex_arr[idx, :]
    l = vex_arr[idx - 1, :]
    r = vex_arr[(idx + 1) % nums, :]
    tri = Polygon([l, v, r])
    tris.append(tri)

    remain_vex_arr = np.concatenate([vex_arr[:idx, :], vex_arr[idx + 1:, :]], axis=0)
    remain_poly = Polygon(remain_vex_arr)
    GetDivTri(remain_poly, tris)
    return tris


def PolyPretreatment(poly_arr):
    temp = poly_arr - np.min(poly_arr, axis=0)
    return temp / np.max(temp)


def MinAngle(tri):
    point = np.asarray(tri.exterior)
    point = point.tolist()
    point = list(point.coords)
    point = np.array(point)
    arc_li = []
    for i in range(3):
        j = (i + 1) % 3;
        k = (i + 2) % 3
        a = np.linalg.norm(point[i, :] - point[j, :])
        b = np.linalg.norm(point[j, :] - point[k, :])
        c = np.linalg.norm(point[k, :] - point[i, :])
        arc = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
        arc_li.append(arc)
    return min(arc_li)


def OptDiv(poly4_vex_arr):
    tri1 = Polygon(poly4_vex_arr[[0, 1, 2]])
    tri2 = Polygon(poly4_vex_arr[[0, 2, 3]])
    arc1 = min([MinAngle(tri1), MinAngle(tri2)])

    tri3 = Polygon(poly4_vex_arr[[0, 1, 3]])
    tri4 = Polygon(poly4_vex_arr[[1, 2, 3]])
    arc2 = min([MinAngle(tri3), MinAngle(tri4)])

    if arc1 >= arc2:
        return tri1, tri2
    else:
        return tri3, tri4


def OptAlltris(tris):
    random.shuffle(tris)
    nums = len(tris)
    for i in range(nums):
        tri_i = tris[i]
        for j in range(i + 1, nums):
            tri_j = tris[j]
            if tri_i.intersection(tri_j).length > 1e-10:
                u = tri_i.union(tri_j)
                vex_arr, dv_vex_li, _ = GetDivideVexIdx(u)
                if len(dv_vex_li) == 4:
                    a, b = OptDiv(vex_arr)
                    flag = True
                    for idx in set(range(nums)) - {i, j}:
                        if a.intersection(tris[idx]).area > 0. or b.intersection(tris[idx]).area > 0.:
                            flag = False
                    if flag:
                        tris[i], tris[j] = a, b

    return tris

def draw_skeleton(tris):
    #print(tris)
    t = []
    lxx = []
    lyy = []
    for i in tris:
        num = 0
        xa, xb, xc, xd = i.exterior.xy[0]
        ya, yb, yc, yd = i.exterior.xy[1]
        po1 = []
        po2 = []
        for m in tris:
            count = 0
            xan, xbn, xcn, xdn = m.exterior.xy[0]
            yan, ybn, ycn, ydn = m.exterior.xy[1]
            po = []
            if (xa==xan and ya==yan) or (xa==xbn and ya==ybn) or (xa==xcn and ya==ycn):
                count = count + 1
                po1.append([xa, ya])
                po.append([xa, ya])
            if (xb==xan and yb==yan) or (xb==xbn and yb==ybn) or (xb==xcn and yb==ycn):
                count = count + 1
                po1.append([xb, yb])
                po.append([xb, yb])
            if (xc==xan and yc==yan) or (xc==xbn and yc==ybn) or (xc==xcn and yc==ycn):
                count = count + 1
                po1.append([xc, yc])
                po.append([xc, yc])
            po2.append(po)
            if count >= 2:
                num = num + 1
        if [xa, ya] in po1:
            po1.remove([xa, ya])
        if [xb, yb] in po1:
            po1.remove([xb, yb])
        if [xc, yc] in po1:
            po1.remove([xc, yc])
        #print(num)
        num = num - 1
        if num==0:
            lx = []
            ly = []
            cx = (xa + xb + xc) / 3
            cy = (ya + yb + yc) / 3
            lx.append(xa)
            lx.append(cx)
            ly.append(ya)
            ly.append(cy)
            lxx.append(lx)
            lyy.append(ly)
        if num==1:
            lx = []
            ly = []
            mid = []
            if [xa, ya] not in po1:
                lx.append(xa)
                ly.append(ya)
            else:
                mid.append([xa, ya])
            if [xb, yb] not in po1:
                lx.append(xb)
                ly.append(yb)
            else:
                mid.append([xb, yb])
            if [xc, yc] not in po1:
                lx.append(xc)
                ly.append(yc)
            else:
                mid.append([xc, yc])
            lx.append((mid[0][0] + mid[1][0]) / 2)
            ly.append((mid[0][1] + mid[1][1]) / 2)
            lxx.append(lx)
            lyy.append(ly)
        if num==2:
            lx = []
            ly = []
            for i in po2:
                if len(i)==2:
                    lx.append((i[0][0] + i[1][0]) / 2)
                    ly.append((i[0][1] + i[1][1]) / 2)
            lxx.append(lx)
            lyy.append(ly)
        if num==3:
            cx = (xa + xb + xc) / 3
            cy = (ya + yb + yc) / 3
            mx1 = (xa + xb) / 2
            mx2 = (xa + xc) / 2
            mx3 = (xc + xb) / 2
            my1 = (ya + yb) / 2
            my2 = (ya + yc) / 2
            my3 = (yc + yb) / 2
            lxx.append([cx, mx1])
            lxx.append([cx, mx2])
            lxx.append([cx, mx3])
            lyy.append([cy, my1])
            lyy.append([cy, my2])
            lyy.append([cy, my3])
    return lxx, lyy

def distance(x, y):
    return np.sqrt((x[0]-x[1])**2+(y[0]-y[1])**2)

def work(poly_array, order):
    poly = Polygon(poly_array)
    # 运算,绘图脚本
    if IsSimplePoly(poly):
        plt.figure(figsize=(16, 8))
        tris = []
        tris = GetDivTri(poly, tris=tris)
        tris = OptAlltris(tris)
        tris = OptAlltris(tris)
        # 用mpl画出，原来图形的线框
        #'''
        plt.subplot(1, 2, 1)
        plt.ylabel(str(order))
        plt.plot(*poly.exterior.xy, label=order)
        plt.axis("equal")
        # 用线框画出剖分
        plt.subplot(1, 2, 2)
        #'''
        x, y = draw_skeleton(tris)
        dis_all = 0
        for i in range(len(x)):
            plt.plot(x[i], y[i], color='r')
            plt.scatter(x[i], y[i], color='b')
            d = distance(x[i], y[i])
            dis_all += d
        #'''
        for tri in tris:  # triangulate得到的所有三角形，这是对凸包的一个划分
            plt.plot(*tri.exterior.xy)
        plt.axis("equal")
        plt.show()
        #'''
        perimeter = 0
        for i in range(len(poly_array)):
            n = len(poly_array)
            if i < n-1:
                d = np.sqrt((poly_array[i+1][0]-poly_array[i][0])**2+(poly_array[i+1][1]-poly_array[i][1])**2)
                perimeter += d
        return dis_all/perimeter

        #plt.show()
    else:
        print("输入的多边形，不是定义要求的简单多边形！")

if __name__ == '__main__':
    ##-----------------------------------------------##
    #p = np.array([(0, 0), (1, 2), (0, 4), (3, 6), (2, 5), (3, 5), (4, 3), (0, 0)])  # 顶点序列
    p = np.array([(1, 2), (0, 4), (3, 6), (1, 2)])
    #poly = Polygon(PolyPretreatment(poly_arr))  # 构造多边形
    #p = np.array([(-12822783.649064405, 4291880.018524872), (-12822380.964580357, 4291857.251910932), (-12822227.289936269, 4291356.386404273), (-12822873.29260679, 4291500.100654763), (-12822783.649064405, 4291880.018524872)])
    print(work(p))