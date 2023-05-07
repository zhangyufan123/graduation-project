from sim import centerpoint, cp
import math
from turning import draw_turning
import matplotlib.pyplot as plt
import shapefile
import numpy as np
from simple import simple_polygon
import copy


def cal(graph, p_result):
    # p_result 表示多边形的对应关系
    #   [(0, 0), (1, 2), (2, 1)]
    #   graph1 中的 p1 与 graph2 中的 p1 对应
    #   graph1 中的 p2 与 graph2 中的 p3 对应
    #   graph1 中的 p2 与 graph2 中的 p2 对应

    p1 = len(graph[0])
    p2 = len(graph[1])

    pp1 = []
    pp2 = []
    # print(p_result)

    for i in p_result:
        x, y = i
        pp1.append(graph[0][x])
        pp2.append(graph[1][y])

    sim = 0
    # point:point
    if p1 == 1 and p2 == 1:
        print('p-p')
        sim = 1
    # point:line
    if (p1 == 1 and p2 == 2) or (p1 == 2 and p2 == 1):
        print('p-l')
        sim = 0.5
    # point:polygon
    if (p1 == 1 and p2 >= 3) or (p1 >= 3 and p2 == 1):
        print('p-po')
        sim = 1 / max(p1, p2)
    # line:line
    if p1 == 2 and p2 == 2:
        print('l-l')
        p11 = centerpoint(graph[0][0])
        p12 = centerpoint(graph[0][1])
        p21 = centerpoint(graph[1][0])
        p22 = centerpoint(graph[1][1])
        d1 = np.sqrt((p11[0] - p12[0]) ** 2 + (p11[1] - p12[1]) ** 2)
        d2 = np.sqrt((p21[0] - p22[0]) ** 2 + (p21[1] - p22[1]) ** 2)

        p0 = [p22[0] - p21[0] + p11[0], p22[1] - p21[1] + p11[1]]
        n = cal_angle(p0, p11, p12)
        print('angle = ' + str(n))
        m = math.cos(math.pi * n / 180)
        m = abs(m)

        sim = m * min(d1, d2) / max(d1, d2)
    # line:polygon
    if (p1 == 2 and p2 >= 2) or (p2 == 2 and p1 >= 2):
        print('l-po')
        # print(p_result[0][1])
        # print(graph[0])
        # print(p_result)
        p11 = centerpoint(graph[0][p_result[0][0]])
        p12 = centerpoint(graph[0][p_result[1][0]])
        # print(p11, p12)
        # 找到对应的多边形
        p21 = centerpoint(graph[1][p_result[0][1]])
        p22 = centerpoint(graph[1][p_result[1][1]])
        # print(p21, p22)
        npc1 = []
        npc2 = []
        pc1 = []
        pc2 = []
        p1x = []
        p1y = []
        p2x = []
        p2y = []
        p3x = []
        p3y = []
        if p1 == 2:
            npc1.append(p11)
            npc1.append(p12)
            p3x.append(p21[0])
            p3x.append(p22[0])
            p3y.append(p21[1])
            p3y.append(p22[1])
            for x in range(p2):
                npc2.append(centerpoint(graph[1][x]))
            npc2.append(npc2[0])
            npc2 = adjust_pts_order(npc2)

            for i in npc1:
                pc1.append((i[0], i[1]))
            for i in npc2:
                pc2.append((i[0], i[1]))
            pc2.append(pc2[0])

            for i in pc1:
                x, y = i
                p1x.append(x)
                p1y.append(y)
            for i in pc2:
                x, y = i
                p2x.append(x)
                p2y.append(y)
        if p2 == 2:
            npc1.append(p21)
            npc1.append(p22)
            p3x.append(p11[0])
            p3x.append(p12[0])
            p3y.append(p11[1])
            p3y.append(p12[1])
            for x in range(p1):
                npc2.append(centerpoint(graph[0][x]))
            npc2.append(npc2[0])
            npc2 = adjust_pts_order(npc2)

            for i in npc1:
                pc1.append((i[0], i[1]))
            for i in npc2:
                pc2.append((i[0], i[1]))
            pc2.append(pc2[0])

            for i in pc1:
                x, y = i
                p1x.append(x)
                p1y.append(y)
            for i in pc2:
                x, y = i
                p2x.append(x)
                p2y.append(y)
        plt.figure()
        #plt.subplot(1, 2, 1)
        plt.plot(p1x, p1y, linewidth=1, color='red')
        #plt.subplot(1, 2, 2)
        plt.plot(p2x, p2y, linewidth=1, color='blue')
        plt.plot(p3x, p3y, linewidth=1, color='green')
        plt.show()

        d1 = np.sqrt((p11[0] - p12[0]) ** 2 + (p11[1] - p12[1]) ** 2)
        d2 = np.sqrt((p21[0] - p22[0]) ** 2 + (p21[1] - p22[1]) ** 2)

        p0 = [p22[0] - p21[0] + p11[0], p22[1] - p21[1] + p11[1]]
        n = cal_angle(p0, p11, p12)
        print('angle = ' + str(n))
        m = math.cos(math.pi * n / 180)
        m = abs(m)

        sim = m * min(d1, d2) / max(d1, d2)
    # polygon:polygon
    if p1 >= 3 and p2 >= 3:
        print('po-po')
        npc1 = []
        npc2 = []
        for x in range(p1):
            npc1.append(cp(graph[0][x]))
        for x in range(p2):
            npc2.append(cp(graph[1][x]))
        #npc1.append(npc1[0])
        #npc2.append(npc2[0])
        #print(npc2)
        #'''
        npc1 = adjust_pts_order(npc1)
        npc2 = adjust_pts_order(npc2)

        #print(npc1)
        tt = 9999999999999999999999999
        tx = 0
        ty = 0
        for a11 in range(len(npc1)):
            for a22 in range(len(npc2)):
                x11, y11 = npc1[a11]
                x22, y22 = npc2[a22]
                #print((x11 - x22) ** 2 + (y11 - y22) ** 2)
                if tt >= (x11 - x22) ** 2 + (y11 - y22) ** 2:
                    tt = (x11 - x22) ** 2 + (y11 - y22) ** 2
                    tx = a11
                    ty = a22
        npc11 = list(npc1[tx:])
        npc11.extend(npc1[0: tx])
        npc22 = list(npc2[ty:])
        npc22.extend(npc2[0: ty])
        # print(npc1)
        # print(npc2)
        # print(npc11)
        # print(npc22)
        pc1 = []
        pc2 = []
        for i in npc11:
            pc1.append((i[0], i[1]))
        for i in npc22:
            pc2.append((i[0], i[1]))
        pc1.append(pc1[0])
        pc2.append(pc2[0])
        # print(pc2)
        # '''
        p1x = []
        p1y = []
        p2x = []
        p2y = []
        for i in pc1:
            x, y = i
            p1x.append(x)
            p1y.append(y)
        for i in pc2:
            x, y = i
            p2x.append(x)
            p2y.append(y)
        plt.figure()
        # plt.subplot(1, 2, 1)
        plt.plot(p1x, p1y, linewidth=1, color='red')
        plt.plot(p2x, p2y, linewidth=1, color='blue')
        plt.show()
        # sim = draw_turning(pc1) - draw_turning(pc2)
        # sim = 1 - np.abs(sim)
        # print(pc1)
        # print(pc2)
        xv1, an1 = draw_turning(pc1)
        xv1 = list(xv1)
        an1 = list(an1)
        xv1, an1 = tun(xv1, an1)
        xv2, an2 = draw_turning(pc2)
        xv2 = list(xv2)
        an2 = list(an2)
        xv2, an2 = tun(xv2, an2)
        plt.xlabel('angle')
        plt.xlabel('length')
        plt.plot(xv1, an1, ls="-", lw=2, label="Turning function d")
        plt.plot(xv2, an2, ls="-", lw=2, label="Turning function c")
        plt.legend()
        plt.show()
        sim = 1 - cal_dif(xv1, an1, xv2,an2)
    return sim


def cal_angle(point_a, point_b, point_c):
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # 点a、b、c的x坐标
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # 点a、b、c的y坐标

    if len(point_a) == len(point_b) == len(point_c) == 3:
        # print("坐标点为3维坐标形式")
        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # 点a、b、c的z坐标
    else:
        a_z, b_z, c_z = 0, 0, 0  # 坐标点为2维坐标形式，z 坐标默认值设为0
        # print("坐标点为2维坐标形式，z 坐标默认值设为0")

    # 向量 m=(x1,y1,z1), n=(x2,y2,z2)
    x1, y1, z1 = (a_x - b_x), (a_y - b_y), (a_z - b_z)
    x2, y2, z2 = (c_x - b_x), (c_y - b_y), (c_z - b_z)

    # 两个向量的夹角，即角点b的夹角余弦值
    cos_b = (x1 * x2 + y1 * y2 + z1 * z2) / (
                math.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) * (math.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)))  # 角点b的夹角余弦值
    B = math.degrees(math.acos(cos_b))  # 角点b的夹角值
    return B


def tun(xvalue, angel):
    x = []
    a = []
    for i in range(1, len(xvalue)):
        x.append(xvalue[i])
    for i in angel:
        a.append(i)
    x.append(1)
    x1 = []
    a1 = []
    for i in range(len(xvalue)):
        x1.append(xvalue[i])
        x1.append(x[i])
        a1.append(angel[i])
        a1.append(a[i])
    return x1, a1


def cal_dif(x1, a1, x2, a2):
    xmin = min(min(a1), min(a2))
    xmax = max(x1)
    s1 = 0
    s2 = 0
    s3 = 0
    cx = []
    cy = []
    xx1 = []
    aa1 = []
    xx2 = []
    aa2 = []
    for x in range(1, len(x1)):
        if a1[x] == a1[x-1]:
            for y in range(1, len(x2)):
                if x2[y] == x2[y - 1]:
                    if (x1[x] > x2[y] > x1[x-1] or x1[x] < x2[y] < x1[x-1]) and (a2[y] > a1[x] > a2[y-1] or a2[y] < a1[x] < a2[y-1]):
                        cx.append(x2[y])
                        cy.append(a1[x])
                        xx1.append(x2[y])
                        aa1.append(a1[x])
            xx1.append(x1[x])
            aa1.append(a1[x])
        '''
        if x1[x] == x1[x-1]: 
            for y in range(1, len(x2)):
                if a2[y] == a2[y - 1]:
                    if (a1[x] > a2[y] > a1[x-1] or a1[x] < a2[y] < a1[x-1]) and (x2[y] > x1[x] > x2[y-1] or x2[y] < x1[x] < x2[y-1]):
                        cx.append(x1[x])
                        cy.append(a2[y])
                        xx2.append(x1[x])
                        aa2.append(a2[y])
            xx1.append(x1[x])
            aa1.append(a1[x])
        '''
    for x in range(1, len(x2)):
        if a2[x] == a2[x-1]:
            for y in range(1, len(x1)):
                if x1[y] == x1[y - 1]:
                    if (x2[x] > x1[y] > x2[x-1] or x2[x] < x1[y] < x2[x-1]) and (a1[y] > a2[x] > a1[y-1] or a1[y] < a2[x] < a1[y-1]):
                        cx.append(x1[y])
                        cy.append(a2[x])
                        xx2.append(x1[y])
                        aa2.append(a2[x])
            xx2.append(x2[x])
            aa2.append(a2[x])
    cx.insert(0, 0)
    cx.append(xmax)
    cx.sort()
    xx1.insert(0, 0)
    aa1.insert(0, 0)
    xx2.insert(0, 0)
    aa2.insert(0, 0)
    #print(xx1, aa1, xx2, aa2)
    for i in range(1, len(cx)):
        ss1 = 0
        ss2 = 0
        for i1 in range(1, len(xx1)):
            if cx[i] >= xx1[i1] > cx[i-1]:
                ss1 += np.abs((xx1[i1] - xx1[i1-1]) * (aa1[i1] - xmin))
        for i2 in range(1, len(xx2)):
            if cx[i] >= xx2[i2] > cx[i-1]:
                ss2 += np.abs((xx2[i2] - xx2[i2-1]) * (aa2[i2] - xmin))

        s1 += ss1
        s2 += ss2
        #print(ss1, ss2, np.abs(ss1 - ss2))
        s3 += np.abs(ss1 - ss2)
    #print(cx, cy)
    #s = 0
    return s3/max(s1, s2)


def adjust_pts_order(pts_2ds):

    ''' sort rectangle points by counterclockwise '''

    cen_x, cen_y = np.mean(pts_2ds, axis=0)
    #refer_line = np.array([10,0])
    d2s = []
    for i in range(len(pts_2ds)):
        o_x = pts_2ds[i][0] - cen_x
        o_y = pts_2ds[i][1] - cen_y
        atan2 = np.arctan2(o_y, o_x)
        dis = o_x ** 2 + o_y ** 2
        if atan2 < 0:
            atan2 += np.pi * 2
        d2s.append([pts_2ds[i], atan2, dis])
    d2s = sorted(d2s, key=lambda x: x[1])
    #print(d2s)
    temp1 = d2s[0][2]
    temp2 = 0
    for i in range(len(d2s)):
        if d2s[i][2] < temp1:
            temp1 = d2s[i][2]
            temp2 = i
    d2ss = d2s[temp2:]
    d2ss.extend(d2s[0: temp2])
    #print(d2ss)
    order_2ds = np.array([x[0] for x in d2ss])
    #print(order_2ds)

    return order_2ds


# a = cal_angle((3**0.5,1), (0,0), (3**0.5,0))  # 结果为 30°
# cal_angle((1,1), (0,0), (1,0))  # 结果为 45°
# cal_angle((-1,1), (0,0), (1,0)) # 结果为 13附上原文出处链接及本声明。
# print(math.sin(math.pi*a/180))
if __name__ == '__main__':

    x1 = [0,5,5,14,14,21]
    a1 = [16,16,12,12,4,4]
    x2 = [0,4,4,10,10,21]
    a2 = [13,13,9,9,2,2]
    '''
    x1 = [0, 2, 2, 5, 5, 9]
    a1 = [5, 5, 3, 3, 0, 0]
    x2 = [0, 4, 4, 6, 6, 9]
    a2 = [4, 4, 1, 1, -1, -1]
    '''
    # print(cal_dif(x1,a1,x2,a2))

    # sf = shapefile.Reader('test/t1/EDU/EDU1/DEU_adm3.shp')
    sf = shapefile.Reader('test/polygon/36.shp')
    # sf = shapefile.Reader('test/simple-polygon/p4/layer.shp')
    # sf = shapefile.Reader('test/position-graph/po-po/huanwei/layer.shp')
    shapes = sf.shapes()
    # sf1 = shapefile.Reader('test/t1/EDU/EDU2/DEU_adm3.shp')
    sf1 = shapefile.Reader('test/polygon/34.shp')
    # sf1 = shapefile.Reader('test/simple-polygon/p4/layer1.shp')
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
    mp[0] = copy.deepcopy(simple_polygon(mp[0], sn))
    mp1[0] = copy.deepcopy(simple_polygon(mp1[0], sn))
    print(len(mp[0]))
    print(len(mp1[0]))
    graph = [mp, mp1]
    npc1 = mp[0][0:-1]
    npc2 = mp1[0][0:-1]
    tt = 9999999999999999999999999
    tx = 0
    ty = 0
    for a11 in range(len(npc1)):
        for a22 in range(len(npc2)):
            x11, y11 = npc1[a11]
            x22, y22 = npc2[a22]
            if tt >= (x11 - x22) ** 2 + (y11 - y22) ** 2:
                tt = (x11 - x22) ** 2 + (y11 - y22) ** 2
                tx = a11
                ty = a22
    sw = 2
    if sw == 1:
        npc11 = list(npc1[tx:])
        npc11.extend(npc1[0: tx])
        npc22 = list(npc2[ty:])
        npc22.extend(npc2[0: ty])
    else:
        npc11 = npc1
        npc22 = npc2
    pc1 = []
    pc2 = []
    for i in npc11:
        pc1.append((i[0], i[1]))
    for i in npc22:
        pc2.append((i[0], i[1]))
    pc1.append(pc1[0])
    pc2.append(pc2[0])
    p1x = []
    p1y = []
    p2x = []
    p2y = []
    for i in pc1:
        x, y = i
        p1x.append(x)
        p1y.append(y)
    for i in pc2:
        x, y = i
        p2x.append(x)
        p2y.append(y)
    ssim = []
    for i in range(len(p1x)):
        #print(pc1)
        pc1.remove(pc1[0])
        pc1.append(pc1[0])
        #print(pc1)
        plt.figure()
        plt.plot(p1x, p1y, linewidth=1, color='red')
        plt.plot(p2x, p2y, linewidth=1, color='blue')
        plt.show()
        xv1, an1 = draw_turning(pc1)
        xv1 = list(xv1)
        an1 = list(an1)
        xv1, an1 = tun(xv1, an1)
        xv2, an2 = draw_turning(pc2)
        xv2 = list(xv2)
        an2 = list(an2)
        xv2, an2 = tun(xv2, an2)
        plt.xlabel('angle')
        plt.xlabel('length')
        plt.plot(xv1, an1, ls="-", lw=2, label="Turning function d")
        plt.plot(xv2, an2, ls="-", lw=2, label="Turning function c")
        plt.legend()
        plt.show()
        sim = 1 - cal_dif(xv1, an1, xv2, an2)
        ssim.append(sim)
    print(max(ssim))
