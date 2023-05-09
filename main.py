import math

import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np


def powerlaw(n, gamma, mint=2):
    x0 = mint
    x1 = n-1
    # x  = ((x1^(alpha+1) - x0^(alpha+1))*y + x0^(alpha+1))^(1/(alpha+1))
    out = []
    for i in range(n):
        y = random.random()
        out.append(round((y * (x1 ** (1 - gamma) - x0 ** (1 - gamma)) + x0 ** (1 - gamma)) ** (1 / (1 - gamma))))
    while sum(out) % 2 != 0:
        out.pop()
        y = random.random()
        out.append(round((y * (x1 ** (1 - gamma) - x0 ** (1 - gamma)) + x0 ** (1 - gamma)) ** (1 / (1 - gamma))))
    return out


def constant(n, c):
    return [c for i in range(n)]


def uniform(n):
    return [i for i in range(1, n)]


# def cmodel(ds):
#     G = nx.G()
#     G.add_nodes_from(range(len(ds)))
#     stubs = []
#     # print(ds)
#     for index, degree in enumerate(ds):
#         stubs += [index for j in range(degree)]
#     # print(stubs)
#     random.shuffle(stubs)
#     while len(stubs) > 0:
#         a = stubs.pop()
#         b = stubs.pop()
#         G.add_edge(a, b)
#     return G


def path(graph):
    nodes = random.sample(list(graph), 2)
    return nx.shortest_path(graph, source=nodes[0], target=nodes[1])


def path2(graph):
    nodes = random.sample(list(graph), 2)
    current = [nodes[0]]
    # print(nodes)
    while nodes[1] not in graph.neighbors(current[-1]):
        most = [0, []]
        for n in graph.neighbors(current[-1]):
            x = sum([1 for n2 in graph.neighbors(n)])
            if x > most[0] and n not in current:
                most[0] = x
                most[1] = [n]
            elif x == most[0] and n not in current:
                most[1].append(n)
        if len(most[1]) == 0:
            print("fail")
            return "0"
        current.append(most[1][0])
        # print(current)
    current.append(nodes[1])
    # print(current)
    return current


def path3(graph, t=True):
    if t:
        nodes = random.sample(list(graph), 2)
        current1 = [nodes[0]]
        current2 = [nodes[1]]
    else:
        nodes = random.sample([i for i in range(10)], 2)
        current1 = [nodes[0]]
        current2 = [nodes[1]]
    # print(current1, current2)
    while not graph.has_edge(current1[-1], current2[-1]):
        most1 = [0, []]
        most2 = [0, []]
        for n1 in graph.neighbors(current1[-1]):
            x = sum([1 for n11 in graph.neighbors(n1)])
            if x > most1[0] and n1 not in current1:
                most1[0] = x
                most1[1] = [n1]
        if len(most1[1]) == 0:
            print("fail")
            return "failed"
        current1.append(random.sample(most1[1], 1)[0])
        if graph.has_edge(current1[-1], current2[-1]):
            return current1 + current2[::-1]
        for n2 in graph.neighbors(current2[-1]):
            x = sum([1 for n22 in graph.neighbors(n2)])
            if x > most2[0] and n2 not in current2:
                most2[0] = x
                most2[1] = [n2]
        if len(most2[1]) == 0:
            print("fail")
            return "failed"
        current2.append(random.sample(most2[1], 1)[0])
        # print(current1, current2)
    return current1 + current2[::-1]


def path4(graph):
    nodes = random.sample(list(graph), 2)
    current1 = [nodes[0]]
    current2 = [nodes[1]]
    # print(current1, current2)
    for i in range(1000):
        if graph.has_edge(current1[-1], current2[-1]):
            return current1 + current2

        current1.append(random.sample([n for n in graph.neighbors(current1[-1])], 1)[0])
        if graph.has_edge(current1[-1], current2[-1]):
            return current1 + current2

        current2.append(random.sample([n for n in graph.neighbors(current2[-1])], 1)[0])
        # print(current1, current2)
    return "failed"


def path5(graph, a, b, visited={}, steps=1):
    if graph.has_edge(a, b):
        return steps
    x = [n for n in graph.neighbors(a)]
    if len(x) == 0:
        return "failed"
    random.seed(1)
    c = random.sample(x, 1)[0]
    return path5(graph, b, c, visited, steps + 1)


def path6(graph, a, b, visited={}, steps=1):
    if graph.has_edge(a, b):
        return steps
    visited[a] = True
    visited[b] = True
    most = [0, []]
    for neighbor in graph.neighbors(a):
        x = sum([1 for n in graph.neighbors(neighbor)])
        if x > most[0] and neighbor not in visited:
            most[0] = x
            most[1] = [neighbor]
        elif x == most[0] and neighbor not in visited:
            most[1].append(neighbor)
    if len(most[1]) == 0:
        return "failed"
    random.seed(1)
    c = random.sample(most[1], 1)[0]
    return path5(graph, b, c, visited, steps + 1)


def path7(graph):
    nodes = random.sample(list(graph), 2)
    current1 = [nodes[0]]
    current2 = [nodes[1]]
    d = [(sum([1 for n in graph.neighbors(current1[0])]) + sum([1 for n in graph.neighbors(current2[0])])) / 2]
    # print(current1, current2)
    while not graph.has_edge(current1[-1], current2[-1]):
        most1 = [0, []]
        most2 = [0, []]
        for n1 in graph.neighbors(current1[-1]):
            x1 = sum([1 for n11 in graph.neighbors(n1)])
            if x1 > most1[0] and n1 not in current1:
                most1[0] = x1
                most1[1] = [n1]
        if len(most1[1]) == 0:
            print("fail")
            return "failed"
        current1.append(random.sample(most1[1], 1)[0])
        d.append(most1[0] / 2)
        if graph.has_edge(current1[-1], current2[-1]):
            return current1 + current2[::-1]
        for n2 in graph.neighbors(current2[-1]):
            x2 = sum([1 for n22 in graph.neighbors(n2)])
            if x2 > most2[0] and n2 not in current2:
                most2[0] = x2
                most2[1] = [n2]
        if len(most2[1]) == 0:
            print("fail")
            return "failed"
        current2.append(random.sample(most2[1], 1)[0])
        d[-1] += most2[0] / 2
        # print(current1, current2)
    return d


def color(graph, nodes):
    colors = []
    for node in graph.nodes():
        if node == nodes[0]:
            colors.append('green')
        elif node == nodes[-1]:
            colors.append('red')
        elif node in nodes:
            colors.append('yellow')
        else:
            colors.append('blue')
    return colors


def sim(n, t=True):
    out = []
    out2 = []
    fail = []
    for i in range(11, 36):
        print(i)
        travel = 0
        n2 = 10
        gamma = i / 10
        ds = 0
        count = 1
        fail.append(0)
        for j in range(n2):
            random.seed(j)
            d = powerlaw(1000, gamma)
            d.sort(reverse=True)
            ds += sum(d) / len(d)
            graph = nx.configuration_model(d)
            while not nx.is_connected(graph):
                graph = nx.configuration_model(d)
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            for k in range(n):
                x = path3(graph, t)
                if not x == "failed":
                    travel += len(x)
                    count += 1
                else:
                    fail[-1] += 1 / (n * n2)
        out.append(travel / count)
        out2.append(ds / n2)
    return out, out2, fail


def sim2(n):
    out = []
    out2 = []
    for i in range(3, 12):
        print(i)
        p = i / 1000
        out2.append(i)
        n2 = 10
        travel = 0
        count = 1
        for j in range(n2):
            random.seed(j)
            graph = nx.binomial_graph(1000, p)
            for k in range(n):
                x = path3(graph)
                if not x == "failed":
                    travel += len(x)
                    count += 1
        out.append(travel / count)
    return out, out2


def sim3(n):
    out = []
    out2 = []
    for i in range(3, 13):
        print(i)
        n2 = 10
        travel = 0
        d = constant(2000, i)
        count = 0
        for j in range(n2):
            random.seed(j)
            graph = nx.configuration_model(d)
            while not nx.is_connected(graph):
                graph = nx.configuration_model(d)
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))
            for k in range(n):
                x = path4(graph)
                if not x == "failed":
                    travel += len(x)
                    count += 1
        out.append(travel / count)
        out2.append(1 - count / (n * n2))
    return out, out2


def pp():
    s = powerlaw(100, 2.2)
    plt.hist(s)
    plt.show()


def path8(graph):
    nodes = random.sample(list(graph), 2)
    a = [nodes[0]]
    b = [nodes[1]]
    out = []
    print(a, b)
    for i in range(100):
        if graph.has_edge(a[-1], b[-1]):
            print("done")
            break
        neighborsa = [n for n in graph.neighbors(a[-1])]  # if n not in a]
        neighborsb = [n for n in graph.neighbors(b[-1])]  # if n not in a]
        if not neighborsa or not neighborsb:
            break
        # print(neighbors)
        max_degreea = max(graph.degree(n) for n in neighborsa)
        max_degreeb = max(graph.degree(n) for n in neighborsb)
        neighbors_with_max_degreea = [n for n in neighborsa if graph.degree(n) == max_degreea]
        neighbors_with_max_degreeb = [n for n in neighborsb if graph.degree(n) == max_degreeb]
        # print(neighbors_with_max_degree)
        chosena = random.sample(neighbors_with_max_degreea, 1)[0]
        # print(chosena)
        a.append(chosena)
        if graph.has_edge(a[-1], b[-1]):
            print("done")
            break
        chosenb = random.sample(neighbors_with_max_degreeb, 1)[0]
        b.append(chosenb)
        out.append((max_degreea + max_degreeb) / 2)
    return out


def g7():
    d1 = powerlaw(1000, 2.33, 2)
    d2 = powerlaw(1000, 2.6, 3)
    d3 = powerlaw(1000, 3, 4)
    d4 = powerlaw(1000, 3.67, 5)
    print(sum(d1), sum(d2), sum(d3), sum(d4))

    graph1 = nx.configuration_model(d1)
    graph1 = nx.Graph(graph1)
    graph1.remove_edges_from(nx.selfloop_edges(graph1))

    graph2 = nx.configuration_model(d2)
    graph2 = nx.Graph(graph2)
    graph2.remove_edges_from(nx.selfloop_edges(graph2))

    g3 = nx.configuration_model(d3)
    g3 = nx.Graph(g3)
    g3.remove_edges_from(nx.selfloop_edges(g3))

    g4 = nx.configuration_model(d4)
    g4 = nx.Graph(g4)
    g4.remove_edges_from(nx.selfloop_edges(g4))

    count = [0, 0, 0, 0]
    fr = [0, 0, 0, 0]

    n = 1000

    for i in range(n):
        x = [path3(graph1), path3(graph2), path3(g3), path3(g4)]
        for j, item in enumerate(x):
            if item == "failed":
                fr[j] += 1
            else:
                count[j] += len(item)

    for i in range(4):
        count[i] /= (n - fr[i])

    print(fr)
    plt.scatter([2.33, 2.6, 3, 4], count)
    plt.savefig("graph7.png")


def g1():
    aa = sim(100)
    print(aa)
    plt.scatter(aa[1], aa[0])
    # plt.show()
    plt.savefig("graph1.png")


def g2():
    aa = sim(100)
    print(aa)
    ab = [i / 10 for i in range(11, 36)]
    plt.scatter(ab, aa[0])
    # plt.show()
    plt.savefig("graph9.png")


def er():
    graph = nx.erdos_renyi_graph(50, 0.5)
    nx.draw_circular(graph)
    plt.savefig("er05.png")


def g6():
    aa = sim3(100)
    bb = [i for i in range(3, 13)]
    bbb = [i / 10 for i in range(30, 130)]
    cc = [2000 / dd for dd in bbb]
    plt.scatter(bb, aa[0], label="Actual")
    plt.plot(bbb, cc, label="Expected")
    print(aa[1])
    plt.legend()
    plt.savefig("graph66.png")
    # plt.show()


def g8():
    aa = sim(10)
    ab = [i / 10 for i in range(21, 36)]
    plt.scatter(ab, aa[0])
    for i, s in enumerate(aa[2]):
        plt.text(ab[i]+0.02, aa[0][i]+0.3, 1 - round(s, 2))
    plt.show()


def sim4(n):
    out = []
    out2 = []
    fail = []
    for i in range(21, 22):
        print(i)
        travel = 0
        n2 = 1
        gamma = i / 10
        ds = 0
        count = 1
        fail.append(0)
        for j in range(n2):
            random.seed(j)
            d = powerlaw(1000, gamma)
            d.sort(reverse=True)
            graph = nx.configuration_model(d)
            while not nx.is_connected(graph):
                graph = nx.configuration_model(d)
            graph = nx.Graph(graph)
            graph.remove_edges_from(nx.selfloop_edges(graph))


g2()
