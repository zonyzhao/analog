##
# @file   readgraph.py
# @author Yibo Lin
# @date   Feb 2020
#

import os
import sys
import pdb
import numpy as np
import networkx as nx
import pickle
import networkx as nx
from itertools import combinations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import random
import math
import json

class SpiceEntry (object):
    def __init__(self):
        self.name = ""
        self.pins = []
        self.cell = None
        self.attributes = {}

    def __str__(self):
        content = "name: " + self.name
        content += "; pins: " + " ".join(self.pins)
        content += "; cell: " + self.cell
        content += "; attr: " + str(self.attributes)
        return content

    def __repr__(self):
        return self.__str__()

class SpiceSubckt (object):
    def __init__(self):
        self.name = ""
        self.pins = []
        self.entries = []

    def __str__(self):
        content = "subckt: " + self.name + "\n"
        content += "pins: " + " ".join(self.pins) + "\n"
        content += "entries: \n";
        for entry in self.entries:
            content += str(entry) + "\n"
        return content

class SpiceNode (object):
    def __init__(self):
        self.id = None
        self.attributes = {} # include name (named in hierarchy), cell
        self.pins = []
    def __str__(self):
        content = "SpiceNode( " + str(self.id) + ", " + str(self.attributes) + ", " + str(self.pins) + " )"
        return content
    def __repr__(self):
        return self.__str__()

class SpiceNet (object):
    def __init__(self):
        self.id = None
        self.attributes = {} # include name
        self.pins = []
    def __str__(self):
        content = "SpiceNet( " + str(self.id) + ", " + str(self.attributes) + ", " + str(self.pins) + " )"
        return content
    def __repr__(self):
        return self.__str__()

class SpicePin (object):
    def __init__(self):
        self.id = None
        self.node_id = None
        self.net_id = None
        self.attributes = {} # include type
    def __str__(self):
        content = "SpicePin( " + str(self.id) + ", node: " + str(self.node_id) + ", net: " + str(self.net_id) + " attributes: " + str(self.attributes) + " )"
        return content
    def __repr__(self):
        return self.__str__()

class SpiceGraph (object):
    def __init__(self):
        self.nodes = []
        self.pins = []
        self.nets = []
    def __str__(self):
        content = "SpiceGraph\n"
        for node in self.nodes:
            content += str(node) + "\n"
        for pin in self.pins:
            content += str(pin) + "\n"
        for net in self.nets:
            content += str(net) + "\n"
        return content

def draw_graph(G, labels, color):
    color_map = []
    for node in G:
        flag = 0
        for i in range(len(color)):
            if node in color[i]:
                color_map.append(10*i+10)
                flag = 1
        if flag == 0:
            color_map.append(10*len(color)+10)
    #for node in G:
        #if node in color:
            #color_map.append('skyblue')
        #else:
            #color_map.append('green')
    pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    options = {'arrowstyle': '-|>', 'arrowsize': 12}
    nx.draw(G, font_weight='bold', pos=pos, node_color=color_map, **options, cmap=plt.cm.Blues)
    nx.draw_networkx_labels(G,pos,labels,font_size=16)
    plt.savefig('graph.pdf', dpi=120)

def convert(integer, length):
    bool_list = [0] * length
    bool_list[integer] = 1
    return bool_list

def type_rule(type1, type2):
    types1 = ['diode', 'res', 'cap']
    types2 = ['pfet', 'nfet', 'pfet_lvt', 'nfet_lvt']
    if type1 in types1:
        return (type1 == type2)
    if type1 in types2:
        return (type2 in types2)
    return 0

def type_rule2(type1, type2):
    types1 = ['pfet', 'pfet_lvt']
    types2 = ['nfet', 'nfet_lvt']
    if type1 in types1:
        return type2 in types1
    if type1 in types2:
        return type2 in types2
    return 0

def readckts(filename):
    """parse: graph, pairs, label
    """
    with open(filename, "rb") as f:
        dataX, dataY = pickle.load(f)

    G = []  # list of graphs for subckts
    all_pairs = []  # list of proper pairs for subckts
    all_type = []
    for i in range(len(dataX)):
        sub_G = nx.Graph()
        node_pair = []
        num_nodes = 0
        
        subckts = dataX[i]["subckts"]
        graph = dataX[i]["graph"]

        for g in graph.nodes:
            sub_G.add_node(g.id)
            if g.attributes['cell'] not in all_type:
                all_type.append(g.attributes['cell'])
            sub_G.nodes[g.id]['type'] = all_type.index(g.attributes['cell'])

        pairs = list(combinations(list(sub_G.nodes()), 2)) # all possible node pairs
        for pair in pairs:
            type1, type2 = graph.nodes[pair[0]].attributes['cell'], graph.nodes[pair[1]].attributes['cell']
            if not type_rule2(type1, type2):
                continue
            w1, l1 = graph.nodes[pair[0]].attributes['w'], graph.nodes[pair[0]].attributes['l']
            w2, l2 = graph.nodes[pair[1]].attributes['w'], graph.nodes[pair[1]].attributes['l']
            if w1 != w2 or l1 != l2:
                continue
            node_pair.append([pair[0], pair[1]])

        num_nodes += len(graph.nodes)

        for p in graph.pins:
            sub_G.add_node(p.id+num_nodes)
            if p.attributes['type'] not in all_type:
                all_type.append(p.attributes['type'])
            sub_G.nodes[p.id+num_nodes]['type'] = all_type.index(p.attributes['type'])
            sub_G.add_edge(p.node_id, p.id+num_nodes)
        for n in graph.nets:
            edges = combinations(n.pins, 2)
            for edge in edges:
                sub_G.add_edge(edge[0]+num_nodes, edge[1]+num_nodes)

        all_pairs.append(node_pair)
        G.append(sub_G)

    return G, all_pairs, dataY

def subgraph_extract(g, v1, v2):
    thres_min = 4
    thres_max = 8
    rad = math.ceil(nx.shortest_path_length(g, source=v1, target=v2) / 2)
    rad = max(rad, thres_min)
    rad = min(rad, thres_max)

    nodes1 = list(nx.single_source_shortest_path_length(g, v1, cutoff=rad).keys())
    nodes2 = list(nx.single_source_shortest_path_length(g, v2, cutoff=rad).keys())

    g1 = g.subgraph(nodes1)
    g2 = g.subgraph(nodes2)
    g1 = nx.convert_node_labels_to_integers(g1)
    g2 = nx.convert_node_labels_to_integers(g2)
    return g1, g2

def prepare_json():
    train_dir = './dataset/train'
    test_dir = './dataset/test'
    filename = './dataset/data.pkl'
    is_train = [4]
    is_test = [0, 1, 2, 3, 5, 6, 7, 8, 9]
    neg_size = 3
    num_train = 0
    num_test = 0

    G, all_pairs, labels = readckts(filename)

    for i in range(len(G)):
        if i not in is_train and i not in is_test:
            continue
        g = G[i]
        node_pair = all_pairs[i]
        label = labels[i]
        num_pairs = 0
        
        random.seed(1234)
        random.shuffle(node_pair)
        for pair in node_pair:
            if [pair[0], pair[1]] in label or [pair[1], pair[0]] in label:
                ged = 0
            else:
                ged = 1
                if i in is_train:
                    num_pairs += 1
                    if num_pairs >= neg_size*len(label):
                        continue
            g1, g2 = subgraph_extract(g, pair[0], pair[1])
            graph1 = [[e[0], e[1]] for e in g1.edges()]
            graph2 = [[e[0], e[1]] for e in g2.edges()]
            labels1 = [g1.nodes[i]['type'] for i in g1.nodes()]
            labels2 = [g2.nodes[i]['type'] for i in g2.nodes()]
            # ged = nx.graph_edit_distance(g1, g2)
            gp_dict = {"graph_1": graph1, "graph_2": graph2, "labels_1": labels1, "labels_2": labels2, "ged": ged}
            if i in is_train:
                with open(os.path.join(train_dir, str(num_train)+'.json'), "w") as f:
                    json.dump(gp_dict, f)
                num_train += 1
            if i in is_test:
                with open(os.path.join(test_dir, str(num_test)+'.json'), "w") as f:
                    json.dump(gp_dict, f)
                num_test += 1

if __name__ == '__main__':
    prepare_json()
