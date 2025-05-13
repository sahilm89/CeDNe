"""
Plotting and visualization utilities for CeDNe

Includes layered, shell, spiral plots, and anatomical position views (2D/3D),
as well as drawing functions for gap junctions and neuropeptides.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.patches import Circle
from matplotlib.text import Annotation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.axes_grid1 import make_axes_locatable

import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import textalloc as ta

import os
from cedne import Neuron 
from cedne import simulator

from .config import *
from .graphtools import is_left_neuron
import cmasher as cmr

def simpleaxis(axes, every=False, outward=False):
    """
    Sets the axis style for the given axes.

    Parameters:
        axes (list or ndarray): The axes to set the style for.
        every (bool, optional): Set the style for every axis. Defaults to False.
        outward (bool, optional): Set the style outward. Defaults to False.

    Returns:
        None
    """
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if (outward):
            ax.spines['bottom'].set_position(('outward', 10))
            ax.spines['left'].set_position(('outward', 10))
        if every:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_title('')


def groupPosition(pos, type_cat):

    x,y = zip(*pos.values())
    x = np.array(x)
    yax = sorted(set(y))[::-1]

    new_pos = {}
    for i,yi in enumerate(yax):
        new_pos[i] = [(xi,yi) for xi in sorted(x[np.where(y==yi)])]
    # print([(i,len(v)) for (i,v) in new_pos.items()])
    # print(type_cat)
    pos2 = {}
    boxes = {}
    order = ['sensory', 'interneuron', 'motorneuron']
    box_height = 0.025
    for i,o in enumerate(order):
        boxes[o] = {}
        j=0
        if o in type_cat:
            sortedKeys = sorted(type_cat[o].keys())
            if o == "interneuron":
                intOrder = ['layer 1 interneuron', 'layer 2 interneuron', 'layer 3 interneuron', 'category 4 interneuron', 'linker to pharynx', 'pharynx', 'endorgan']
                sortedKeys = [intn for intn in intOrder if intn in sortedKeys]
                # p = [sortedKeys[0]]
                # print(p, sortedKeys)
                # sortedKeys= sortedKeys[1:4] + p + [sortedKeys[4]]
                
            for cat in sortedKeys: 
                cat_pos = []
                # print(cat, len(new_pos[i]), len(type_cat[o][cat]))
                for n in sorted(type_cat[o][cat]):
                #print(n, cat, new_pos[i], i, j )
                    pos2[n] = new_pos[i][j]
                    cat_pos.append(pos2[n])
                    j+=1
                boxes[o][cat] = [(cat_pos[0][0]-0.035, cat_pos[0][1]-(box_height/2)), (cat_pos[-1][0] - cat_pos[0][0]) + 0.07, box_height]

    return pos2, boxes

def plot_spiral(neunet, save=False, figsize=(8,8), font_size=11):
    """
    Generates a spiral layout for the network.

    Parameters:
    - neunet (NeuralNetwork): The neural network object.
    - save (bool or str): If True or a string, saves the plot to a file.
    - figsize (tuple): Figure size in inches.
    - font_size (int): Font size for node labels.

    Returns:
    - pos (dict): A dictionary mapping node names to their positions in the graph.
    """
    node_size = 800
    node_color = ['lightgray' for node in neunet.nodes]
    edge_color = []
    edge_weight = []
    for (u,v,attrib_dict) in list(neunet.edges.data()):
        if 'color' in attrib_dict:
            edge_color.append(attrib_dict['color'])
        else:
            edge_color.append('gray') 
        
        if 'weight' in attrib_dict:
            edge_weight.append(attrib_dict['weight'])
        else:
            edge_weight.append(1)

    pos = nx.spiral_layout(neunet, equidistant=True, resolution=0.5)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(neunet, node_size=node_size, ax=ax, pos=pos, labels={node: node.name for node in neunet.nodes}, with_labels=True, node_color=node_color, edge_color=edge_color, font_size=font_size)
    if save:
        plt.savefig(save)
    plt.close()
    
    return pos

def add_gapjn_symbols(ax, pos, edges, line_length=0.05, alpha=1, color='k'):
    for edge in edges:
        start, end = edge[0], edge[1]
        x1, y1 = pos[start]
        x2, y2 = pos[end]
         
        # Calculate the midpoint of the edge
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

        # Calculate direction vector and normalize it
        direction = np.array([x2 - x1, y2 - y1])
        line_length = 0.02*np.linalg.norm(direction)
        direction = direction / np.linalg.norm(direction)

    
        # Calculate the perpendicular direction
        perp_direction = np.array([-direction[1], direction[0]])

        # Calculate the positions of the capacitor lines
        line_start1 = np.array([mid_x, mid_y]) - (line_length / 2) * perp_direction
        line_end1 = np.array([mid_x, mid_y]) + (line_length / 2) * perp_direction

        line_start2 = np.array([mid_x, mid_y]) - (line_length / 2) * perp_direction + direction * 0.02
        line_end2 = np.array([mid_x, mid_y]) + (line_length / 2) * perp_direction + direction * 0.02

        # Draw the capacitor lines
        ax.plot([line_start1[0], line_end1[0]], [line_start1[1], line_end1[1]], color=color, alpha=alpha, lw=2)
        ax.plot([line_start2[0], line_end2[0]], [line_start2[1], line_end2[1]], color=color, alpha=alpha, lw=2)

def add_neuropep_symbols(ax, pos, edges, node_size, color='k', crescent_radius=0.04, offset_factor=1.0):
    # Get node positions
    node_radius = np.sqrt(node_size) / 220
    for edge in edges:
        source, target = edge[0], edge[1]
        source_pos = np.array(pos[source])
        target_pos = np.array(pos[target])
        
        # Calculate direction vector
        direction = target_pos - source_pos
        norm = np.linalg.norm(direction)
        direction /= norm  # Normalize
        
        # Calculate the position for the overlapping circles
        outer_center = target_pos - (crescent_radius* 2 + node_radius)  * direction
        inner_center = outer_center - offset_factor * crescent_radius * direction
        
        # Draw the line
        #ax.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], color=color, zorder=1)
        
        # Draw the inner circle (white) to create crescent effect
        inner_circle = Circle(inner_center, crescent_radius, color=color, zorder=2)
        ax.add_patch(inner_circle)
        
        # Draw the outer circle (colored) to create crescent effect
        outer_circle = Circle(outer_center, crescent_radius, color='white', zorder=3)
        ax.add_patch(outer_circle)

def add_circular_arrowheads(ax, pos, edges, node_size, scale_factor=0.001, color='k'):
    for edge in edges:
        start, end = edge[0], edge[1]
        x1, y1 = pos[start]
        x2, y2 = pos[end]
        
        # # Calculate direction vector and normalize it
        # direction = np.array([x2 - x1, y2 - y1])
        # direction = direction / np.linalg.norm(direction)

        # # Calculate the radius of the circle
        # radius = np.sqrt(node_size) * scale_factor

        # # Calculate the position of the circle's center (slightly outside the endpoint)
        # circle_center = np.array([x2, y2]) - (radius + np.sqrt(node_size) * 0.05) * direction

        # # Draw the circle at the end of the arrow
        # circle = Circle(circle_center, radius, color=color, zorder=2)
        # ax.add_patch(circle)

        direction = np.array([x2 - x1, y2 - y1])
        direction = direction / np.linalg.norm(direction)

        # Calculate the radius of the circle
        radius = np.sqrt(node_size) * scale_factor

        # Adjust position to place the circle just outside the node
        node_radius = np.sqrt(node_size) / 2 * 0.01
        circle_center = np.array([x2, y2]) - (node_radius + radius) * direction

        # Draw the circle at the end of the arrow
        circle = Circle(circle_center, radius, color=color, zorder=2)
        ax.add_patch(circle)
            
def plot_shell(neunet, center=None, shells=None, save=False, figsize=(8,8), edge_color_dict=None, edge_alpha_dict=None,\
                node_color_dict=None, edge_labels=False, handles=None, fontsize=11, width_logbase=2, title=None):
    """
    Generates a shell layout for the network.

    Parameters:
    - neunet (NeuralNetwork): The neural network object.

    Returns:
    - pos (dict): A dictionary mapping node names to their positions in the graph.
    """
    MAX_LENGTH_FOR_STRAIGHT_TEXT = 20
    arc_rad = 0.1
    node_size = 800
    arrow_size=20
    rotation = 0 #45
        
    if shells is None:
        if isinstance(center, str):
            assert center in neunet.neurons
            shells = [[neunet.neurons[center]], [neu for nname,neu in neunet.neurons.items() if nname !=center]]
        elif isinstance(center, list):
            #print(center, neunet.neurons.keys(), [c in neunet.neurons.keys() for c in center])
            assert all([c in neunet.neurons.keys() or isinstance(c, Neuron) for c in center])
            new_center = [c if c in neunet.neurons.keys() else c.name for c in center]
            shells = [[neunet.neurons[c] for c in new_center], [neu for nname,neu in neunet.neurons.items() if nname not in new_center]]
        elif isinstance(center, Neuron):
            shells = [[neunet.neurons[center.name]], [neu for nname,neu in neunet.neurons.items() if nname != center.name]]
        else:
            raise ValueError("center must be a neuron name or a list of neuron names")
    if node_color_dict is None:
        node_color = ['lightgray' if not hasattr(node, 'color') else node.color for node in neunet.nodes]
    else:
        node_color = [node_color_dict[node] for node in neunet.nodes]

    if edge_color_dict is None:
        edge_color_dict = {edge:'gray' if not hasattr(edge, 'color') else edge.color for edge in neunet.edges}
    if edge_alpha_dict is None:
        edge_alpha_dict = {edge:1 if not hasattr(edge, 'alpha') else edge.alpha for edge in neunet.edges}

    if edge_labels:
        edge_labels = {edge: edge.weight for edge in neunet.edges}
    else:
        edge_labels = {edge: '' for edge in neunet.edges}
    # edge_color = []
    # edge_weight = []
    # for (u,v,attrib_dict) in list(neunet.edges.data()):
    #     edge_color.append(attrib_dict['color'])
    #     edge_weight.append(attrib_dict['weight'])

    pos = nx.shell_layout(neunet, nlist=shells, scale=2)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.set_aspect('equal', 'datalim', anchor='C')
    # print(len(pos))
    # print([(n, pos[neu]) for (n,neu) in neunet.neurons.items()])
    # nx.draw(neunet, node_size = 800, ax=ax, pos=pos, with_labels=False, node_color = node_color, edge_color=edge_color)
    nx.draw_networkx_nodes(neunet, pos=pos, node_size = node_size, ax=ax, node_color = node_color)
    
    if len(neunet.connections.keys()):
        for edge,connection in neunet.connections.items():
            edge_color = edge_color_dict[edge]
            edge_alpha = edge_alpha_dict[edge]
            if connection.connection_type == 'chemical-synapse':
                if (edge[1], edge[0]) in neunet.edges():
                    rad = arc_rad
                else:
                    rad = 0.0
                width = 1 + np.log2(connection.weight)/np.log2(width_logbase)
                nx.draw_networkx_edges(neunet, pos=pos, edgelist=[edge], node_size=node_size, connectionstyle=f'arc3,rad={rad}', arrows=True, arrowstyle='-|>', arrowsize=arrow_size, edge_color=edge_color, alpha=edge_alpha, width=width, ax=ax)
                # add_circular_arrowheads(ax, pos, edges=[edge], node_size=node_size, color='k')
                # add_half_circle_arrowheads(ax, pos, edges=[edge], node_size=node_size, color='k')
            elif connection.connection_type == 'gap-junction':
                width = 1 + np.log2(connection.weight)/np.log2(width_logbase)
                nx.draw_networkx_edges(neunet, pos=pos, edgelist=[edge], node_size=node_size, connectionstyle='arc3,rad=0.0', arrowstyle='-', arrowsize=arrow_size, edge_color='k', alpha=edge_alpha, width=width, ax=ax)
                add_gapjn_symbols(ax, pos, edges=[edge], alpha=edge_alpha, color='k')
            else:
                width = 1 + np.log2(connection.weight)/np.log2(width_logbase)
                nx.draw_networkx_edges(neunet, pos=pos, edgelist=[edge], node_size=node_size, connectionstyle='arc3,rad=0.0', arrowstyle='-', arrowsize=arrow_size, edge_color='k', alpha=edge_alpha, width=width, ax=ax)
                add_neuropep_symbols(ax, pos, edges=[edge], node_size=node_size)
                #add_neuropep_symbols(ax, pos, edges=[edge], alpha=edge_alpha, color='k')
            if edge_labels:
                nx.draw_networkx_edge_labels(neunet, pos=pos, edge_labels=edge_labels, ax=ax)
    else:
        nx.draw_networkx_edges(neunet, pos=pos, node_size=node_size, arrows=True, arrowstyle='-|>', arrowsize=20, ax=ax)

    for node, (x, y) in pos.items():
        label = str(node.name)
        if len(label)>4 and len(pos)>MAX_LENGTH_FOR_STRAIGHT_TEXT:
            if (x>0 and y>0) or (x<0 and y<0):
                plt.text(x, y, label, fontsize=fontsize, rotation=rotation, ha='center', va='center')
            elif (x>0 and y<0) or (x<0 and y>0):
                plt.text(x, y, label, fontsize=fontsize, rotation=-rotation, ha='center', va='center')
            else:
                plt.text(x, y, label, fontsize=fontsize, ha='center', va='center')
        else:
            plt.text(x, y, label, fontsize=fontsize, ha='center', va='center')
    
    plt.axis('off')
    if handles:
        fig.legend(handles=handles,loc='outside center right')
    if save:
        if isinstance(save, str):
            plt.savefig(save)
        elif isinstance(save, bool):
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            plt.savefig(OUTPUT_DIR + '_'.join([n.name for n in shells[0]]) + '.svg')
    #plt.show()
    if title:
        fig.suptitle(title)
    return fig
    
    
def plot_layered(interesting_conns, neunet, nodeColors=None, edgeColors = None, save=False, title='', extraNodes=[], extraEdges=[], pos=[], mark_anatomical=False, colorbar=False):
    """
    Generates a graph visualization of a given neural network. Change this function to make more streamlined.

    Parameters:
    - interesting_conns (list): A list of tuples representing the connections between nodes in the neural network.
    - edge_colors (list): A list of values representing the color of each edge in the graph.
    - node_colors (dict): A dictionary mapping node names to their corresponding colors.
    - title (str): The title of the graph.
    - neunet (NeuralNetwork): The neural network object.
    - save (bool, optional): Whether to save the graph as an image file. Defaults to False.
    - extra_nodes (list, optional): A list of additional nodes to include in the graph. Defaults to [].
    - extra_edges (list, optional): A list of additional edges to include in the graph. Defaults to [].
    - pos (dict, optional): A dictionary mapping node names to their positions in the graph. Defaults to [].
    - mark_anatomical (bool, optional): Whether to mark anatomically significant connections. Defaults to False.

    Returns:
    - pos (dict): A dictionary mapping node names to their positions in the graph.
    """
    classcolor = 'green'

    G = nx.Graph()
    G.add_edges_from(interesting_conns)
    if edgeColors is None:
        edge_color=[]
    elif isinstance(edgeColors, str):
        edge_color = [edgeColors for e in interesting_conns]
    elif len(edgeColors) == len(interesting_conns):
        if all (isinstance(ec, str) for ec in edgeColors):
            edge_color = [ec for ec in edgeColors]
        elif all (isinstance(ec, float) for ec in edgeColors):
            #cmap2 = sns.diverging_palette(220, 20, as_cmap=True)
            cmap = plt.get_cmap('PuOr')
            max_color = max(np.abs(edgeColors))
            norm = matplotlib.colors.Normalize(vmin=-max_color,vmax=max_color)
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            edge_color = [m.to_rgba(col) for col in edgeColors]
    else:
        raise ValueError('edge_colors must either be a string or a list of strings or floats \
            of the same length as interesting_conns')
    
    if nodeColors is None:
        node_color=[]
    elif isinstance(nodeColors, list):
        if all (isinstance(nc, str) for nc in nodeColors):
            node_color = [nc for nc in nodeColors]
    elif isinstance(nodeColors, dict):
        cmap2 = plt.get_cmap('RdYlGn')
        norm2 = matplotlib.colors.Normalize(vmin=-1.5,vmax=1.5)
        o = cm.ScalarMappable(norm=norm2, cmap=cmap2)

    if mark_anatomical:
        for i,e in enumerate(G.edges):
            u,v= e[0], e[1]
            if neunet.has_edge(neunet.neurons[u], neunet.neurons[v]) or neunet.has_edge(neunet.neurons[v], neunet.neurons[u]):
                if (neunet.neurons[u], neunet.neurons[v], 'gapjn') in neunet.edges or (neunet.neurons[v], neunet.neurons[u], 'gapjn') in neunet.edges:
                    edge_color[i] = 'k'
                else:
                    edge_color[i] = 'blue'

    edge_labels = {}
    neuronTypes = ['motorneuron', 'interneuron', 'sensory']
    node_color = {'sensory': [], 'interneuron':[], 'motorneuron':[]}
    node_alpha = {'sensory': [], 'interneuron':[], 'motorneuron':[]}

    edgeList_1, edgeList_2 = [], []
    colors_1, colors_2 = [], []
    node_shapes = {'sensory': 'o', 'interneuron':'s', 'motorneuron':'H'}
    node_types = {'sensory': [], 'interneuron':[], 'motorneuron':[]}
    #node_shapes = {'sensory': [], 'interneuron':[], 'motorneuron':[]}
    
    edges_within = []
    edges_across = []
    color_within = []
    color_across = []
    for e,c in zip(G.edges, edge_color):
        if neunet.neurons[e[0]].type == neunet.neurons[e[1]].type:
            edges_within.append(e)
            color_within.append(c)
        else:
            edges_across.append(e)
            color_across.append(c) 
            
    if len(extraEdges):
        G.add_edges_from(extraEdges)
    
    oriNodes = tuple(G.nodes)
    for n in extraNodes:
        if not n in G.nodes:
            G.add_node(n)

    # if not len(extraNodes):
    #     nodelist = G.nodes
    # else:
    #     nodelist = extraNodes
    categories = {} 
    for n in G.nodes:
        # print(n)
        if n in oriNodes:
            if n in nodeColors.keys():
                #print("Y", n, nodeColors.keys())
                if nodeColors is not None:
                    if isinstance(nodeColors[n], str):
                        node_color[neunet.neurons[n].type].append(nodeColors[n])
                    elif isinstance(nodeColors[n], float):
                        node_color[neunet.neurons[n].type].append(o.to_rgba(nodeColors[n]))
                    else:
                        raise ValueError('node_colors must either be a string or a list of strings or floats \
                            of the same length as nodelist')
                else:
                    node_color[neunet.neurons[n].type].append(nodeColors[n])
                    node_alpha[neunet.neurons[n].type].append(0.8)
            else:
                #print(n, nodeColors.keys())
                node_color[neunet.neurons[n].type].append('lightgray') 
                node_alpha[neunet.neurons[n].type].append(0.8)
            node_types[neunet.neurons[n].type].append(n)
        # else:
        #     node_color[nn.neurons[n].type].append("None")
        #     node_alpha[nn.neurons[n].type].append(None)
        categories[n] = (neunet.neurons[n].type, neunet.neurons[n].category)
        nx.set_node_attributes(G, {n:{"layer":neuronTypes.index(neunet.neurons[n].type)}})
        # print(n, neuronTypes.index(neunet.neurons[n].type))
        #G.nodes[n].layer = nn.neurons[n].type 
    print(G.nodes())
    cats = set(categories.values())
    type_cat = {}
    for n,cat in categories.items():
        cat_alt = cat[1]
        if not cat[0] in type_cat:
            type_cat[cat[0]] = {cat_alt:[n]}
        else:
            if not cat_alt in type_cat[cat[0]]:
                type_cat[cat[0]][cat_alt] = [n]
            else:
                type_cat[cat[0]][cat_alt].append(n) 

    array_op = lambda x, sx: np.array([x[0]*sx, x[1]])
    sx=2
    #pos =  nx.spring_layout(G, k=1, iterations=70)# 
    
    if not len(pos):
        pos =  nx.multipartite_layout(G,subset_key="layer", align='horizontal')
        # print(pos)
        pos = {p:array_op(pos[p],sx) for p in pos}
        print(pos)
        pos, boxes = groupPosition(pos, type_cat)

    #print(edges_within, edges_across)
    fig, ax = plt.subplots(figsize=(40,4))
    #edge_color = 'k'
    #nx.draw(G, node_size = 1000, ax=ax, pos=pos, with_labels=True, edge_color=edge_color, node_color = node_color)#, width=weights, font_size=12)
    #plotNodes = []
    node_size=  2000
    for type in node_types:
        nx.draw_networkx_nodes(node_types[type], node_shape=node_shapes[type], node_size = node_size, ax=ax, pos=pos, node_color = node_color[type],  linewidths=1, alpha=0.8)#node_alpha[type]))#, width=weights, font_size=12)
    for node, (x, y) in pos.items():
        label = str(node)
        if len(label)>3:
            plt.text(x, y, label, fontsize=20, rotation=45, ha='center', va='center')
        else:
            plt.text(x, y, label, fontsize=20, ha='center', va='center')

    # nx.draw_networkx_labels(G, pos=pos, labels={o:o for o in oriNodes if len(o)<4}, ax=ax, font_size=20)
    # nx.draw_networkx_labels(G, pos=pos, labels={o:o for o in oriNodes if len(o)>3}, ax=ax, font_size=20, rotation=45)
    #nx.draw_networkx_edges(G, edgelist=interesting_conns, ax=ax, pos=pos, edge_color=edge_color, node_size = node_size, arrows=True, connectionstyle='arc3, rad=0.3', width=2)#colors_2, style='--')
    nx.draw_networkx_edges(G, edgelist=edges_within, ax=ax, pos=pos, edge_color=color_within, node_size = node_size, width=2, arrows=True, connectionstyle='arc3, rad=0.3')#colors_2, style='--')
    nx.draw_networkx_edges(G, edgelist=edges_across, ax=ax, pos=pos, edge_color=color_across, node_size = node_size, width=2, arrows=True)#, connectionstyle='arc3, rad=0.3')#colors_2, style='--')

    within_extra = []
    across_extra = []
    for e in extraEdges:
        if neunet.neurons[e[0]].type == neunet.neurons[e[1]].type:
            within_extra.append(e)
        else:
            across_extra.append(e)
    
    nx.draw_networkx_edges(G, edgelist = within_extra, ax=ax, edge_color='gray', pos=pos, alpha=0.2, width=1, arrows=True, connectionstyle='arc3, rad=0.3')
    nx.draw_networkx_edges(G, edgelist = across_extra, ax=ax, edge_color='gray', pos=pos, alpha=0.2, width=1)
    #nx.draw_networkx_edge_labels(G, pos, label_pos=0.5, edge_labels=edge_labels, ax=ax)
    if colorbar:
        divider = make_axes_locatable(ax)
        
        # cax = inset_axes(ax, loc='upper right', width="100%", height="100%",
        #                bbox_to_anchor=(0.75,0.85,.05,.05), bbox_transform=ax.transAxes) #
        # cax.set_xticks([-1,0,1])
        # cax.set_title("$\\Delta$ correlation", fontsize=24)
        # #cax = divider.append_axes("right", size="1%", pad=0.05)    
        # cbar = plt.colorbar(m, cax= cax, orientation='horizontal')
        # cbar.ax.tick_params(labelsize=24) 

        cax2 = divider.append_axes("bottom", size="2%", pad=0.1)
        cbar2= plt.colorbar(o, cax=cax2, orientation='horizontal')
        cbar2.ax.tick_params(labelsize='xx-large') 

    for types in boxes:
        for cat, pars in boxes[types].items():
            xy, width, height = pars[0], pars[1], pars[2]
            rect = matplotlib.patches.Rectangle(xy,width,height, linewidth=1, edgecolor='gray', facecolor='none', alpha=0.75, linestyle='dashed')
            if cat in ['linker to pharynx']:
                cat = "link. to phar."
                cat = cat.replace(' ', '\n')
            elif cat in ['sex-specific neuron']:
                cat = 'sex-spec'
                #cat = cat.replace(' ', '\n')]
            for suffix in ["interneuron", "motor neuron"]:
                if suffix in cat:
                    cat = cat.replace(suffix,'')
            ax.text(xy[0], xy[1]-0.002, cat, #-height
            horizontalalignment='left',
            verticalalignment='top', fontsize=28, color=classcolor)
            # Add the patch to the Axes
            ax.add_patch(rect)
    simpleaxis(ax, every=True)
    ax.set_title(title, y=1.01, fontsize=40)
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.colorbar(m)
    plt.tight_layout()
    plt.subplots_adjust(top=2)
    if not save==False:
        plt.savefig(save)
    
    plt.show()
    plt.close()
    
    return pos


def plot_position(nn, axis='AP-DV', highlight=None, booleanDictionary=None, title='', label='all', save=False, figsize=(20,3), limit=None):
    """
    A function to plot the positions of neurons based on their coordinates and attributes.

    Parameters:
        nn: NeuronNetwork object representing the neurons to be plotted.
        axis: String, the orientation of the plot (default: 'AP-DV').
        special: List of special neurons to highlight.
        booleanDictionary: Dictionary mapping boolean values to colors.
        title: String, the title of the plot.
        save: Boolean, whether to save the plot as an image.

    Returns:
        None
    """

    if booleanDictionary is None:
        active_color, passive_color = "#CC5500", "lightgrey"
        booleanDictionary = {True: active_color, False: passive_color}

    coords = ['AP', 'DV', 'LR']
    nlabels = np.array([n for n in nn.neurons])
    pos = [[] for i in range(len(nlabels))]
    if highlight is None:
        highlight = []
    elif isinstance(highlight, list):
        if isinstance(highlight[0], str):
            assert all([n in nn.neurons for n in highlight]), "Neuron(s) not found in the network."
        elif isinstance(highlight[0], list):
            assert all([n in nn.neurons for j in range(len(highlight)) for n in highlight[j]]), "Neuron(s) not found in the network."
        else:
            raise ValueError("Highlight must be a list of strings or a list of lists.")
            
    for i,n in enumerate(nlabels):
        if n in ['CANL']:
            pos[i] = [510, 20, -1]
        elif n in ['CANR']:
            pos[i] = [490, 20, 1]
        else:
            for x in coords:
                if x in nn.neurons[n].position: 
                    pos[i].append(nn.neurons[n].position[x])
                else:
                    raise ValueError(f"Neuron {n} has no position in axis {x}.")
    pos = np.array(pos)

    if limit:
        # limit should be a dict like {'AP': (0, 150)} or {'DV': (-50, 50), 'AP': (0, 200)}
        mask = np.ones(len(pos), dtype=bool)
        for coord, (low, high) in limit.items():
            axis_idx = coords.index(coord)
            mask &= (pos[:, axis_idx] >= low) & (pos[:, axis_idx] <= high)
        # Filter positions and labels
        pos = pos[mask]
        nlabels = nlabels[mask]

    posDict = dict(zip(coords, pos.T))
    posDict['LR'] = -posDict['LR'] # flipping LR for plotting in the correct direction
    posDict['DV'] = -posDict['DV'] # flipping DV for plotting in the correct direction

    xax, yax = axis.split('-')
    if xax in coords:
        x = posDict[xax]
    elif xax[::-1] in coords:
        x = -posDict[xax[::-1]]
    else:
        raise ValueError(f"Invalid axis: {xax}, use some combination of {coords}")

    if yax in coords:
        y = posDict[yax]
    elif yax[::-1] in coords:
        y = -posDict[yax[::-1]]
    else:
        raise ValueError(f"Invalid axis: {yax}, use some combination of {coords}")
    
    x = np.array(x)
    y = np.array(y)

    if isinstance(highlight[0], str):
        f,ax = plt.subplots(figsize=figsize, dpi=300, layout='constrained')
        plt.axis('off')
        facecolors = np.array([booleanDictionary[n in highlight] for n in nlabels])
        alphas = np.array([1 if n in highlight else 0.25 for n in nlabels])
        boolList = np.array([n in highlight for n in nlabels])
        #print([(n,facecolors[j]) for j,n in enumerate(nlabels) if n in highlight])
        ax.scatter(x[~boolList], y[~boolList], s=200, facecolor=facecolors[~boolList], edgecolor=facecolors[~boolList], alpha=alphas[~boolList], zorder=1)
        ax.scatter(x[boolList], y[boolList], s=200, facecolor=facecolors[boolList], edgecolor=facecolors[boolList], alpha=alphas[boolList], zorder=2)
        if label:
            ta.allocate_text(f,ax,x[boolList], y[boolList],
                        nlabels[boolList],
                        x_scatter=x[boolList], y_scatter=y[boolList],
                        textsize='xx-large', linecolor='gray', avoid_label_lines_overlap=True)

    elif isinstance(highlight[0], list):
        buffer = 1.1
        f,ax = plt.subplots( dpi=300, layout='constrained',figsize=figsize) #figsize=(2,0.3), dpi=300, layout='constrained'
        plt.axis('off') 
        plt.xlim(min(x)*buffer,max(x)*buffer)
        plt.ylim(min(y)*buffer,max(y)*buffer)
        # ax.set_aspect('equal')
        trans=ax.transData.transform
        trans2=f.transFigure.inverted().transform

        facecolors = cmr.take_cmap_colors('cmr.tropical', len(highlight), cmap_range=(0.15, 0.85), return_fmt='hex')
        color_dict = {n:[] for n in nlabels}
        alpha_dict = {n:0.25 for n in nlabels}
        boolList = np.array([False]*len(nlabels))
        for i,n in enumerate(nlabels):
            for j,hlc in enumerate(highlight):
                if n in hlc:
                    color_dict[n].append(facecolors[j])
                    alpha_dict[n] = 1
                    if label == 'all':
                        boolList[i] = True
                    elif label == 'left' and is_left_neuron(n):
                        boolList[i] = True
            if len(color_dict[n]) == 0:
                color_dict[n].append("lightgrey")
        piesize=0.3
        if not label == 'none':
            # if isinstance(label, int):
            #     randnum = np.random.default_rng()
            #     boolList = randnum.choice(boolList, size=label, replace=False)
            ta.allocate_text(f,ax,x[boolList], y[boolList],
                        nlabels[boolList],
                        x_scatter=x[boolList], y_scatter=y[boolList],
                        textsize='xx-large', linecolor='gray', avoid_label_lines_overlap=True)
        # print(boolList, ~boolList, x[boolList])
        # ax.scatter(x[~boolList], y[~boolList], s=200)
        # ax.scatter(x[boolList], y[boolList], s=200)
        for i,n in enumerate(nlabels):
            xx,yy=trans((x[i],y[i])) # figure coordinates
            xa,ya=trans2((xx,yy)) # axes coordinates
            #plot_pie(n, (x[i], y[i]), a, color_dict, alpha_dict, piesize)
            a = plt.axes([xa-piesize/2,ya-piesize/2, piesize, piesize])
            a.set_aspect('equal')
            plot_pie(n, (0,0), a, color_dict, alpha_dict, piesize=piesize)
        #     ax.pie([1/len(color_dict[n])]*len(color_dict[n]), center = (x[i], y[i]), colors=color_dict[n], radius = piesize, counterclock=False, wedgeprops={'alpha': alpha_dict[n], 'zorder': len(color_dict[n])})


    #Add the legend to the plot
    legend_ax = f.add_axes([0.95, 0.1, 0.1, 0.1])
    # legend_ax.set_xlim(0, 1)
    # legend_ax.set_ylim(0, 1)
    legend_ax.set_xlabel(xax, fontsize="x-large")
    legend_ax.set_ylabel(yax, fontsize="x-large")
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_aspect('equal')
    simpleaxis(legend_ax)
    ax.set_title(title, fontsize="xx-large")
    # f.draw_without_rendering()
    #ax.set_xlim((-0.9,-0.))
    if save:
        plt.savefig(save)
    plt.show()

def plot_pie(n, center, ax, color_dict, alpha_dict, pie_division = None, piesize=1): 
    # radius for pieplot size on a scatterplot
    if not pie_division:
        pie_division = [1/len(color_dict[n])]*len(color_dict[n])
    ax.pie(pie_division, center = center, colors=color_dict[n], radius = piesize, counterclock=False, wedgeprops={'alpha': alpha_dict[n], 'zorder': len(color_dict[n])})

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
    return tag

def plot_position_3D(nn, highlight=None, booleanDictionary=None, title='', label=True, save=False):
    """
    A function to plot the positions of neurons based on their coordinates and attributes.

    Parameters:
        nn: NeuronNetwork object representing the neurons to be plotted.
        axis: String, the orientation of the plot (default: 'AP-DV').
        special: List of special neurons to highlight.
        booleanDictionary: Dictionary mapping boolean values to colors.
        title: String, the title of the plot.
        save: Boolean, whether to save the plot as an image.

    Returns:
        None
    """
    coords = ['AP', 'DV', 'LR']
    if booleanDictionary is None:
        active_color, passive_color = "#CC5500", "lightgrey"
        booleanDictionary = {True: active_color, False: passive_color}
    nlabels = np.array([n for n in nn.neurons])
    pos = [[] for i in range(len(nlabels))]
    if highlight is None:
        highlight = []
    else:
        assert all([n in nn.neurons for n in highlight]), "Neuron(s) not found in the network."
            
    for i,n in enumerate(nlabels):
        if n in ['CANL']:
            pos[i] = [510, 20, -1]
        elif n in ['CANR']:
            pos[i] = [490, 20, 1]
        else:
            for x in coords:
                if x in nn.neurons[n].position: 
                    pos[i].append(nn.neurons[n].position[x])
                else:
                    raise ValueError(f"Neuron {n} has no position in axis {x}.")
    pos = np.array(pos)
    posDict = dict(zip(coords, pos.T))
    posDict['LR'] = -posDict['LR'] # flipping LR for plotting in the correct direction
    posDict['DV'] = -posDict['DV'] # flipping DV for plotting in the correct direction
    facecolors = np.array([booleanDictionary[n in highlight] for n in nlabels])
    alphas = np.array([1 if n in highlight else 0.25 for n in nlabels])
    boolList = np.array([n in highlight for n in nlabels])

    xax, yax, zax = coords
    x = np.array(posDict[xax])
    y = np.array(posDict[yax])
    z = np.array(posDict[zax])
    
    xscale =np.max(x)-np.min(x)
    yscale =np.max(y)-np.min(y)

    # Create the figure and GridSpec
    f = plt.figure(figsize=(xscale,yscale), layout='constrained')
    gs = GridSpec(10, 10, figure=f)

    # Create the 3D subplots using GridSpec
    ax = f.add_subplot(gs[:8,:], projection='3d')
    ax.set_box_aspect(aspect = (10,2,1))
    plt.axis('off') 
    ax_leg = f.add_subplot(gs[8,8], projection='3d')
    ax_leg.grid(visible=False, which='both', axis='both')
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    ax_leg.set_zticks([])

    ax_leg.set_aspect('equal')
    sc_bg = ax.scatter(x[~boolList], y[~boolList], z[~boolList], s=25, facecolor=facecolors[~boolList], edgecolor=facecolors[~boolList], alpha=alphas[~boolList], zorder=1)
    sc_fg = ax.scatter(x[boolList], y[boolList], z[boolList], s=25, facecolor=facecolors[boolList], edgecolor=facecolors[boolList], alpha=alphas[boolList], zorder=2)

    arrowprops = dict(
    arrowstyle="-",
    #connectionstyle="angle,angleA=0,angleB=90,rad=10"
    )
    annot_bg = {}
    for i,n in enumerate(nlabels):
        lr_sign = 1 if z[i]>0 else -1
        if n in highlight:
            annotate3D(ax, s=n, xyz=(x[i], y[i], z[i]), fontsize=10, xytext=(np.random.uniform(-30,30),lr_sign*30),
               textcoords='offset points', ha='right',va='bottom',arrowprops=arrowprops) 
        else:
            annot_bg[i] = annotate3D(ax, s=n, xyz=(x[i], y[i], z[i]), fontsize=10, xytext=(np.random.uniform(-30,30),lr_sign*30),
               textcoords='offset points', ha='right',va='bottom',arrowprops=arrowprops)
            annot_bg[i].set_visible(False)

    ax_leg.set_xlabel(coords[0])
    ax_leg.set_ylabel(coords[1])
    ax_leg.set_zlabel(coords[2])

    ax.set_xlim3d(np.min(x),np.max(x))
    ax.set_ylim3d(np.min(y), np.max(y))
    ax.set_zlim3d(np.min(z), np.max(z))
    ax.set_title(title, fontsize="xx-large")

    ax.shareview(ax_leg)

    # def update_annot(ind):
    #     #pos_xy = sc_bg.get_offsets()[ind["ind"][0]]
    #     #print(pos_xy)
    #     xs, ys, zs = sc_bg._offsets3d
    #     vxs, vys, vzs = proj_transform(xs, ys, zs, ax.get_proj())
    #     sorted_z_indices = np.argsort(vzs)[::-1]
    #     for j in sorted_z_indices[ind["ind"]]:
    #         annot_bg[j].set_visible(True)
    #     #annot_bg.set_text(nlabels[sorted_z_indices[ind['ind'][0]]])

    #     #annot_bg.xy = pos_xy
    #     #print(ind["ind"])
    #     #text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
    #                         #" ".join([nlabels[n] for n in ind["ind"]]))
    #     #text = ", ".join([nlabels[n] for n in ind["ind"]]) #nlabels[n]

    #     #annot_bg.set_text(text)
    #     #annot_bg.get_bbox_patch().set_facecolor('lightgrey')
    #     #annot_bg.get_bbox_patch().set_alpha(0.4)
    

    # def hover(event):
    #     #vis = annot_bg.get_visible()
    #     if event.inaxes == ax:
    #         cont, ind = sc_bg.contains(event)
    #         if cont:
    #             update_annot(ind)
    #             # for n in ind["ind"]:
    #             #     annot_bg[n].set_visible(True)
    #             f.canvas.draw_idle()
    #         else:
    #             #if vis:
    #             for n in ind["ind"]:
    #                 annot_bg[n].set_visible(False)
    #             f.canvas.draw_idle()

    # f.canvas.mpl_connect("motion_notify_event", hover)

    if save:
        plt.savefig(save)
    plt.show()

## For simulation module
def plot_simulation_results(results, twinx=True):
    rate_model, inputs, rates = results
    f, ax = plt.subplots(figsize=(2.5, 2.5), layout='constrained')
    for node in rates:
        ax.plot(rate_model.time_points, rates[node], label=node.name, lw=2)
    # ax.set_ylim((-15,15))

    if twinx:
        ax2 = ax.twinx()
    else:
        ax2=ax
    for inp in inputs:
        ax2.plot(rate_model.time_points, [inp.process_input(t) for t in rate_model.time_points], ls='--', color='k')
    ax2.axhline(y=0, ls='--', alpha=0.25, color='gray')
    # ax2.set_ylim((-4,4))
    for inp in inputs:
        if isinstance(inp, simulator.StepInput):
            ax2.axvline(x=inp.tstart, ls='--', color='gray', alpha=0.25)
            ax2.axvline(x=inp.tend, ls='--', color='gray', alpha=0.25)
    # ax.set_ylim((-1,1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate')
    # ax.set_xticks([0,30,60,90])
    simpleaxis(ax)
    # simpleaxis(ax2)
    f.legend(loc='outside upper center', ncol=len(rates), frameon=False)
    return f

def compare_simulation_results(results1, results2, twinx=True):
    rate_model, inputs, rates1 = results1
    _, _, rates2 = results2
    f, ax = plt.subplots(figsize=(2.5, 2.5), layout='constrained')
    colors = {node: plt.cm.viridis(i/len(rates1)) for i, node in enumerate(rates1)}
    for node in rates1:
        ax.plot(rate_model.time_points, rates1[node], label=node.label or node, lw=2, color=colors[node])
        ax.plot(rate_model.time_points, rates2[node], label=node.label or node, lw=2, color=colors[node], ls='--')
    if twinx:
        ax2 = ax.twinx()
    else:
        ax2=ax
    for inp in inputs:
        ax2.plot(rate_model.time_points, [inp.process_input(t) for t in rate_model.time_points], ls='--', color='k')
    ax2.axhline(y=0, ls='--', alpha=0.25, color='gray')
    # ax2.set_ylim((-4,4))
    for inp in inputs:
        if isinstance(inp, simulator.StepInput):
            ax2.axvline(x=inp.tstart, ls='--', color='gray', alpha=0.25)
            ax2.axvline(x=inp.tend, ls='--', color='gray', alpha=0.25)
    # ax.set_ylim((-1,1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate')
    # ax.set_xticks([0,30,60,90])
    simpleaxis(ax)
    f.legend(loc='outside upper center', ncol=len(rates1), frameon=False)
    return f