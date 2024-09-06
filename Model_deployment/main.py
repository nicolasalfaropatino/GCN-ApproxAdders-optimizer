import networkx as nx
import numpy as np
import torch
import torch_geometric
import re
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from karateclub import Node2Vec
from torch_geometric.data import Dataset, Data

def netlist_preprocesing(netlist, filename):
    filename = filename[:-2]
    output_folder = "outputs/graphs_parsed_outputs/raw"
    output_csv = os.path.join(output_folder, filename + '.csv')
    df = pd.DataFrame(columns=['source_node', 'target_node', 'source_node_type', 'target_node_type'])
    df2 = pd.DataFrame(columns=['source_node', 'target_node', 'source_node_type', 'target_node_type','source_node_init','target_node_init'])
    G = nx.DiGraph()
    node_id_map = {}
    next_id = 0

    netlist_lines = netlist.readlines()
    i = 0
    elements = []
    elements_for_recreating_netlist = []
    while i < len(netlist_lines):
        line = netlist_lines[i].strip()
        node_input = []
        node_output = []
        node_type = None
        node_name = None

        if "GND" in line:
            x = i
            while x < len(netlist_lines)-1 and not "IBUF \\" in netlist_lines[x+1] and not "OBUF \\" in netlist_lines[x+1] and not re.search(r"LUT\d", netlist_lines[x+1]):
                if re.search(r'\.G\(', netlist_lines[x+1].strip()):
                    node_output.append('\\<const0> ')
                    node_input.append('GND')
                    node_type = 'GND'
                    node_name = 'GND' + str(i)
                x +=1
                element = {'node_type': node_type,
                        'node_name': node_name,
                        'node_input': node_input,
                        'node_output': node_output}

                element2 = {'node_name' : node_name,
                            'node_specific_name' : node_name,
                            'node_type' : node_type,
                            'node_init' : None}
                elements.append(element)
                elements_for_recreating_netlist.append(element2)


        elif "VCC" in line:
            x = i
            while x < len(netlist_lines)-1 and not "IBUF \\" in netlist_lines[x+1] and not "OBUF \\" in netlist_lines[x+1] and not re.search(r"LUT\d", netlist_lines[x+1]):
                if re.search(r'\.P\(', netlist_lines[x+1].strip()):
                    node_output.append('\\<const1> ')
                    node_input.append('VCC')
                    node_type = 'VCC'
                    node_name = 'VCC' + str(i)
                x +=1
            element = {'node_type': node_type,
                    'node_name': node_name,
                    'node_input': node_input,
                    'node_output': node_output,
                    'node_init' : None}

            element2 = {'node_name' : node_name,
                    'node_specific_name' : node_name,
                    'node_type' : node_type,
                    'node_init' : None
                    }
            elements.append(element)
            elements_for_recreating_netlist.append(element2)

        elif "IBUF \\" in line or "OBUF \\" in line:
            parts = line.split(" ")
            node_type = parts[0]
            node_name = parts[1].lstrip().strip("\\")
            x = i
            while x < len(netlist_lines)-1 and not "IBUF \\" in netlist_lines[x+1] and not "OBUF \\" in netlist_lines[x+1] and not re.search(r"LUT\d", netlist_lines[x+1]):
                if re.search(r'\.I\(', netlist_lines[x+1]):
                    node_input.append(re.search(r'\.I\(([^)]+)\)', netlist_lines[x+1].strip()).group(1) if re.search(r'\.I\(([^)]+)\)', netlist_lines[x+1].strip()) else None)
                elif re.search(r'\.O\(', netlist_lines[x+1]):
                    node_output.append(re.search(r'\((.*?)\)', netlist_lines[x+1].strip()).group(1) if re.search(r'\((.*?)\)', netlist_lines[x+1].strip()) else None)
                x += 1
            element = {'node_type': node_type ,
                    'node_name': node_name,
                    'node_input': node_input,
                    'node_output': node_output,
                    'node_init' : None
                    }
            
            element2 = {'node_name' : node_name,
                    'node_specific_name' : node_name,
                    'node_type' : node_type,
                    'node_init' : None
                    }
            elements.append(element)
            elements_for_recreating_netlist.append(element2)

        elif re.search(r"LUT\d", line):
            node_type = 'LUT'
            node_name_lut = line.strip()[:-3]
            node_name = netlist_lines[i+2].strip().strip('\\')
            node_init = netlist_lines[i+1]
            x = i
            while x < len(netlist_lines)-1 and not "IBUF \\" in netlist_lines[x+1] and not "OBUF \\" in netlist_lines[x+1] and not re.search(r"LUT\d", netlist_lines[x+1]):
                if re.search(r'\.I\d\(([^)]+)\)', netlist_lines[x+1].strip()):
                    node_input.append(re.search(r'\.I\d\(([^)]+)\)', netlist_lines[x+1].strip()).group(1))
                elif re.search(r'\.O\(', netlist_lines[x+1]):
                    node_output.append(re.search(r'\((.*?)\)', netlist_lines[x+1].strip()).group(1) if re.search(r'\((.*?)\)', netlist_lines[x+1].strip()) else None)
                x +=1
            element = {'node_type': node_type,
                    'node_name': node_name_lut,
                    'node_input': node_input,
                    'node_output': node_output,
                    'node_init' : node_init
                    }
            element2 = {'node_name' : node_name,
                    'node_specific_name' : node_name_lut,
                    'node_type' : node_type,
                    'node_init' : node_init
                    }
            
            elements.append(element)
            elements_for_recreating_netlist.append(element2)
            
        i +=1
    
    for element in elements:
      if element['node_name'] not in node_id_map:
        node_id_map[element['node_name']] = next_id
        next_id += 1

    for element in elements:
      G.add_node(element['node_name'], node_type=element['node_type'])
      for output in element['node_output']:
        for target_element in elements:
          if output in target_element['node_input']:
            G.add_edge(element['node_name'], target_element['node_name'])
            df = df.append({'source_node': node_id_map[element['node_name']],
                            'target_node': node_id_map[target_element['node_name']],
                            'source_node_type': element['node_type'],
                            'target_node_type': target_element['node_type']}, ignore_index = True)

            df2 = df2.append({'source_node': node_id_map[element['node_name']],
                            'target_node': node_id_map[target_element['node_name']],
                            'source_node_name': element['node_name'],
                            'target_node_name': target_element['node_name'],
                            'source_node_type': element['node_type'],
                            'target_node_type': target_element['node_type'], 
                            'source_node_init': element['node_init'],
                            'target_node_init': target_element['node_init']}, ignore_index = True)
    
    df_for_recreating_netlist = pd.DataFrame.from_dict(elements_for_recreating_netlist)
    df.to_csv(output_csv, index=False)
    print(f"Archivo CSV generado: {output_csv}")
    color_map = {
      'IBUF': 'green',
      'OBUF': 'red',
      'LUT': 'blue',
      'VCC': 'yellow',
      'GND': 'black'}

    node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]
    #nx.draw(G, with_labels = True,node_size=300, node_color=node_colors)

    return df2

class GraphDataset(Dataset):
    def __init__(self,root, test=False, transform=None, pre_transform=None):
        self.test = test
        super(GraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        #Si estos archivos se encuentran en raw_dir, se omite su procesamiento
        processed_files = []
        for raw_file in self.raw_file_names:
            processed_file_name = os.path.basename(raw_file).replace('.csv', '.pt')
            if self.test:
                processed_file_name = f'data_test_{processed_file_name}'
            else:
                processed_file_name = f'data_{processed_file_name}'
            processed_files.append(processed_file_name)
            return processed_files

    def download(self):
        pass

    def process(self):
        for raw_path in self.raw_paths:
            data = pd.read_csv(raw_path,
                                header=0,
                                names=['source_node', 'target_node', 'source_node_type', 'target_node_type'])
            graph_obj = self._create_graph_object(data)
            torch.save(graph_obj, os.path.join(self.processed_dir, os.path.basename(raw_path).replace('.csv', '.pt')))

    def _create_graph_object(self,data):
        G = nx.DiGraph()
        seen_target_nodes = []
        seen_nodes = []
        node_mapping = {}
        next_node_id = 0
        for _, row in data.iterrows():
            source_node, target_node, source_node_type, target_node_type = row[0], row[1], row[2], row[3]
            
            if source_node not in node_mapping:
                node_mapping[source_node] = next_node_id
                G.add_node(next_node_id, node_type=source_node_type)
                next_node_id += 1

            if target_node not in node_mapping:
                node_mapping[target_node] = next_node_id
                G.add_node(next_node_id, node_type=target_node_type)
                next_node_id += 1

        for _, row in data.iterrows():
            source_node, target_node = row['source_node'], row['target_node']
            source_index = node_mapping[source_node]
            target_index = node_mapping[target_node]
            G.add_edge(source_index, target_index)

        label = self._get_labels(G)
        node_feats = self._get_node_features(G)
        edge_index = self._get_adjacency_info(G)
        #label = self._get_labels(G)
        graph_obj = Data(x=node_feats, edge_index=edge_index, y=label)

        color_map = {
        'IBUF': 'green',
        'OBUF': 'red',
        'LUT': 'blue',
        'VCC': 'yellow',
        'GND': 'black'}
        node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]
        plt.figure(figsize=(12, 8))
        nx.draw_networkx(G, with_labels=True, node_size=300, node_color=node_colors)
        plt.show()


        nx.draw(G, with_labels=True)
        plt.show()
        return graph_obj

    def _get_embeddings(self, G):
        node_list = sorted(G.nodes())
        emb = Node2Vec(walk_length=10, walk_number=10,window_size=20, dimensions=20, p=1.0, q=0.5)
        emb.fit(G)
        embeddings = emb.get_embedding()
        ordered_embeddings = embeddings[[node_list.index(i) for i in range(len(G.nodes()))]]
        return embeddings

    def _get_node_features(self, G):
        degrees = np.array([val for (node, val) in G.degree()])
        closeness = np.array([nx.closeness_centrality(G, u=node) for node in G.nodes()])
        embeddings = self._get_embeddings(G)
        node_feats = np.column_stack((degrees,closeness, embeddings))
        #node_feats = np.column_stack((degrees))
        return torch.tensor(node_feats, dtype=torch.float)

    def _get_adjacency_info(self, G):
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        edge_index = torch.tensor([(node_mapping[source], node_mapping[target]) for source, target in G.edges()],dtype=torch.long).t().contiguous()
        return edge_index

    def _get_labels(self, G):
        label_dict = {'IBUF': 0, 'OBUF': 1, 'LUT': 2, 'GND': 3, 'VCC': 4}
        labels = []
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get('node_type')
            label = label_dict.get(node_type, -1)
            labels.append(label)
        return torch.tensor(labels, dtype=torch.long)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        file_name = self.processed_file_names[idx]
        if file_name.startswith("data_"):
            file_name = file_name[5:]
        data_path = os.path.join(self.processed_dir, file_name)
        data = torch.load(data_path)
        return data

class Net(torch.nn.Module):
    #definimos la estructura de la GCN
    def __init__(self, in_channels, hidden_channels1, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, out_channels)
        #self.conv3 = GCNConv(hidden_channels2, out_channels)
        #self.conv4 = GCNConv(hidden_channels3, out_channels)
        # self.readout = GlobalPooling()

    #realizamos el paso de encode, es decir hacemos el message passing
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
        #return self.conv3(x, edge_index)
        # return self.conv4(x, edge_index)

        #Despues de tener la matriz de vectores del message passing, se realiza el producto punto
        #Entre los nodos que estemos viendo
    def decode(self, z, edge_label_index):
        return(z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

        #Lo hacemos para todos los distintos edges
    def decode_all(self, z):
        prob_adj = z @ z.t()    #matrix multiplication in numpy
        print(prob_adj)
        return (prob_adj > 0.9).nonzero(as_tuple=False).t()
    
    def forward(self, x, edge_index, edge_label_index=None):
        z = self.encode(x, edge_index)

        if edge_label_index is not None:
            return self.decode(z, edge_label_index)
        else:
            return self.decode_all(z)

def getGraph(filename):

    my_model = torch.load('models/model_all_netlist.pth')
    my_model.eval()

    #cargar netlist subido
    graph_path = 'static/graphs/' + filename
    #preprocessing del netlist (parsearlo y generar un grafo)
    netlist = open(graph_path, 'r')
    netlist_dataframe = netlist_preprocesing(netlist, filename)

    node_features_dataframe = pd.DataFrame(columns=['node', 'node_name', 'node_type', 'node_init'])
    
    netlist_dataframe_seen_nodes = []
    for _, row in netlist_dataframe.iterrows():
        node = row['source_node']
        node_type = row['source_node_type']
        node_init = row['source_node_init']
        node_name = row['source_node_name']
        if node not in netlist_dataframe_seen_nodes:
            node_features_dataframe = node_features_dataframe.append({'node': node,
                                            'node_name' : node_name,
                                            'node_type' : node_type,
                                            'node_init' : node_init}, ignore_index = True)
            netlist_dataframe_seen_nodes.append(node)

        node = row['target_node']
        node_type = row['target_node_type']
        node_init = row['target_node_init']
        node_name = row['target_node_name']
        if node not in netlist_dataframe_seen_nodes:
            node_features_dataframe = node_features_dataframe.append({'node': node,
                                        'node_name' : node_name,
                                        'node_type' : node_type,
                                        'node_init' : node_init}, ignore_index = True)
            netlist_dataframe_seen_nodes.append(node)
        
    raw_csv_path = 'outputs/graphs_parsed_outputs/'

    user_graph = GraphDataset(raw_csv_path)
    user_graph = user_graph.get(0)

    x = user_graph.x
    edge_index = user_graph.edge_index
    edge_label_index = user_graph.edge_label_index if hasattr(user_graph, 'edge_label_index') else None

    z = my_model.encode(user_graph.x, user_graph.edge_index)
    final_edge_index = my_model.decode_all(z)

    graph_opto = my_model(x, edge_index, edge_label_index)

    modified_graph_dataframe = pd.DataFrame(columns=['source_node', 
                                                     'target_node', 
                                                     'source_node_name', 
                                                     'target_node_name', 
                                                     'source_node_type', 
                                                     'target_node_type', 
                                                     'source_node_init',
                                                     'target_node_init'])
    
    for i in range(final_edge_index.shape[1]):
        source = final_edge_index[0, i].item()
        target = final_edge_index[1, i].item()
        if source in node_features_dataframe['node'].values:
            source_node_name = node_features_dataframe.loc[node_features_dataframe['node'] == source, 'node_name'].values[0]
            source_node_type = node_features_dataframe.loc[node_features_dataframe['node'] == source, 'node_type'].values[0]
            source_node_init = node_features_dataframe.loc[node_features_dataframe['node'] == source, 'node_init'].values[0]

        if target in node_features_dataframe['node'].values:
            target_node_name = node_features_dataframe.loc[node_features_dataframe['node'] == target, 'node_name'].values[0]
            target_node_type = node_features_dataframe.loc[node_features_dataframe['node'] == target, 'node_type'].values[0]
            target_node_init = node_features_dataframe.loc[node_features_dataframe['node'] == target, 'node_init'].values[0]


        modified_graph_dataframe = modified_graph_dataframe.append({'source_node' : source, 
                                                                    'target_node' : target, 
                                                                    'source_node_name' : source_node_name,
                                                                    'target_node_name' : target_node_name,
                                                                    'source_node_type' : source_node_type,
                                                                    'target_node_type' : target_node_type,
                                                                    'source_node_init' : source_node_init,
                                                                    'target_node_init' : target_node_init}, ignore_index = True)

#modified graph dataframe cleaning
#removing rows with self looping edges and IBUF in taget node type
    modified_graph_dataframe = modified_graph_dataframe[modified_graph_dataframe.target_node_type != 'IBUF']
    modified_graph_dataframe = modified_graph_dataframe[modified_graph_dataframe.source_node_type != 'OBUF']
    modified_graph_dataframe = modified_graph_dataframe[modified_graph_dataframe.source_node != modified_graph_dataframe.target_node]
    


#plotear el grafo modificado
    G = nx.DiGraph()
    

    for idx, row in modified_graph_dataframe.iterrows():
        G.add_edge(row['source_node'],row['target_node'])
        G.nodes[row['source_node']]['node_type'] = row['source_node_type']
        G.nodes[row['source_node']]['node_name'] = row['source_node_name']
        G.nodes[row['source_node']]['node_init'] = row['source_node_init']
        G.nodes[row['target_node']]['node_type'] = row['target_node_type']
        G.nodes[row['target_node']]['node_name'] = row['target_node_name']
        G.nodes[row['target_node']]['node_init'] = row['target_node_init']
    
    color_map = {
      'IBUF': 'green',
      'OBUF': 'red',
      'LUT': 'blue',
      'VCC': 'yellow',
      'GND': 'black'}

    node_colors = [color_map[G.nodes[node]['node_type']] for node in G.nodes()]
    nx.draw(G, with_labels = True,node_size=300, node_color=node_colors)
    

    # num_nodes = final_edge_index.max().item() + 1   
    # G.add_nodes_from(range(num_nodes))

    # for i in range(final_edge_index.shape[1]):
    #     source = final_edge_index[0, i].item()
    #     target = final_edge_index[1, i].item()
    #     G.add_edge(source, target)

    # nx.draw(G, with_labels=True)
    # plt.show()

    # #transform tensor output given by the model to a graph and then to netlist representation
    # return graph_opto

test_graph = getGraph('8bitAdderExact.v')


