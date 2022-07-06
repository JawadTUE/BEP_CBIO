from random import choice
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph import Graph
from distinctipy import distinctipy
import matplotlib.colors

class HCS:
    """
    How to run:
        
    1. HCS_model = HCS(data_matrix, Graph())
    2. sim_threshold = HCS_model.getSimilarityThreshold(fraction_threshold = 0.2) #optional step
    3. HCS_model.buildGraph(sim_threshold = sim_threshold) #optional step if input graph is given
    4. clusters = HCS_model.cluster()
    """
    def __init__(self, data_matrix = None, graph = None):
        """ 
        Initiate HCS model. 
        Either give an input graph to cluster or give a data matrix and make the graph within this class with self.buildGraph()
        
        Parameters: 
            data_matrix (numpy matrix): n x n symmetric matrix with ones on diagonal. 
                                        Each entry represents the similarity between 2 nodes. 
            graph: Graph object: default None so Graph is created in __init__

        Returns:
            None
        """
        self.data_matrix = data_matrix
        
        if graph:
            self.graph = graph #start with given graph
        else:
            self.graph = Graph() #start with empty graph
    
    def getSimilarityThreshold(self, fraction_threshold = 0.5):
        """ 
        Calculate similarity threshold based on the given fraction of total edges that should remain
        
        Parameters:
            fraction_threshold (0 <= float <= 1): fraction of total possible amount of 
                                                    edges that should be in the graph 

        Returns:
            sim_threshold (0 <= float <= 1): similarity threshold for assigning edges between 2 nodes
        """
        sim_threshold_values = np.arange(0,1.02,0.02) #test 20 similarity threholds between 0 and 1
        fraction_values = [] #values of f(s): fraction of total edges
        n = self.data_matrix.shape[0]
        total = n*(n-1)/2 #maximum possible amount of edges in graph of n nodes
        
        #calculate f(s) for each s
        for s in sim_threshold_values:
            count = 0 #amount of pairs of nodes with similarity >= s
            for node1 in range(1,n):
                for node2 in range(node1):
                    if self.data_matrix[node1,node2] >= s:
                        count += 1
            
            fraction_values.append(count/total)
        
        #calculate which similarity value (s) belongs to the given fraction threshold (f(s))
        sim_threshold = sim_threshold_values[np.argmin(np.abs([t - fraction_threshold for t in fraction_values]))]
        
        #plot f(s) as function of s, also add straight lines on the exact thresholds
        plt.figure()
        plt.plot(sim_threshold_values, fraction_values, marker = 'o')
        plt.hlines(fraction_threshold,0,1,linestyles='--',color = 'orange')
        plt.vlines(sim_threshold,0,1,linestyles='--',color = 'green')
        plt.grid()
        plt.xlabel('s')
        plt.ylabel('f(s)')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title('Fraction of edges (f(s)) in graph as function of similarity threshold (s)')
        plt.legend(['f(s)','f(s) threshold','s threshold'])
        plt.show()

        return sim_threshold

    def cluster(self, remove_low_degree_nodes = (True,3), singleton_adoption = True):
        """ 
        Initiate the clustering algorithm
        
        Parameters:
            remove_low_degree_nodes (tuple) (bool, int): whether or not to remove low degree nodes from the input graph and the degree threshold
            singleton_adoption (bool): whether or not to place singleton clusters in the most similar non-singleton cluster 

        Returns:
            clusters (list): list of Graph objects, each Graph represents a cluster and contains the nodes
        """
        self.clusters = [] #list of Graph objects: each graph is a cluster
        self.clustered_nodes = [] #list of nodes that are assigned to a cluster
        self.singletons = [] #store singleton clusters
        self.finished_points = 0 #amount of points that are either assigned a cluster or are singletons
        
        self.original_graph = deepcopy(self.graph)
        
        if remove_low_degree_nodes[0]: #remove low degree vertices
            self.remove_low_degree_nodes(remove_low_degree_nodes[1])

        self.size = len(self.graph.nodes) 

        # assert nx.is_connected(nx.Graph(self.graph.graph_dic)), \
        #     'Similarity graph is not connected, give graph with lower similarity threshold or delete more low degree nodes'
        
        print('Initializing clustering')
        # self.hcs_clustering(self.graph)

        nx_graph = nx.Graph(self.graph.graph_dic)
        connected_graphs = [set(nx_graph.subgraph(comp).copy().nodes) for comp in nx.connected_components(nx_graph)]

        print('Similarity graph contains {} components'.format(len(connected_graphs)))

        for component in connected_graphs:
            self.hcs_clustering(self.graph.subgraph(component)) #initiate recursive hcs clustering function

        if singleton_adoption: #add singletons to clusters
            print('Clustering Finished. Initializing singleton adoption.')
            self.singletons_adoption()
            print('Singleton adoption finished')

        print('HCS clustering finished: {} points out of {} clustered into {} clusters'.\
                format(len(self.clustered_nodes), len(self.original_graph.nodes),len(self.clusters)))

        return self.clusters

    def buildGraph(self, sim_threshold = 0.1):
        """ 
        Build similarity graph of n nodes. 
        Each row/column of the data matrix is a node
        An edge is placed between 2 nodes if their similarity in the data_matrix is >= similarity threshold.
        
        Parameters:
            sim_threshold (0 <= float <= 1): similarity threshold for assigning edges between 2 nodes 

        Returns:
            None
        """
        #add nodes: each row/column
        for node in range(self.data_matrix.shape[0]):
            self.graph.addNode(node)
        
        #add edges: check if similarity is >= threshold
        rows,columns = self.data_matrix.shape
        # only loop through bottomleft triangle of matrix since it is symmetrical
        for node1 in range(1,rows):
            for node2 in range(node1):
                if self.data_matrix[node1,node2] >= sim_threshold:
                    self.graph.addEdge(node1,node2) 
        
        print('Similarity graph with threshold {} build successfully: {} nodes and {} edges.'.format(\
                sim_threshold,len(self.graph.nodes), len(self.graph.edges)))

    def remove_low_degree_nodes(self, threshold):
        """
        Remove nodes with low degree from the similarity graph

        Parameters:
            threshold (int): nodes with degree lower than threshold are removed

        Returns:
            None
        """
        nodes = self.graph.nodes.copy() #all nodes of the graph
        count = 0 #amount of deleted nodes

        for node in nodes:
            if len(self.graph.graph_dic[node]) < threshold: #degree
                self.graph.deleteNode(node)
                count += 1
        
        print('{} low degree nodes removed.'.format(count))
        print('Graph: {} nodes and {} edges.'.format(len(self.graph.nodes), len(self.graph.edges)))

    def hcs_clustering(self,graph):
        """ 
        Recursive hcs clustering algorithm
        
        Parameters:
            graph (Graph object) 

        Returns:
            None
        """
        if len(graph.nodes) == 1: #singleton graphs are added as clusters
            self.singletons.append(min(graph.nodes))
            self.finished_points += 1
            print('Singleton cluster created. {}/{} points clustered'.format(self.finished_points, self.size))
        else:
            graph1, graph2, minCut = self.kargerCut(graph) #split graph in 2
            if graph.isHighlyConnected(minCut): #highly connected subgraphs are added clusters
                self.clusters.append(graph)
                self.clustered_nodes += graph.nodes
                self.finished_points += len(graph.nodes)
                print('Cluster of size {} created. {}/{} points clustered'.format(len(graph.nodes),self.finished_points, self.size))
            else: #recurse on the 2 non-highly connected subgraphs
                self.hcs_clustering(graph1)
                self.hcs_clustering(graph2)
        
    def kargerCut(self,graph):
        """ 
        Karger's minimal cut algorithm: randomly merge 2 nodes until 2 nodes remain. This represents the minimal cut
        
        Parameters:
            graph (Graph object) 

        Returns:
            subgraph1, subgraph2 (Graph objects): 2 subgraphs of the original graph
            best_mincut (int): minimal amount of edges to cut to disconnect the input graph into the subgraphs
        """
        g = graph.graph_dic #work with the graph dictionary

        n = len(g.keys())
        max_iterations = 2*n #perform Karger's algorithm n^2 times to get high probability of getting minimal cut

        best_mincut = float('inf')
        best_subgraphs = []

        for i in range(max_iterations):
            copy_g = deepcopy(g) #copy dictionary: this one gets editted
            supernodes = {key:[] for key in list(copy_g.keys())} # {node: [list of nodes with which node had merged]}

            while len(copy_g) > 2: #continue untill 2 nodes remain
                #randomly choose an edge (v,w)
                v = choice(list(copy_g.keys()))
                w = choice(copy_g[v])
                
                copy_g = self.mergeNodes(copy_g,v,w) #merge edge (v,w)
                
                #store node w in supernodes of v and delete w: new node is now vw
                supernodes[v].append(w)
                supernodes[v] += supernodes[w]
                supernodes.pop(w)

            current_mincut = len(list(copy_g.values())[0]) #amount of edges to cut to disconnect in this iteration
            
            #store smallest minCut and the corresponding subgraphs
            if current_mincut < best_mincut:
                best_mincut = current_mincut
                #nodes of the subgraphs: nodes of the 2 supernodes at the end
                best_subgraphs = [{key}.union(set(values)) for key,values in list(supernodes.items())]
        
        #make 2 new Graph objects based on the best subgraphs
        subgraph1, subgraph2 = graph.cut(best_subgraphs[0],best_subgraphs[1])

        return subgraph1, subgraph2, best_mincut

    def mergeNodes(self,graph_dic,v,w):
        """ 
        Merge edge (v,w) in the given graph dictionary
        
        Parameters:
            graph_dic (dict): dictionary of Graph object. keys are nodes and their values are the adjacent nodes
            v,w (int): adjacent nodes to merge. Node w merged into v and will be deleted as a node
            
        Returns:
            graph_dic (dict): dictionary of Graph object in which the edge (v,w) is merged
        """
        #make edges between adjacent nodes of w and node v
        for adj_node in graph_dic[w]:
            if adj_node != v:
                graph_dic[v].append(adj_node)
                graph_dic[adj_node].append(v)
        
        #node w will be deleted in the whole dictionary
            graph_dic[adj_node].remove(w)

        del graph_dic[w]

        return graph_dic

    def singletons_adoption(self):
        """
        Nodes that are in a cluster of size 1 are placed in one of the real clusters. 
        Singleton nodes belong to clusters with which it has most neighbours in the original graph

        Parameters:
            None

        Returns:
            None
        """
        for singleton in self.singletons: #loop through singletons
            #keep track of cluster with most neighbours with singleton node in original graph
            maxNeighbours = 0
            bestCluster = None

            singleton_neighbours = self.graph.graph_dic[singleton] #all neighbours of the singleton node in original graph

            #check each cluster
            for cluster in self.clusters:
                cluster_nodes = cluster.nodes #nodes of this cluster

                #amount of overlap between neigbours of singleton and nodes in cluster
                numberNeighbours = len(set(singleton_neighbours) & set(cluster_nodes)) 

                #update best cluster
                if numberNeighbours > maxNeighbours:
                    maxNeighbours = numberNeighbours
                    bestCluster = cluster
            
            #add singleton node to best cluster if there is any
            if bestCluster:
                bestCluster.addNode(singleton)
                self.clustered_nodes.append(singleton)
                self.finished_points += 1

    def drawGraphs(self):
        """
        Plot the original graph and color the clusters

        Parameters:
            None

        Returns:
            None
        """
        G = nx.Graph(self.original_graph.graph_dic)

        colors = ['#000000'] + [matplotlib.colors.rgb2hex(color) for color in distinctipy.get_colors(len(self.clusters)+1, exclude_colors=[(1,1,1),(0,0,0)])]
        plt.figure(figsize=(9,7))
        nodesize = 50

        pos = nx.spring_layout(G, k=1, iterations=1000)

        for index,cluster in enumerate(self.clusters):
            nx.draw_networkx_nodes(G, pos=pos, node_size = nodesize,nodelist=list(cluster.nodes), node_color=colors[index+1], label='Cluster {}'.format(index+1))

        non_clustered_nodes = [node for node in self.original_graph.nodes if node not in self.clustered_nodes]

        nx.draw_networkx_nodes(G, pos=pos, node_size = nodesize, nodelist=non_clustered_nodes, node_color=colors[0], label='No Cluster')
        nx.draw_networkx_edges(G, pos=pos)

        plt.legend(scatterpoints = 1)
        plt.show()


