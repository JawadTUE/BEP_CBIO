from random import choice
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class HCS:
    """
    How to run:
        
    1. HCS_model = HCS(data_matrix, Graph())
    2. sim_threshold = HCS_model.getMinSimilarity(fraction_threshold = 0.2) #optional step
    3. clusters = HCS_model.cluster(sim_threshold = sim_threshold)
    """
    def __init__(self, data_matrix, graph):
        """ 
        Initiate HCS model
        
        Parameters:
            data_matrix (numpy matrix): n x n symmetric matrix with ones on diagonal. 
                                        Each entry represents the similarity between 2 nodes. 
            graph: (empty) Graph object

        Returns:
            None
        """
        self.data_matrix = data_matrix
        self.graph = graph #can be empty graph or graph with nodes and edges
    
    def getMinSimilarity(self, fraction_threshold = 0.5):
        """ 
        Calculate similarity threshold based on the given fraction of total edges that should remain
        
        Parameters:
            fraction_threshold (0 <= float <= 1): fraction of total possible amount of 
                                                    edges that should be in the graph 

        Returns:
            sim_threshold (0 <= float <= 1): similarity threshold for assigning edges between 2 nodes
        """
        sim_threshold_values = np.arange(0,1.05,0.05) #test 20 similarity threholds between 0 and 1
        fraction_values = [] #values of f(s): fraction of total edges
        n = self.data_matrix.shape[0]
        total = 0.5*n*(n-1) #maximum possible amount of edges in graph of n nodes
        
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

    def cluster(self, sim_threshold = 0.1):
        """ 
        Initiate the clustering algorithm
        
        Parameters:
            sim_threshold (0 <= float <= 1): similarity threshold for assigning edges between 2 nodes 

        Returns:
            clusters (list): list of Graph objects, each Graph represents a cluster and contains the nodes
        """
        self.sim_threshold = sim_threshold

        self.clusters = [] #list of Graph objects: each graph is a cluster

        self.buildGraph() #build similarity graph
        self.hcs_clustering(self.graph) #initiate recursive hcs clustering function

        return self.clusters

    def buildGraph(self):
        """ 
        Build similarity graph of n nodes. 
        Each row/column of the data matrix 
        An edge is placed between 2 nodes if their similarity in the data_matrix is >= similarity threshold.
        
        Parameters:
            None 

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
                if self.data_matrix[node1,node2] >= self.min_similarity:
                    self.graph.addEdge(node1,node2) 

    def hcs_clustering(self,graph):
        """ 
        Recursive hcs clustering algorithm
        
        Parameters:
            graph (Graph object) 

        Returns:
            None
        """
        if len(graph.nodes) == 1: #singleton graphs are added as clusters
            self.clusters.append(graph)
        else:
            graph1, graph2, minCut = self.kargerCut(graph) #split graph in 2
            if graph.isHighlyConnected(minCut): #highly connected subgraphs are added clusters
                self.clusters.append(graph)
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
        max_iterations = n**2 #perform Karger's algorithm n^2 times to get high probability of getting minimal cut

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




