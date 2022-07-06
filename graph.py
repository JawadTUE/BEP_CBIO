class Graph:
    """
    Build undirected graph without self-loops
    Nodes are represented as integers in the keys of the graph dictionary
    Adjacent nodes are stored in the list of the values of the dictionary
    """
    def __init__(self, graph_dic = None):
        """ 
        Initiate Graph class
        
        Parameters:
            graph_dic (dict): each key is a node (int)
                              each value is list of adjacent nodes (list of ints)

        Returns:
            None
        """
        if graph_dic:
            self.graph_dic = graph_dic
        else:
            self.graph_dic = {}

        self.nodes = set(self.graph_dic.keys()) #set of nodes 

        self.edges = [] #list of edges (edge = binary set)
        for u in self.nodes:
            for v in self.graph_dic[u]:
                if {u,v} not in self.edges:
                    self.edges.append({u,v})

    def addNode(self,node):
        """
        Add node to graph
        
        Parameters:
            node (int): node to be added

        Returns:
            None
        """
        if node not in self.graph_dic:
            self.graph_dic[node] = []
            self.nodes.add(node)

    def deleteNode(self,node):
        """
        Delete given node from graph dictionary, nodes set, edge list

        Parameters:
            node (int): node to delete

        Returns:
            None
        """
        self.nodes.remove(node) #delete from nodes

        #find all edges that contain node
        for adj_node in self.graph_dic[node]:
            self.graph_dic[adj_node].remove(node)
            self.edges.remove({node,adj_node})
        
        del self.graph_dic[node]

    def addEdge(self,node1,node2):
        """
        Add edge to graph
        
        Parameters:
            node1, node2 (int): edge (node1,node2) is added

        Returns:
            None
        """
        edge = {node1,node2}
        
        #both nodes should exist in graph and edge should not be in graph yet
        assert node1 in self.nodes, "Node {} not in graph".format(node1)
        assert node2 in self.nodes, "Node {} not in graph".format(node2)
        assert edge not in self.edges, "Edge {} already exists".format(edge)

        self.edges.append(edge)
        self.graph_dic[node1].append(node2)
        self.graph_dic[node2].append(node1)
    
    def isHighlyConnected(self, minCut):
        """
        Graph with n nodes is highly connected if the mincut > n/2 or n == 1
        
        Parameters:
            minCut (int): minimal amount of edges to remove from graph to disconnect it

        Returns:
            boolean: whether this Graph is highly connected or not
        """
        n = len(self.nodes)

        return minCut > n//2

    def subgraph(self, nodes):
        """
        Create subgraph of current graph based on given set of nodes such that the subgraph contains only the given nodes and the edges belonging to those nodes

        Parameters:
            nodes (set of ints): nodes in the subgraph 

        Returns:
            1 Graph object
        """
        subgraph_dict = {}

        for node in self.graph_dic:
            if node in nodes: #subgraph 1 
                subgraph_dict[node] = [adj_node for adj_node in self.graph_dic[node] if adj_node in nodes]
        
        return Graph(subgraph_dict)
    
    def cut(self, nodes1, nodes2):
        """
        Split current graph into 2 subgraphs based on the 2 sets of nodes
        return 2 subgraphs based on given nodes sets
        
        Parameters:
            nodes1, nodes2 (set): nodes (int) in each subgraph (nodes1 ∩ nodes2 = None)

        Returns:
            2 Graph objects
        """
        #graph dictionaries of the 2 subgrahps
        s1 = {}
        s2 = {}
        
        #go through nodes of this graph and check in which subgraph it belongs
        for node in self.graph_dic:
            if node in nodes1: #subgraph 1 
                s1[node] = [adj_node for adj_node in self.graph_dic[node] if adj_node in nodes1]
            else: #subgraph 2
                s2[node] = [adj_node for adj_node in self.graph_dic[node] if adj_node in nodes2]
        
        return Graph(s1), Graph(s2)