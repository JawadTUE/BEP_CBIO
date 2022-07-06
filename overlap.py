import numpy as np 
inf = float('inf')

def n2_overlap(sublists):
    """
    Inefficient algorithm to calculate overlap between all combinations of lists in quadratic time.
    
    Parameters:
        sublists (list): list of k sublist of integers. Each sublist contains an even amount of integers.
                       The sublists are ordered ascending. 
                       Each sublist contains start and end points (alternately) of regions

    Returns:
        overlap_matrix (numpy matrix): k x k symmetrical matrix representing the amount of overlap between 
                                        each combination of sublists of the input
    """
    overlap_matrix = np.zeros((len(sublists),len(sublists)))
    
    #test every combination of active regions
    for i1,c1 in enumerate(sublists): #sublist 1
        for i2,c2 in enumerate(sublists): #sublist 2
            for g1 in range(0,len(c1),2): #region of sublist 1
                for g2 in range(0,len(c2),2): #region of sublist 2
                    no_overlap = (c2[g2] > c1[g1+1] or c2[g2+1] < c1[g1])
                    if not no_overlap:
                        overlap_matrix[i1,i2] = overlap_matrix[i1,i2] + 1

    return overlap_matrix

def sweepline_overlap(sublists):
    """
    Sweepline algorithm to calculate overlap between all combinations of lists in linear time.
    
    Parameters:
        sublists (list): list of k sublist of integers. Each sublist contains an even amount of integers.
                       The sublists are strictly ordered ascending. 
                       Each sublist contains start and end points (alternately) of regions

    Returns:
        overlap_matrix (numpy matrix): k x k symmetrical matrix representing the amount of overlap between 
                                        each combination of sublists of the input
    """
    n = sum([len(c) for c in sublists])//2 #total amount of regions in all sublists
    k = len(sublists) #amount of sublists

    overlap_matrix = np.zeros((k,k)) 
    #fill diagonal of matrix: overlap of a sublist with it self equals the amount of regions in the sublist
    for i in range(k): 
        overlap_matrix[i,i] = len(sublists[i])//2

    sweepline = set()
 
    cursors = [0 for c in sublists] #index of cursors on each of the k sublists: initially all on start of lists (index 0)
    cursor_values = [l[cursors[i]] for i,l in enumerate(sublists)] #actual values of the k cursors

    i = 0
    while i < 2*n: #each of the 2*n points (n regions) is visited once
        min_indices = minIndices(cursor_values) #find the sublists on which the cursors have the minimal point of all sublists
        min_cursors = np.array(cursors)[min_indices] #index of smallest cursors on their sublist: relevant for even/odd check
        sorted_min_indices = index_sorted(min_indices,min_cursors) #sort sublists with minimum based on position of their cursor (even/odd)
        
        #loop through each sublist with the minimum (sorted from even to odd positioned cursor)
        for min_index in sorted_min_indices:
            #add and remove from sweepline
            if isEven(cursors[min_index]): #even indexed points: start of region
                #update overlap of new sublist with all old sublists in sweepline and add to sweepline
                for c in sweepline:
                    overlap_matrix[min_index,c] += 1
                    overlap_matrix[c,min_index] += 1
                sweepline.add(min_index)
            else: #odd indexed points: end of region: remove from sweepline
                sweepline.remove(min_index)

            #update cursors
            if cursors[min_index] < (len(sublists[min_index]) - 1): #move cursor to next point of sublist
                cursors[min_index] += 1
                cursor_values[min_index] = sublists[min_index][cursors[min_index]]
            else: #if cursor is on the end of the sublist: this sublist cannot have minimum ever again
                cursor_values[min_index] = inf
            
            i += 1 #increase when a point is processed
    
    return overlap_matrix

def minIndices(l):
    """
    Get index/indices of the minimum of list l 

    Parameters:
        l (list): list of values
        
    Returns:
        list of indices of the minimum
    """
    min_value = min(l) #find all indices of this value
    return [index for index,value in enumerate(l) if value == min_value]

def isEven(n):
    """
    Check if n is even

    Parameters:
        n (int)
        
    Returns:
        boolean
    """
    return n%2 == 0

def index_sorted(l1,l2):
    """
    Sort list l1 based on the indexed parity sorting of l2: see even_odd_index_sorting(l) function
    
    Parameters:
        l1: list to be sorted based on indices of l2
        l2: list from which sorted indices are taken
    
    Returns:
        l1 (list): sorted list
    """
    return np.array(l1)[even_odd_index_sorting(l2)]

def even_odd_index_sorting(l):
    """
    Argsort list l based on parity (even numbers first) and return the indices in correct order
    
    Parameters:
        l (list): list to be sorted
        
    Returns:
        list of indices
    """
    return [index for index,value in sorted(enumerate(l), key=lambda x: x[1]%2)]