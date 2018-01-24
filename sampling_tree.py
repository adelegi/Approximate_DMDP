import numpy as np
import math



class Node:
    """
    Node of sampling tree. If Node is a leaf val is an event, else val is a tuple
    of proba of (respectively) left and right events. Number of children are registered
    so as to build well-balanced (min depth, from left to right) binary trees.
    
    """
    
    def __init__(self, depth, max_depth):
                 
        self.L = None
        self.R = None
        self.val = [0, 0]
        self.depth = depth
        self.num_children = 0
        self.max_children = 2**(max_depth-depth)
        self.is_full = (depth == max_depth)
    
    def one_more_child_left(self, proba):
        
        self.num_children += 1
        self.is_full = (self.num_children == self.max_children)
        self.val[0] += proba
    
    def one_more_child_right(self, proba):
        
        self.num_children += 1
        self.is_full = (self.num_children == self.max_children)
        self.val[1] += proba


class Sampling_tree:
    """
    Class designed to sample efficiently from events with discrete probabilities using
    a binary tree. Adapted from: "An efficient method for weighted samplingwithout replacement"
    by Wong and Easton.

    Context: drawings from a fixed number of events with given probabilities
    
    """
                 
    def __init__(self, probabilities):
        
        self.proba = probabilities
        self.max_depth = math.ceil(np.log(len(probabilities))/np.log(2))
        
        self.root = Node(0, self.max_depth)
        self.preprocess_tree()

    def add(self, val, node):
        
        if node.L == None and node.depth < self.max_depth:
            
            node.L = Node(node.depth + 1, self.max_depth)
            node.one_more_child_left(self.proba[val])
            self.add(val, node.L)
        
        elif node.L != None and node.L.is_full == False:
            
            node.one_more_child_left(self.proba[val])
            self.add(val, node.L)
        
        elif node.R == None and node.depth < self.max_depth:
            
            node.R = Node(node.depth + 1, self.max_depth)
            node.one_more_child_right(self.proba[val])
            self.add(val, node.R)
        
        elif node.R != None and node.R.is_full == False:
            
            node.one_more_child_right(self.proba[val])
            self.add(val, node.R)
            
        elif node.depth == self.max_depth:
            
            node.val = val
            
        else:
            
            print("one more mysterious case")
        
    def preprocess_tree(self):
        
         for state in range(len(self.proba)):
                
                self.add(state, self.root)
                 
    def sample(self):
        """
        random event drawing according to input probabilities
        
        """
        
        x = np.random.random()
        bound = 0
        node = self.root
        
        while node.depth < self.max_depth:
            
            if x < node.val[0] + bound:
                
                node = node.L
            
            else:
                
                bound += node.val[0]
                node = node.R
        
        return node.val
        
    def delete_tree(self):
        
        self.root = None



class Sampling_tree_with_policy_updates:
    """
    Class designed to sample efficiently from events with discrete probabilities using
    a binary tree. Adapted from: "An efficient method for weighted samplingwithout replacement"
    by Wong and Easton.
    
    this version allows to update the probability of the last visited event with a new probability
    
    """
                 
    def __init__(self, weights):
        
        self.weights = weights
        self.sum = sum(weights)
        self.max_depth = math.ceil(np.log(len(weights))/np.log(2))
        
        self.root = Node(0, self.max_depth)
        self.preprocess_tree()
        
        ### keep track of last path down the three for updates
        self.last_path = [0]*self.max_depth
        self.last_visit = None

    def add(self, val, node):
        
        if node.L == None and node.depth < self.max_depth:
            
            node.L = Node(node.depth + 1, self.max_depth)
            node.one_more_child_left(self.weights[val])
            self.add(val, node.L)
        
        elif node.L != None and node.L.is_full == False:
            
            node.one_more_child_left(self.weights[val])
            self.add(val, node.L)
        
        elif node.R == None and node.depth < self.max_depth:
            
            node.R = Node(node.depth + 1, self.max_depth)
            node.one_more_child_right(self.weights[val])
            self.add(val, node.R)
        
        elif node.R != None and node.R.is_full == False:
            
            node.one_more_child_right(self.weights[val])
            self.add(val, node.R)
            
        elif node.depth == self.max_depth:
            
            node.val = val
            
        else:
            
            print("one more mysterious case")
        
    def preprocess_tree(self):
        
         for state in range(len(self.weights)):
                
                self.add(state, self.root)
                 
    def sample(self):
        """
        random event drawing according to input probabilities
        
        """
        
        x = np.random.random()*self.sum
        bound = 0
        node = self.root
        
        #print(x, self.sum)
        #count = 0
        
        while node.depth < self.max_depth:
            
            #count += 1
            #print(count)
            
            if x < node.val[0] + bound:
                
                self.last_path[node.depth] = 0
                node = node.L
            
            else:
                
                self.last_path[node.depth] = 1
                bound += node.val[0]
                node = node.R
        
        self.last_visit = node.val
        
        return node.val
    
    def update_weights(self, new_proba):
        """
        update last visited event with probability
        new_proba
        
        """
        
        old_w = self.weights[self.last_visit]
        new_w = new_proba / (1 - new_proba) * (self.sum - old_w)
        
        node = self.root
        
        for i in range(self.max_depth):
            
            go_right = self.last_path[i]

            node.val[go_right] += new_w - old_w
            
            if go_right == 1:
                
                node = node.R
            
            else:
                
                node = node.L
        
        self.weights[self.last_visit] = new_w
        self.sum += new_w - old_w
        
        
    def delete_tree(self):
        
        self.root = None