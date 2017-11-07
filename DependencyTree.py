import Config

"""
Represents a partial or complete dependency parse of a sentence, and
provides convenience methods for analyzing the parse.

"""

class DependencyTree:

    def __init__(self):
        self.n = 0
        self.head = [Config.NONEXIST]
        self.label = [Config.UNKNOWN]
        self.counter = -1

    """
    Add the next token to the parse.
    h: Head of the next token
    l: Dependency relation label between this node and its head
    """
    def add(self, h, l):
        self.n += 1
        self.head.append(h)
        self.label.append(l)

    """
    Establish a labeled dependency relation between the two given nodes.
    k: Index of the dependent node
    h: Index of the head node
    l: Label of the dependency relation
    """
    def set(self, k, h, l):
        self.head[k] = h
        self.label[k] = l

    def getHead(self, k):
        if k <= 0 or k > self.n:
            return Config.NONEXIST
        else:
            return self.head[k]

    def getLabel(self, k):
        if k <= 0 or k > self.n:
            return Config.NULL
        else:
            return self.label[k]

    """
    Get the index of the node which is the root of the parse
    (i.e., the node which has the ROOT node as its head).
    """
    def getRoot(self):
        for k in range(1, self.n+1):
            if self.getHead(k) == 0:
                return k
        return 0

    """
    Check if this parse has only one root.
    """
    def isSingleRoot(self):
        roots = 0
        for k in range(1, self.n+1):
            if self.getHead(k) == 0:
                roots += 1
        return roots == 1

    """
    Check if the tree is legal.
    """
    def isTree(self):
        h = []
        h.append(-1)
        for i in range(1, self.n+1):
            if self.getHead(i) < 0 or self.getHead(i) > self.n:
                return False
            h.append(-1)

        for i in range(1, self.n+1):
            k = i
            while k > 0:
                if h[k] >= 0 and h[k] < i: break
                if h[k] == i:
                    return False
                h[k] = i
                k = self.getHead(k)

        return True

    """
    Check if the tree is projective
    """
    def isProjective(self):
        if self.isTree() == False:
            return False
        self.counter = -1
        return self.visitTree(0)

    """
    Inner recursive function for checking projective of tree
    """
    def visitTree(self, w):
        for i in range(1, w):
            if self.getHead(i) == w and self.visitTree(i) == False:
                return False
        self.counter += 1
        if w != self.counter:
            return False
        for i in range(w+1, self.n+1):
            if self.getHead(i) == w and self.visitTree(i) == False:
                return False
        return True

    def equal(self, t):
        if t.n != self.n:
            return False
        for i in range(1, self.n+1):
            if self.getHead(i) != t.getHead(i):
                return False
            if self.getLabel(i) != t.getLabel(i):
                return False
        return True

    def print_tree(self):
        for i in range(1, self.n+1):
             print (str(i) + " " + str(self.getHead(i)) + " " + self.getLabel(i))
        # print

