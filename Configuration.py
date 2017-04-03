from DependencyTree import DependencyTree
import Config

"""
Descrive the current configuration of a parser

This class uses an indexing scheme where an index of zero refers to
the ROOT node and actual word indices begin at one.

Author: Danqi Chen
Modified by: Heeyoung Kwon
"""

class Configuration:

    def __init__(self, sentence):
        self.stack = []
        self.buffer = []

        self.tree = DependencyTree()
        self.sentence = sentence

    def shift(self):
        k = self.getBuffer(0)
        if k == Config.NONEXIST:
            return False
        self.buffer.pop(0)
        self.stack.append(k)
        return True

    def removeSecondTopStack(self):
        nStack = self.getStackSize()
        if nStack < 2:
            return False
        self.stack.pop(-2)
        return True

    def removeTopStack(self):
        nStack = self.getStackSize()
        if nStack <= 1:
            return False
        self.stack.pop()
        return True

    def getStackSize(self):
        return len(self.stack)

    def getBufferSize(self):
        return len(self.buffer)

    def getSentenceSize(self):
        return len(self.sentence)


    def getHead(self, k):
        return self.tree.getHead(k)

    def getLabel(self, k):
        return self.tree.getLabel(k)

    """
    Get the token index of the kth word on the stack.
    If stack doesn't have an element at this index, return Config.NONEXIST
    """
    def getStack(self, k):
        nStack = self.getStackSize()
        if k >= 0 and k < nStack:
            return self.stack[nStack-1-k]
        else:
            return Config.NONEXIST

    """
    Get the token index of the kth word on the buffer.
    If buffer doesn't have an element at this index, return Config.NONEXIST
    """
    def getBuffer(self, k):
        if k >= 0 and k < self.getBufferSize():
            return self.buffer[k]
        else:
            return Config.NONEXIST

    """
    Get the word at index k
    """
    def getWord(self, k):
        if k == 0:
            return Config.ROOT
        else:
            k -= 1

        if k < 0 or k >= len(self.sentence):
            return Config.NULL
        else:
            return self.sentence[k]['word']

    """
    Get the POS at index k
    """
    def getPOS(self, k):
        if k == 0:
            return Config.ROOT
        else:
            k -= 1

        if k < 0 or k >= len(self.sentence):
            return Config.NULL
        else:
            return self.sentence[k]['POS']

    """
    Add an arc with the label l from the head node h to the dependent node t.
    """
    def addArc(self, h, t, l):
        self.tree.set(t, h, l)

    """
    Get cnt-th leftmost child of k.
    (i.e., if cnt = 1, the leftmost child of k will be returned,
           if cnt = 2, the 2nd leftmost child of k will be returned.)
    """
    def getLeftChild(self, k, cnt):
        if k < 0 or k > self.tree.n:
            return Config.NONEXIST

        c = 0
        for i in range(1, k):
            if self.tree.getHead(i) == k:
                c += 1
                if c == cnt:
                    return i
        return Config.NONEXIST

    """
    Get cnt-th rightmost child of k.
    (i.e., if cnt = 1, the rightmost child of k will be returned,
           if cnt = 2, the 2nd rightmost child of k will be returned.)
    """

    def getRightChild(self, k, cnt):
        if k < 0 or k > self.tree.n:
            return Config.NONEXIST

        c = 0
        for i in range(self.tree.n, k, -1):
            if self.tree.getHead(i) == k:
                c += 1
                if c == cnt:
                    return i
        return Config.NONEXIST

    def hasOtherChild(self, k, goldTree):
        for i in range(1, self.tree.n+1):
            if goldTree.getHead(i) == k and self.tree.getHead(i) != k:
                return True
        return False

    """
    Returns a string that concatenates all elements on the stack and buffer, and head / label
    """
    def getStr(self):
        s = "[S]"
        for i in range(self.getStackSize()):
            if i > 0:
                s += ","
            s += self.stack[i]

        s += "[B]"
        for i in range(self.getBufferSize()):
            if i > 0:
                s += ","
            s += self.buffer[i]

        s += "[H]"
        for i in range(1, self.tree.n+1):
            if i > 1:
                s += ","
            s += self.getHead(i) + "(" + self.getLabel(i) + ")"

        return s
