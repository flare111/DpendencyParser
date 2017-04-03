from Configuration import Configuration
import Config

"""
Defines a transition-based parsing framework for dependency parsing
based on an arc-standard transition-based dependency parsing system (Nivre, 2004).

Author: Danqi Chen
Modified by: Heeyoung Kwon
"""

class ParsingSystem:
    def __init__(self, labels):
        self.singleRoot = True
        self.labels = labels
        self.transitions = []
        self.rootLabel = labels[0]
        self.makeTransitions()

    """
    Generate all possible transitions which this parsing system can
    take for any given configuration.
    """
    def makeTransitions(self):
        for label in self.labels:
            self.transitions.append("L(" + label + ")")
        for label in self.labels:
            self.transitions.append("R(" + label + ")")

        self.transitions.append("S")

    def initialConfiguration(self, s):
        c = Configuration(s)
        length = len(s)

        # For each token, add dummy elements to the configuration's tree
        # and add the words onto the buffer
        for i in range(1, length+1):
            c.tree.add(Config.NONEXIST, Config.UNKNOWN)
            c.buffer.append(i)

        # Put the ROOT node on the stack
        c.stack.append(0)

        return c

    def isTerminal(self, c):
        return c.getStackSize() == 1 and c.getBufferSize() == 0

    """
    Provide a static-oracle recommendation for the next parsing step to take
    """
    def getOracle(self, c, tree):
        w1 = c.getStack(1)
        w2 = c.getStack(0)
        if w1 > 0 and tree.getHead(w1) == w2:
            return "L(" + tree.getLabel(w1) + ")"
        elif w1 >= 0 and tree.getHead(w2) == w1 and not c.hasOtherChild(w2, tree):
            return "R(" + tree.getLabel(w2) + ")"
        else:
            return "S"

    """
    Determine whether the given transition is legal for this
    configuration.
    """
    def canApply(self, c, t):
        if t.startswith("L") or t.startswith("R"):
            label = t[2:-1]
            if t.startswith("L"):
                h = c.getStack(0)
            else:
                h = c.getStack(1)
            if h < 0:
                return False
            if h == 0 and label != self.rootLabel:
                return False

        nStack = c.getStackSize()
        nBuffer = c.getBufferSize()

        if t.startswith("L"):
            return nStack > 2
        elif t.startswith("R"):
            if self.singleRoot:
                return (nStack > 2) or (nStack == 2 and nBuffer == 0)
            else:
                return nStack >= 2
        else:
            return nBuffer > 0

    def apply(self, c, t):
        s1 = c.getStack(1)
        s2 = c.getStack(0)

        if t.startswith("R"):
            c.addArc(s1, s2, t[2:-1])
            c.removeTopStack()
            # print("applied transition: Right")
        elif t.startswith("L"):
            c.addArc(s2, s1, t[2:-1])
            c.removeSecondTopStack()
            # print("applied transition: Left")
        else:
            # print("applied transition: Shift")
            c.shift()
        return c

        """
        =================================================================

        Implement arc standard algorithm based on
        Incrementality in Deterministic Dependency Parsing(Nirve, 2004):
        Left-reduce
        Right-reduce
        Shift

        =================================================================
        """

    def numTransitions(self):
        return len(self.transitions)

    def printTransitions(self):
        for t in self.transitions:
            print (t)

    def getPunctuationTags(self):
        return ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]

    """
    Evaluate performance on a list of sentences, predicted parses, and gold parses
    """
    def evaluate(self, sentences, trees, goldTrees):
        result = []
        punctuationTags = self.getPunctuationTags()

        if len(trees) != len(goldTrees):
            print("Incorrect number of trees.")
            return None

        correctArcs = 0
        correctArcsNoPunc = 0
        correctHeads = 0
        correctHeadsNoPunc = 0

        correctTrees = 0
        correctTreesNoPunc = 0
        correctRoot = 0

        sumArcs = 0
        sumArcsNoPunc = 0

        for i in range(len(trees)):
            tree = trees[i]
            goldTree = goldTrees[i]
            tokens = sentences[i]

            if tree.n != goldTree.n:
                print("Tree", i+1, ": incorrect number of nodes.")
                return None

            if not tree.isTree():
                print("Tree", i+1, ": illegal.")
                return None

            nCorrectHead = 0
            nCorrectHeadNoPunc = 0
            nNoPunc = 0

            for j in range(1, tree.n+1):
                if tree.getHead(j) == goldTree.getHead(j):
                    correctHeads += 1
                    nCorrectHead += 1
                    if tree.getLabel(j) == goldTree.getLabel(j):
                        correctArcs += 1
                sumArcs += 1

                tag = tokens[j-1]['POS']
                if tag not in punctuationTags:
                    sumArcsNoPunc += 1
                    nNoPunc += 1
                    if tree.getHead(j) == goldTree.getHead(j):
                        correctHeadsNoPunc += 1
                        nCorrectHeadNoPunc += 1
                        if tree.getLabel(j) == goldTree.getLabel(j):
                            correctArcsNoPunc += 1

            if nCorrectHead == tree.n:
                correctTrees += 1
            if nCorrectHeadNoPunc == nNoPunc:
                correctTreesNoPunc += 1
            if tree.getRoot() == goldTree.getRoot():
                correctRoot += 1

        result = ""
        result += "UAS: " + str(correctHeads * 100.0 / sumArcs) + "\n"
        result += "UASnoPunc: " + str(correctHeadsNoPunc * 100.0 / sumArcsNoPunc) + "\n"
        result += "LAS: " + str(correctArcs * 100.0 / sumArcs) + "\n"
        result += "LASnoPunc: " + str(correctArcsNoPunc * 100.0 / sumArcsNoPunc) + "\n\n"

        result += "UEM: " + str(correctTrees * 100.0 / len(trees)) + "\n"
        result += "UEMnoPunc: " + str(correctTreesNoPunc * 100.0 / len(trees)) + "\n"
        result += "ROOT: " + str(correctRoot * 100.0 / len(trees)) + "\n"

        return result


