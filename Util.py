from DependencyTree import DependencyTree

def loadConll(inFile):
    sents = []
    trees = []
    with open('data/' + inFile, 'r') as fin:
        sentenceTokens = []
        tree = DependencyTree()
        for line in fin:
            line = line.strip()
            line = line.split('\t')
            if len(line) < 10:
                if len(sentenceTokens) > 0:
                    trees.append(tree)
                    sents.append(sentenceTokens)
                    tree = DependencyTree()
                    sentenceTokens = []
            else:
                word = line[1]
                pos = line[4]
                if line[6] == '_':
                    head = -1
                else:
                    head = int(line[6])
                depType = line[7]

                token = {}
                token['word'] = word
                token['POS'] = pos
                token['head'] = head
                token['depType'] = depType
                sentenceTokens.append(token)

                tree.add(head, depType)

    return sents, trees


def writeConll(outFile, sentences, trees):
    with open(outFile, 'w') as fout:
        for i in range(len(sentences)):
            sent = sentences[i]
            tree = trees[i]
            for j in range(len(sent)):
                fout.write("%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t_\t_\n" % (j+1, sent[j]['word'], sent[j]['POS'], sent[j]['POS'], tree.getHead(j+1), tree.getLabel(j+1)))

            fout.write("\n")

def MakeUnlabeledConll(inFile):
    with open('data/' + inFile, 'r') as fin, open('data/' + inFile.split(".")[0]+"_unlabeld.conll", 'w') as fout:
        for line in fin:
            line = line.strip()
            line = line.split('\t')
            if len(line) < 10:
                fout.write("\n")
            else:
                line[6] = "_"
                line[7] = "_"
                fout.write("\t".join(line) + "\n")
"""
sents, trees = loadConll("train.conll")
print sents[1]
trees[1].print_tree()
"""

MakeUnlabeledConll("test.conll")
