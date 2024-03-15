import configparser
import os
import networkx as nx
import matplotlib.pyplot as plt

class DynamicBayesianNetwork(object):
    def __init__(self):
        pass

    def getGraphDetails(self,filepath):
        config = configparser.ConfigParser()
        config.read(filePath)
        try:
            graph = config.get('Details', 'graph2')
            graphpath = config.get('Details', 'graphNetwork.txt')
            nodecount = config.get('GraphDetails', 'nodecount')
            timeslice = config.get('GraphDetails', 'timeslice')
            query = config.get('Algorithm', 'query')
            applyPF = config.get('Algorithm', 'ParticleFilter')
        except configparser.Error as e:
            print(f"Error: {e}")
        print(graphpath)
        print(graph)
        print(nodecount)
        print(timeslice)
        print(query)
        print(applyPF)
        return graph, nodecount, timeslice, query, applyPF

#get the Configuration.ini details
def getDetails(self,filename):
    config = configparser.ConfigParser()
    config.read(filePath)
    graphpath=""
    if 'Details' in config:
        section = config['Details']
        graphpath = section.get('graphpath','graph.txt')
        timeslices = section.get('timeslices','1')
        PFAflag = section.get('PFA',True)
        Priors=section.get('Priors')
        TransitionModel=section.get('TransitionModel')
        Query=section.get('Query')

    print(graphpath)
    return graphpath,timeslices,PFAflag,Priors,TransitionModel,Query

#check if file exists
def file_exists(filename):
    return os.path.exists(filename)

#takes the graph that's given by the user and builds the bdn with timeslices, for future provide parameter to make it fully connected
def buildGraph(self, graphpath):
    with open(graphpath,"r") as file:
        next(file)
        parentchilddata = file.readlines()
    graphdict = {}
    for line in parentchilddata:
        components=line.split()
        parent = components[0]
        children = components[1:]
        graphdict[parent]=children
    print(graphdict)
    helperGraphBuilder(graphdict)

def helperGraphBuilderTest(graphdict):
    DBNgraph = nx.DiGraph()
    print("-------------------")
    print(graphdict)

#takes paretn-children dictionary and helps build the graph, needs to be customised
def helperGraphBuilder(graphdict):
    DBNgraph = nx.DiGraph()
    for node, neighbors in graphdict.items():
        DBNgraph.add_node(node)
        for neighbor in neighbors:
            DBNgraph.add_edge(node, neighbor)
    #draw the dbn
    pos = nx.spring_layout(DBNgraph) #customise graph later
    nx.draw(DBNgraph,pos,with_labels=True,node_color='green', node_size=2000, font_size=12, font_weight='bold', arrowsize=20)
    plt.show()

#returns parent and respective children, also have a function for a particular timeslice
def getParentChild(graphdict):
    with open(graphpath,"r") as file:
        next(file)
        parentchilddata = file.readlines()
    graphdict = {}
    for line in parentchilddata:
        components=line.split()
        parent = components[0]
        children = components[1:]
        graphdict[parent]=children
    return graphdict

#**must be in a different class/package
'''
Applying the PFA, to be revised
'''
def applyPFA(PFAflag, priors, queries, timestep):

    return 0
'''
currently hardcoded
Priors = {'1:A': ['1:0.9', '0:0.1'], '1:B': ['1:1']}
TransitionModel = {'A': [[0, 0, 0.4], [0, 1, 0.6], [1, 0, 0.8], [1, 1, 0.2]],
                       'B': [[0, 0, 0.3], [0, 1, 0.7], [1, 0, 0.1], [1, 1, 0.9]]}
Query = {'1': ['1:B=1', '1:A=1'], '2': ['2:C=0', '1:C=0', '2:B=1']}
'''
'''
input parameter: evidence
N: Number of samples to be maintained
dbn: The dynamic bayesian network-> 
priors-> P(X0), 
TM-> P(X1|X0), 
SM-> P(E1|X1)
'''
def applyPFATest(evidence, effect,N, TransitionModelDictionary, SensorModel, Prior):
    Samples={}
    for i in range(0,N):
        #propagation step 1
        nodesample, e= getSampleProbability(TransitionModelDictionary,Prior)
        if nodesample in Samples:
            Samples[nodesample].append(e)
        else:
            Samples[nodesample]=[e]

        #weights computation step 2
        Weights={}
        Weights= computeSensorModel(effect,evidence,cptsOfGraph,Prior)

    Samples = resampling(N,Samples,Weights)
    return Samples

#helper functions for PFA
def getSampleProbability(TransitionModelDictionary,Prior):
    return 0

def computeSensorModel(effect,evidence,cptsOfGraph,prior):
    #compute the cpts
    if prior == evidence:
       pass#take value from cpt
    else:
       pass #compute the cpt for sensor model
    smodel={}
    return smodel

#{'1:A':[1:0.9,0:0.1],'1:B':[1:1]}
def initialise(Priors):
    priors={}
    priors=eval(Priors)
    return priors

def getTransitionModel(TransitionModel):
    transitionModel={}
    return transitionModel

#make a separate getter?
def getQueries(Query):
    Queries={}
    return Queries

'''
@input: sampleProbabilities is a dictionary with samples and their respective probabilities
@output: normalised samples
'''
def resampling(N,S,sampleProbabilities):
    denominator=sum(sampleProbabilities.values())
    resampledNormalisedDictionary={}
    for nodesample, weight in sampleProbabilities.items():
        resampledNormalisedDictionary[nodesample]=float(weight/denominator)
    return resampledNormalisedDictionary

if __name__ == "__main__":
    print("The Dynamic Bayesian Network is:")
    g1 = DynamicBayesianNetwork
    filePath = "/Users/prathyushamallela/Documents/ThesisLearning/ParticleFilter/causalsupplychainProject/dbnetwork/Configuration.ini"
    if file_exists(filePath):
        print(f"The file exists.")
    else:
        print(f"The file '{filePath}' does not exists.")

    graphpath,timeslices,PFAflag,Priors,TransitionModel,Query=getDetails(g1,filePath)
    if file_exists(graphpath):
        print(f"The file exists.")
    else:
        print(f"The file '{graphpath}' does not exists.")
    print(timeslices, PFAflag, Priors, TransitionModel, Query)
    #buildGraph(g1, graphpath)

    #PFA if PFAflag==True
    #applyPFATest()
    TransitionModelDict=eval(TransitionModel)
    #print(TransitionModelDict.keys())
    QueryDict=eval(Query)
    #print(QueryDict.keys())
    N=3 #add it to configuration file**
    #QueryDict_items = list(QueryDict.items())
    #print(QueryDict_items)
    Prior = initialise(Priors)
    cptsOfGraph={}
    for i in range(1,int(timeslices)+1):
        effectEvidenceList=QueryDict.get(str(i))
        effect=effectEvidenceList[0]
        evidenceList=effectEvidenceList[1:]
        SensorModel = computeSensorModel(effect,evidenceList,cptsOfGraph,Prior)
        applyPFATest(evidenceList,effect, N, TransitionModelDict, SensorModel, Prior)








