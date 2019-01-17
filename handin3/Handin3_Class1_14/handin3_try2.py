import math
import numpy as np
import random

        
def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

def translate_path_to_indices(path):
    #return list(map(lambda x: int(x), path))
    sequence = list(path)
    index=0
    indices = []
    for c,i in enumerate(path):
        prev_index=index
        if i=='N':
            index = 3
        elif i=='C':
            if prev_index == 3:
                index = 2
            elif prev_index == 2:
                index = 1
            elif prev_index == 1:
                index = 0
            elif prev_index == 0:
                index = 2
        elif i=='R':
            if prev_index == 3:
                index = 4
            elif prev_index == 4:
                index = 5
            elif prev_index == 5:
                index = 6
            elif prev_index == 6:
                index = 4
        indices.append(index)
    assert len(indices) == len(sequence)
    return indices

def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_indices_to_path(indices):
    mapping = ['C', 'C', 'C', 'N', 'R', 'R', 'R']
    return ''.join([mapping[i] for i in indices])

class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)

def read_fasta_file(filename):
        """
        Reads the given FASTA file f and returns a dictionary of sequences.

        Lines starting with ';' in the FASTA file are ignored.
        """
        sequences_lines = {}
        current_sequence_lines = None
        with open(filename) as fp:
            for line in fp:
                line = line.strip()
                if line.startswith(';') or not line:
                    continue
                if line.startswith('>'):
                    sequence_name = line.lstrip('>')
                    current_sequence_lines = []
                    sequences_lines[sequence_name] = current_sequence_lines
                else:
                    if current_sequence_lines is not None:
                        current_sequence_lines.append(line)
        sequences = {}
        for name, lines in sequences_lines.items():
            sequences[name] = ''.join(lines)
        return sequences


def createPredFastaFile(name, ann):
    filename = name + ".fa"
    ofile = open(filename, "w+")
    ofile.write("> " + name + "\n" +ann + "\n")

    ofile.close()

def count_transitions_and_emissions(K, D, X, Z):
    """
    Returns a Kx1, KxK matrix and a KxD matrix containing counts cf. above
    """  
    trans_matrix = np.zeros((K,K))
    emi_matrix = np.zeros((K,D))
    
    for x1,z1 in zip(X,Z):
        print("Started counting transitions")
        z = translate_path_to_indices(z1)
        x = translate_observations_to_indices(x1)
        for i in range(len(z)-1):
            trans_matrix[z[i-1], z[i]] += 1 

        print("Started counting emissions")
        for i in range(len(z)): 
            emi_matrix[z[i], x[i]] += 1
    
    #Calculate the probabilities
    trans_matrix /= trans_matrix.sum(1, keepdims=True)
    emi_matrix /= emi_matrix.sum(1, keepdims=True)
    return trans_matrix,emi_matrix

    
def training_by_counting(K, D, x, z):
    """
    Returns a HMM trained on x and z cf. training-by-counting.
    """
    init_probs = np.zeros((K,))
    init_probs[3] = 1.0

    trans, emi = count_transitions_and_emissions(K,D,x,z)

    my_hmm = hmm(init_probs,trans,emi)
    return my_hmm

def opt_path_prob_log(w):
    return max([row[-1] for row in w])

def compute_w_log(model, x):
    K = len(model.init_probs)
    N = len(x)   
    w = np.full((K, N), -np.inf)
    log_init_probs = np.log(model.init_probs)
    log_emission_probs = np.log(model.emission_probs)
    log_trans_probs = np.log(model.trans_probs)
    w[:, 0] = log_init_probs + log_emission_probs[:, x[0]]
    
    #W recursion
    def compute_w_k_n(k,n):
        w[k, n] = np.max(log_emission_probs[k, x[n]] + w[:, n-1] + log_trans_probs[:, k])

    prev_percent = 0
    for j in range(1, N):
        for i in range(0, K):
            compute_w_k_n(i, j)
    print(opt_path_prob_log(w))
    return w

def backtrack_log(model, w, x):
    n = len(w[0])
    z = [0] * n
    z[-1] = np.argmax(w[:, n-1])
    for i in range(n-2, -1, -1):
        probs = [log(model.emission_probs[z[i+1]][x[i+1]]) + w[k][i]+ log(model.trans_probs[k][z[i+1]]) 
                 for k in range(w.shape[0])]
        z[i] = np.argmax(probs)
    return z

def viterbi(x,model):
    x_indices = translate_observations_to_indices(x)
    w = compute_w_log(model,x_indices)
    ann = backtrack_log(model,w,x_indices)
    vit = translate_indices_to_path(ann)
    return vit

def predictUnnanotatedGenomes(K,D):
    #GENOME 1
    g1 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome1.fa')
    x1 = g1['genome1']
    true_ann1 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann1.fa')
    z1 = true_ann1['true-ann1']
    
    #GENOME 2
    g2 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome2.fa')
    x2 = g2['genome2']
    true_ann2 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann2.fa')
    z2 = true_ann2['true-ann2']
    
    #GENOME 3
    g3 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome3.fa')
    x3 = g3['genome3']
    true_ann3 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann3.fa')
    z3 = true_ann3['true-ann3']
    
    #GENOME 4
    g4 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome4.fa')
    x4 = g4['genome4']
    true_ann4 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann4.fa')
    z4 = true_ann4['true-ann4']
    
    #GENOME 5
    g5 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome5.fa')
    x5 = g5['genome5']
    true_ann5 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann5.fa')
    z5 = true_ann5['true-ann5']
    
    #GENOME 6
    g6 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome6.fa')
    x6 = g6['genome6']
    
    #GENOME 7
    g7 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome7.fa')
    x7 = g7['genome7']
    
    #GENOME 8
    g8 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome8.fa')
    x8 = g8['genome8']
    
    #GENOME 9
    g9 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome9.fa')
    x9 = g9['genome9']
    
    #GENOME 10
    g10 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome10.fa')
    x10 = g10['genome10']
    
    #HMM
    X = [x1,x2,x3,x4,x5]
    Z = [z1,z2,z3,z4,z5]
    
    hmm = training_by_counting(K,D,X,Z)
    
    annotation6 = viterbi(x6,hmm)
    createPredFastaFile("pred-ann6",annotation6)
    
    annotation7 = viterbi(x7,hmm)
    createPredFastaFile("pred-ann7",annotation7)
    
    annotation8 = viterbi(x8,hmm)
    createPredFastaFile("pred-ann8",annotation8)
    
    annotation9 = viterbi(x9,hmm)
    createPredFastaFile("pred-ann9",annotation9)
    
    annotation10 = viterbi(x10,hmm)
    createPredFastaFile("pred-ann10",annotation10)
    
def predictAnnotatedGenomesCompare():
    #GENOME 1
    g1 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome1.fa')
    x1 = g1['genome1']
    true_ann1 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann1.fa')
    z1 = true_ann1['true-ann1']
    
    #GENOME 2
    g2 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome2.fa')
    x2 = g2['genome2']
    true_ann2 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann2.fa')
    z2 = true_ann2['true-ann2']
    
    #GENOME 3
    g3 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome3.fa')
    x3 = g3['genome3']
    true_ann3 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann3.fa')
    z3 = true_ann3['true-ann3']
    
    #GENOME 4
    g4 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome4.fa')
    x4 = g4['genome4']
    true_ann4 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann4.fa')
    z4 = true_ann4['true-ann4']
    
    #GENOME 5
    g5 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome5.fa')
    x5 = g5['genome5']
    true_ann5 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann5.fa')
    z5 = true_ann5['true-ann5']
    
    #GENOME 6
    g6 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome6.fa')
    x6 = g6['genome6']
    
    #GENOME 7
    g7 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome7.fa')
    x7 = g7['genome7']
    
    #GENOME 8
    g8 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome8.fa')
    x8 = g8['genome8']
    
    #GENOME 9
    g9 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome9.fa')
    x9 = g9['genome9']
    
    #GENOME 10
    g10 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome10.fa')
    x10 = g10['genome10']
    
    pred_ann6 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred_ann/pred-ann6.fa')
    pred6 = pred_ann6[' pred-ann6']
    
    pred_ann7 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred_ann/pred-ann7.fa')
    pred7 = pred_ann7[' pred-ann7']
    
    pred_ann8 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred_ann/pred-ann8.fa')
    pred8 = pred_ann8[' pred-ann8']
    
    pred_ann9 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred_ann/pred-ann9.fa')
    pred9 = pred_ann9[' pred-ann9']
    
    pred_ann10 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred_ann/pred-ann10.fa')
    pred10 = pred_ann10[' pred-ann10']
    
    X_pred = [x6,x7,x8,x9,x10]
    Z_pred = [pred6,pred7,pred8,pred9,pred10]
    
    hmm_pred = training_by_counting(K,D,X_pred,Z_pred)
    
    annotation1 = viterbi(x1,hmm_pred)
    createPredFastaFile("pred-ann1",annotation1)
    
    annotation2 = viterbi(x2,hmm_pred)
    createPredFastaFile("pred-ann2",annotation2)
    
    annotation3 = viterbi(x3,hmm_pred)
    createPredFastaFile("pred-ann3",annotation3)

    annotation4 = viterbi(x4,hmm_pred)
    createPredFastaFile("pred-ann4",annotation4)
    
    annotation5 = viterbi(x5,hmm_pred)
    createPredFastaFile("pred-ann5",annotation5)

def getAllX():
        #GENOME 1
    g1 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome1.fa')
    x1 = g1['genome1']
    
    #GENOME 2
    g2 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome2.fa')
    x2 = g2['genome2']
    
    #GENOME 3
    g3 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome3.fa')
    x3 = g3['genome3']
    
    #GENOME 4
    g4 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome4.fa')
    x4 = g4['genome4']
    
    #GENOME 5
    g5 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome5.fa')
    x5 = g5['genome5']
    
    #GENOME 6
    g6 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome6.fa')
    x6 = g6['genome6']
    
    #GENOME 7
    g7 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome7.fa')
    x7 = g7['genome7']
    
    #GENOME 8
    g8 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome8.fa')
    x8 = g8['genome8']
    
    #GENOME 9
    g9 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/genome9.fa')
    x9 = g9['genome9']
    
    X = [x1,x2,x3,x4,x5]
    return X

def getAllZ():
    true_ann1 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann1.fa')
    z1 = true_ann1['true-ann1']
    
    true_ann2 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann2.fa')
    z2 = true_ann2['true-ann2']
    
    true_ann3 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann3.fa')
    z3 = true_ann3['true-ann3']
    
    true_ann4 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann4.fa')
    z4 = true_ann4['true-ann4']
    
    true_ann5 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/datav2/true-ann5.fa')
    z5 = true_ann5['true-ann5']
    
    pred_ann6 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred_ann/pred-ann6.fa')
    z6 = pred_ann6[' pred-ann6']
    
    pred_ann7 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred_ann/pred-ann7.fa')
    z7 = pred_ann7[' pred-ann7']
    
    pred_ann8 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred_ann/pred-ann8.fa')
    z8 = pred_ann8[' pred-ann8']
    
    pred_ann9 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred_ann/pred-ann9.fa')
    z9 = pred_ann9[' pred-ann9']
    
    pred_ann10 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred_ann/pred-ann10.fa')
    z10 = pred_ann10[' pred-ann10']
    
    Z = [z1,z2,z3,z4,z5]
    return Z
    

class my_set:
    def __init__(self, train_set, validation_set):
        self.train_set = train_set
        self.validation_set = validation_set

def getFolds(nFolds,x,z):
    folds =[]

    return x_folds,z_folds
        
        
def kFoldValidation(X,Z): #all training data sets
    nFolds = len(X)
    print(len(X))
    
    for i in range(nFolds):
        print("Round ",i)
        x_val = X[i] # X for validation genome
        z_val = Z[i] #Z for validation genome
        x_train = []
        z_train = []
        for j in range(nFolds):
            print(j)
            if j != i:
                print("Adding to train set")
                x_train.append(X[j])
                z_train.append(Z[j])
        
        hmm = training_by_counting(7,4,x_train,z_train)
        pred = viterbi(x_val,hmm)
        num = i+1
        name = "K-fold_"+str(num)
        createPredFastaFile(name,pred)
        
        
if __name__ == '__main__':
    
    X = getAllX()
    print(len(X))
    Z = getAllZ()
    print(len(Z))
    
    kFoldValidation(X,Z)
    print("K-Fold validation done")
   
    

    
    
    
    

    
