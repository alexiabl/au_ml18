import math
import numpy as np

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

def translate_path_to_indices(path):
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
    
def validate_hmm(model):
    "Validates the hmm model"
    print("Validating model")
    validate_init = False
    validate_trans = False
    validate_emi = False
    
    if math.isclose(abs(sum(model.init_probs)),1.0,rel_tol=0.05):
        print(sum(model.init_probs))
        validate_init = True
    
    print("validate_init = ",validate_init)
    
    for i in range(len(model.trans_probs)):
        if math.isclose(abs(sum(model.trans_probs[i])),1.0,rel_tol=0.05):
            print(sum(model.trans_probs[i]))
            validate_trans = True
    
    print("validate_trans = ",validate_trans)
    
    for i in range(len(model.emission_probs)):
        print(sum(model.emission_probs[i]))
        if math.isclose(abs(sum(model.emission_probs[i])),1.0,rel_tol=0.05):
            print(sum(model.emission_probs[i]))
            validate_emi = True
    
    print("validate_emi = ",validate_emi)
    
    validate = validate_init & validate_trans & validate_emi
    
    return validate

class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs
              
def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)

def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]

def compute_w_log(model, x,z):
    x = translate_observations_to_indices(x1)
    w = [{}]
    for st in z:
        w[0][st] = {"prob": log(model.init_probs[st]) + log(model.emission_probs[st][x[0]]), "prev": None}
    
    for t in range(1, len(x)):
        w.append({})
        for st in z:
            max_trans_prob = w[t-1][z[0]]["prob"]+ log(model.trans_probs[z[0]][st])
            previous_state = z[0]
            for prev_st in z[1:]:
                trans_prob = w[t-1][prev_st]["prob"] + log(model.trans_probs[prev_st][st])
                if trans_prob > max_trans_prob:
                    max_trans_prob = trans_prob
                    previous_state = prev_st
                    
            max_prob = max_trans_prob + log(model.emission_probs[st][x[t]])
            w[t][st] = {"prob": max_prob, "prev": previous_state}

    return w

def backtrack_log(w):
    optimal_path = []
    max_prob = max(value["prob"] for value in w[-1].values())
    previous = None
    for state, value in w[-1].items():
        if value["prob"] == max_prob:
            optimal_path.append(state)
            previous = state
            break
    for t in range(len(w) - 2, -1, -1):
        optimal_path.insert(0, w[t + 1][previous]["prev"])
        previous = w[t + 1][previous]["prev"]
    
    ann = translate_indices_to_path(optimal_path)
    print("Highest probability= ",max_prob)
    return ann

def opt_path_prob_log(w):
    return max(w[i][-1] for i in range(len(w)))

def backtrack_log1(model,w):
    #z[n] = arg max( log p(x[n+1] | z[n+1]) + ω^[k][n] + log p(z[n+1] | k ) )
    #N = length of sequence
    N = len(w[0])
    k = len(w[1])
    np_w = np.asarray(w)
    
    z = [None for i in range(N)]
    z[N-1] = (np.argmax(w[i][N]) for i in range(k))
    previous = None
    
    np_trans = np.asarray(model.trans_probs)
    np_emi = np.asarray(model.emission_probs)
    
    for j in range(N-2,-1,-1):
        #for i in range(k):
            #z[j] = np.argmax(log(np_trans[j+1][j+1]) + np_w[i][j] + log(np_trans[z[j+1]][i]))
        #z[j] = np.argmax(log(np_emi[j+1][j+1]) + np_w[i][j] + log(np_trans[j+1][k]))
            #temp = 
        z[j] = np.argmax((log(np_emi[j+1][z[j+1]]) + w[i][j] + log(np_trans[z[j+1]][i]) for i in range(k)))
        
    return list(z)

def viterbi(x,z, model):
    z_indices = translate_path_to_indices(z)
    x_indices = translate_observations_to_indices(x)
    w = compute_w_log(model,x_indices,z_indices)
    ann = backtrack_log(w)
    return ann

def compute_accuracy(true_ann, pred_ann):
    if len(true_ann) != len(pred_ann):
        return 0.0
    return sum(1 if true_ann[i] == pred_ann[i] else 0 
               for i in range(len(true_ann))) / len(true_ann)

def count_transitions_and_emissions(K, D, x, z):
    """
    Returns a Kx1, KxK matrix and a KxD matrix containing counts cf. above
    """  
    trans_matrix = [ [ 0 for i in range(K) ] for j in range(K) ]
    emi_matrix = [ [ 0 for i in range(D) ] for j in range(K) ]  
    
    print("Started counting transitions")
    for i in range(len(z)-1):
        trans_matrix[z[i]][z[i+1]] += 1 

    print("Started counting emissions")
    size_x = len(x)
    size_z = len(z)
    for i,_ in enumerate(zip(x,z)):
        emi_matrix[z[i]][x[i]] += 1
                
    return trans_matrix,emi_matrix

    
def training_by_counting(K, D, X, Z):
    """
    Returns a HMM trained on x and z cf. training-by-counting.
    """
    init_probs = [ 0 for i in range(K) ]
    trans_probs = [ [ 0 for i in range(K) ] for j in range(K) ]
    emi_probs = [ [ 0 for i in range(D) ] for j in range(K) ] 
    occurrences_z = [0 for i in range(K)]
    occurrences_x = [0 for i in range(D)]
    
    for x,z in zip(X,Z):
        z_indices = translate_path_to_indices(z)
        x_indices = translate_observations_to_indices(x)

        num_sequences = len(z_indices)/3

        for i in range(0,len(z_indices),3):
            init_probs[z_indices[i]] += 1

        for j in range(K):
            init_probs[j] = init_probs[j] / num_sequences

        trans, emi = count_transitions_and_emissions(K,D,x_indices,z_indices)
        
        for i in range(K):
            for j in range(K):
                trans_probs[i][j] += trans[i][j]
        
        for i in range(K):
            for j in range(D):
                emi_probs[i][j] += emi[i][j]
        
        for i in range(len(z_indices)):
            occurrences_z[z_indices[i]] +=1

        for i in range(len(x_indices)):
            occurrences_x[x_indices[i]] += 1
    
    for i in range(K):
        for j in range(K):
            trans_probs[i][j] /= occurrences_z[j]

    for i in range(K):
        for j in range(D):
            emi_probs[i][j] /= occurrences_x[j]

    my_hmm = hmm(init_probs,trans_probs,emi_probs)
    print(init_probs)
    print("\n",trans_probs)
    print("\n",emi_probs)
    return my_hmm

        
def createPredFastaFile(name, ann):
    ofile = open(name, "w+")

    ofile.write(">" + name + "\n" +ann + "\n")

    ofile.close()
    

if __name__ == '__main__':
    
    K = 7
    D = 4
    
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
    X = {x1,x2,x3,x4,x5}
    Z = {z1,z2,z3,z4,z5}
    
    hmm = training_by_counting(K,D,X,Z)
    
    annotation6 = viterbi(x6,hmm)
    createPredFastaFile("pred-ann6",annotation6)
    pred_ann6 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred-ann6.fa')
    pred6 = pred_ann6['pred-ann6']
    
    annotation7 = viterbi(x7,hmm)
    createPredFastaFile("pred-ann7",annotation7)
    pred_ann7 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred-ann7.fa')
    pred7 = pred_ann7['pred-ann7']
    
    annotation8 = viterbi(x8,hmm)
    createPredFastaFile("pred-ann8",annotation8)
    pred_ann8 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred-ann8.fa')
    pred8 = pred_ann8['pred-ann8']
    
    annotation9 = viterbi(x9,hmm)
    createPredFastaFile("pred-ann9",annotation9)
    pred_ann9 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred-ann9.fa')
    pred9 = pred_ann9['pred-ann9']
    
    annotation10 = viterbi(x10,hmm)
    createPredFastaFile("pred-ann10",annotation10)
    pred_ann10 = read_fasta_file('/Users/alexiaborchgrevink/Desktop/AU_MachineLearning/Theoretical Excercises/au_ml18/handin3/Handin3_Class1_14/pred-ann10.fa')
    pred10 = pred_ann10['pred-ann10']
    
    


    
        
    