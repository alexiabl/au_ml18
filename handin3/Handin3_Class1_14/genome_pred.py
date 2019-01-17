import numpy as np
import math

class hmm:
    def _init_(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

def count_transitions_and_emissions(K, D, x_list, z_list): #x and z are lists of 4 training sets
    """
    Returns a KxK matrix and a KxD matrix containing counts cf. above
    """
    A = np.zeros((K, K))
    phi = np.zeros((K, D))   
    for x1, z1 in zip(x_list, z_list):
        x=translate_observations_to_indices(x1)
        z=translate_path_to_indices(z1)
        # count transitions
        for i in range(len(z) - 1):
            A[z[i-1], z[i]] += 1
        # count emissions
        print(len(z))
        print(len(x))
        for i in range(len(z)): 
            phi[z[i], x[i]] += 1        
    A /= A.sum(1, keepdims=True)
    phi /= phi.sum(1, keepdims=True)
    #do this at the end
    return A, phi

def training_by_counting(K, D, x_list, z_list):
    """
    Returns a HMM trained on x and z cf. training-by-counting.
    """
    pi = np.zeros((K,))
    pi[3] = 1.    
    A, phi = count_transitions_and_emissions(K, D, x_list, z_list)
    return hmm(pi, A, phi)


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
    
    def compute_w_k_n(k,n):
        w[k, n] = np.max(log_emission_probs[k, x[n]] + w[:, n-1] + log_trans_probs[:, k])

    prev_percent = 0
    for j in range(1, N):   # for each column
        for i in range(0, K):   # for each row
            compute_w_k_n(i, j)
    print(opt_path_prob_log(w))
    return w

def backtrack_log(model, w, x):
    n = len(w[0])
    Z = [0] * n
    Z[-1] = np.argmax(w[:, n-1])
    for i in range(n-2, -1, -1):
        e = model.emission_probs[Z[i+1]][x[i+1]]
        probs = [log(e) + w[k][i]+ log(model.trans_probs[k][Z[i+1]]) 
                 for k in range(w.shape[0])]
        Z[i] = np.argmax(probs)
    return Z








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
    ofile = open(name, "w+")

    ofile.write(">" + name + "\n" +ann + "\n")

    ofile.close()

    
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
    

if _name_ == '_main_':
    
    K = 7
    D = 4
    
    #GENOME 1
    g1 = read_fasta_file('genome1.fa')
    x1 = g1['genome1']
    true_ann1 = read_fasta_file('true-ann1.fa')
    z1 = true_ann1['true-ann1']
    
    #GENOME 2
    g2 = read_fasta_file('genome2.fa')
    x2 = g2['genome2']
    true_ann2 = read_fasta_file('true-ann2.fa')
    z2 = true_ann2['true-ann2']
    
    #GENOME 3
    g3 = read_fasta_file('genome3.fa')
    x3 = g3['genome3']
    true_ann3 = read_fasta_file('true-ann3.fa')
    z3 = true_ann3['true-ann3']
    
    #GENOME 4
    g4 = read_fasta_file('genome4.fa')
    x4 = g4['genome4']
    true_ann4 = read_fasta_file('true-ann4.fa')
    z4 = true_ann4['true-ann4']
    
    #GENOME 5
    g5 = read_fasta_file('genome5.fa')
    x5 = g5['genome5']
    true_ann5 = read_fasta_file('true-ann5.fa')
    z5 = true_ann5['true-ann5']
    
    #GENOME 6
    g6 = read_fasta_file('genome6.fa')
    x6 = g6['genome6']
    
    #GENOME 7
    g7 = read_fasta_file('genome7.fa')
    x7 = g7['genome7']
    
    #GENOME 8
    g8 = read_fasta_file('genome8.fa')
    x8 = g8['genome8']
    
    #GENOME 9
    g9 = read_fasta_file('genome9.fa')
    x9 = g9['genome9']
    
    #GENOME 10
    g10 = read_fasta_file('genome10.fa')
    x10 = g10['genome10']
    
    #HMM
    X = [x1,x2,x3,x4,x5]
    Z = [z1,z2,z3,z4,z5]
    
    hmm = training_by_counting(K,D,X,Z)
    
    x6=translate_observations_to_indices(x6)
    annotation6 = compute_w_log(hmm,x6)
    annotation6 = backtrack_log(hmm,annotation6,x6)
    annotation6=''.join(str(e) for e in annotation6)
    createPredFastaFile("pred_ann6.fa",annotation6)
    print('annotation6 done')

    
    x7=translate_observations_to_indices(x7)
    annotation7 = compute_w_log(hmm,x7)
    annotation7 = backtrack_log(hmm,annotation7,x7)
    annotation7=''.join(str(e) for e in annotation7)
    createPredFastaFile("pred_ann6.fa",annotation7)
    print('annotation7 done')
    
    x8=translate_observations_to_indices(x8)
    annotation8 = compute_w_log(hmm,x8)
    annotation8 = backtrack_log(hmm,annotation8,x8)
    annotation8=''.join(str(e) for e in annotation8)
    createPredFastaFile("pred_ann6.fa",annotation8)
    print('annotation8 done')
    
    x9=translate_observations_to_indices(x9)
    annotation9 = compute_w_log(hmm,x9)
    annotation9 = backtrack_log(hmm,annotation9,x9)
    annotation9=''.join(str(e) for e in annotation9)
    createPredFastaFile("pred_ann6.fa",annotation9)
    print('annotation9 done')
    
    x10=translate_observations_to_indices(x10)
    annotation10 = compute_w_log(hmm,x10)
    annotation10 = backtrack_log(hmm,annotation10,x10)
    annotation10=''.join(str(e) for e in annotation10)
    createPredFastaFile("pred_ann6.fa",annotation10)
    print('annotation10 done')
    
    '''
    
    x6=translate_observations_to_indices(x6)
    annotation6 = compute_w_log(hmm,x6)
    annotation6 = backtrack_log(hmm,annotation6,x6)
    print(annotation6)
    
    annotation7 = viterbi(x7,z2,hmm)
    createPredFastaFile("pred_ann7",annotation7)
    pred_ann7 = read_fasta_file('pred_ann7.fa')
    
    annotation8 = viterbi(x8,z3,hmm)
    createPredFastaFile("pred_ann8",annotation8)
    pred_ann8 = read_fasta_file('pred_ann8.fa')
    
    annotation9 = viterbi(x9,z4,hmm)
    createPredFastaFile("pred_ann9",annotation9)
    pred_ann9 = read_fasta_file('pred_ann9.fa')
    
    annotation10 = viterbi(x10,z5,hmm)
    createPredFastaFile("pred_ann10",annotation9)
    pred_ann10 = read_fasta_file('pred_ann10.fa')
    
    '''