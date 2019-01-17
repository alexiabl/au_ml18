import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.mixture import GaussianMixture as EM
from scipy.stats import multivariate_normal
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def lloyds_algorithm(X, k, T):
    """ Clusters the data of X into k clusters using T iterations of Lloyd's algorithm. 
        The data is assumed to have dimension 2 and each step of the algorithm is visualized. 
    
        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations to run Lloyd's algorithm. 
        
        Returns
        -------
        clustering: A vector of shape (n, ) where the i'th entry holds the cluster of X[i]. 
    """
    n, d = X.shape
    
    #assert d == 2, "The data is assumed to have dimension 2 so we can visualize it. "
    
    # Initialize clusters random. 
    clustering = np.random.randint(0, k, (n, )) 
    centroids  = np.zeros((k, d))
    
    # Used to stop if cost isn't improving (decreasing)
    cost = 0
    oldcost = 0
    
    # Column names
    
    for i in range(T):
        
        # 1. UPDATE CENTROIDS
        for i in range(0, k):

            ith_cluster_indexes = np.argwhere(clustering == i)
            sum_c = np.sum(X[ith_cluster_indexes], axis=0)

            if(len(ith_cluster_indexes) > 0):
                centroids[i] = sum_c/len(ith_cluster_indexes)
          
        # 2. REASSIGN POINTS TO CLUSTERS       
        for i in range(0, n):

            min_dist = float('inf')
            dst_cluster = 0

            for j in range(0, k):
                dist = np.linalg.norm(X[i] - centroids[j])**2
                if(dist < min_dist):
                    #print("found new min dist", dist, "OLD MIN", min_dist)
                    min_dist = dist
                    dst_cluster = j
            
            clustering[i] =  dst_cluster
        
        
        # COMPUTE COST
        cost = 0
        for j in range(n):
            cost += np.linalg.norm(X[j] - centroids[clustering[j]])**2    
        
        # Stop if cost didn't improve (decrease)
        if np.isclose(cost, oldcost): break #TODO
        oldcost = cost
        
    return clustering, centroids, cost

def compute_probs_cx(points, means, covs, probs_c):
    '''
    Input
      - points:    (n times d) array containing the dataset
      - means:     (k times d) array containing the k means
      - covs:      (k times d times d) array such that cov[j,:,:] is the covariance matrix of the j-th Gaussian.
      - probs_c:   (k) array containing priors
    Output
      - probs_cx:  (k times n) array such that the entry (i,j) represents Pr(C_i|x_j)
    '''
    # Convert to numpy arrays.
    points, means, covs, probs_c = np.asarray(points), np.asarray(means), np.asarray(covs), np.asarray(probs_c)
    
    # Get sizes
    n, d = points.shape
    k = means.shape[0]
    
    # Compute probabilities
    # This will be a (k, n) matrix where the (i,j)'th entry is Pr(C_i)*Pr(x_j|C_i).
    probs_cx = np.zeros((k, n))
    for i in range(k):
        
         # Handle numerical issues, these lines are unimportant for understanding the algorithm. 
        if np.allclose(np.linalg.det(covs[i]), 0):  # det(A)=0 <=> singular. 
            print("Numerical issues, run again. ") 
            return None, None
        
        probs_cx[i] = probs_c[i] * multivariate_normal.pdf(mean=means[i], cov=covs[i], x=points)
        
    
    # The sum of the j'th column of this matrix is P(x_j); why?
    probs_x = np.sum(probs_cx, axis=0, keepdims=True) 
    assert probs_x.shape == (1, n)
    
    # Divide the j'th column by P(x_j). The the (i,j)'th then 
    # becomes Pr(C_i)*Pr(x_j)|C_i)/Pr(x_j) = Pr(C_i|x_j)
    probs_cx = probs_cx / probs_x
    
    return probs_cx, probs_x


def em_algorithm(X, k, T, epsilon = 0.001, means=None):
    """ Clusters the data X into k clusters using the Expectation Maximization algorithm. 
    
        Parameters
        ----------
        X : Data matrix of shape (n, d)
        k : Number of clusters.
        T : Maximum number of iterations
        epsilon :  Stopping criteria for the EM algorithm. Stops if the means of
                   two consequtive iterations are less than epsilon.
        means : (k times d) array containing the k initial means (optional)
        
        Returns
        -------
        means:     (k, d) array containing the k means
        covs:      (k, d, d) array such that cov[j,:,:] is the covariance matrix of 
                   the Gaussian of the j-th cluster
        probs_c:   (k, ) containing the probability Pr[C_i] for i=0,...,k. 
        llh:       The log-likelihood of the clustering (this is the objective we want to maximize)
    """
    n, d = X.shape
    normalize = 1/n
    # Initialize and validate mean
   
    if means is None: 
        i_means = None
        means = np.random.rand(k, d)
    else:
        i_means = means.copy()

    # Initialize cov, prior
    probs_x  = np.zeros(n) 
    probs_cx = np.zeros((k, n)) 
    probs_c  = np.zeros(k) + np.random.rand(k)
    
    covs = np.zeros((k, d, d))
    for i in range(k):
        covs[i] = np.identity(d)
    probs_c = np.ones(k) / k
    
    # Column names
    # print("Iterations\tLLH")
    
    close = False
    old_means = np.zeros_like(means)
    iterations = 0

    while not(close) and iterations < T:
        old_means = means.copy()
        # Expectation step
        probs_cx, probs_x = compute_probs_cx(X, means, covs, probs_c)
        if probs_cx is None: return em_algorithm(X, k, T, epsilon = epsilon, means=i_means)
        assert probs_cx.shape == (k, n)
        # Maximization step
        # YOUR CODE HERE
        probs_c = np.sum(probs_cx, axis = 1)/n
        for i in range(k):
            #2: compute new means
            means[i] = np.dot(probs_cx[i],X)/np.sum(probs_cx[i])
  
            #3: compute new covariance
            sum = 0
            for j in range(n):
                xm = np.reshape((X[j] - means[i]), (2,1))
                sum += probs_cx[i][j] * np.dot(xm, xm.T)
            covs[i] = sum/np.sum(probs_cx[i])
        # END CODE
        
        # Compute per-sample average log likelihood (llh) of this iteration     
        llh = 1/n*np.sum(np.log(probs_x))
        # print(iterations+1, "\t\t", llh)

        dist = np.sqrt(((means - old_means) ** 2).sum(axis=1))
        close = np.all(dist < epsilon)
        iterations+=1

    # Validate output
    assert means.shape == (k, d)
    assert covs.shape == (k, d, d)
    assert probs_c.shape == (k,)

    return means, covs, probs_c, llh

def compute_em_cluster(means, covs, probs_c, data):
    probs_cx, _ = compute_probs_cx(data, means, covs, probs_c)

    clustering = np.argmax(probs_cx, axis = 0)

    assert clustering.shape[0] == data.shape[0]
    return clustering

def silhouette(data, clustering): # give figure at TA session

    n, d = data.shape
    k = np.unique(clustering)[-1] + 1

    #get clusters centroids
    centroids  = np.zeros((k, d))
    for i in range(0, k):
        ith_cluster_indexes = np.argwhere(clustering == i)
        sum_c = np.sum(X[ith_cluster_indexes], axis=0)

        if(len(ith_cluster_indexes) > 0):
            centroids[i] = sum_c/len(ith_cluster_indexes)

    #find closest cluster to each cluster i
    closest_clusters = np.zeros(k)
    for i in range(0, k):

        min_dist = float('inf')
        dst_cluster = 0

        for j in range(0, k):
            if(j!=i):
                dist = np.linalg.norm(centroids[i] - centroids[j])**2
                
                if(dist < min_dist):
                    min_dist = dist
                    dst_cluster = j
        
        closest_clusters[i] = dst_cluster
    


    s_i = np.zeros(n)
    a = np.zeros(n)
    b = np.zeros(n)

    for i in range(0, n):
        
        xi_cluster = clustering[i]
        closter_cluster = closest_clusters[clustering[i]]

        ith_cluster_indexes = np.argwhere(clustering == xi_cluster)
        closest_cluster_indexes = np.argwhere(clustering == closter_cluster)

        for j in range(0, len(ith_cluster_indexes)):
            a[i] += np.linalg.norm(X[i] - X[ith_cluster_indexes[j]])**2

        for j in range(0, len(closest_cluster_indexes)):
            b[i] += np.linalg.norm(X[i] - X[closest_cluster_indexes[j]])**2
        
        #average
        a[i] = a[i]/len(ith_cluster_indexes)
        b[i] = b[i]/len(closest_cluster_indexes)

        s_i[i] = (b[i] - a[i])/max(a[i], b[i])

    #print(s_i)

    assert(s_i.all() >=-1 and s_i.all() <= 1)


    s = s_i.sum()/n

    return s

def f1(predicted, labels):
    n, = predicted.shape
    r = np.max(predicted) + 1
    k = np.max(labels) + 1

    #compute contingency table N: r x k 
    contingency = np.zeros((r, k))
    for i in range(0, n):
        contingency[predicted[i]][labels[i]] += 1
    
    #calculate F1 score for each cluster
    max_row_contingency = contingency.argmax(axis=1)
    F_individual = np.zeros(r)

    for i in range(0, r):

        if i < k:
            n_i = np.sum(contingency[i])
            n_ij_i = contingency[i][i]
            m_ji = np.sum(contingency[:,i])
            prec_i = n_ij_i/n_i
            recall_i = n_ij_i/m_ji
            F_individual[i] = 2/((1/prec_i) + (1/recall_i))

    #calculate overall F score
    F_overall = np.sum(F_individual)/k

    assert contingency.shape == (r, k)
    return F_individual, F_overall, contingency


def download_image(url):
        filename = url[url.rindex('/')+1:]
        try:
            with open(filename, 'rb') as fp:
                return imageio.imread(fp) / 255
        except FileNotFoundError:
            import urllib.request
            with open(filename, 'w+b') as fp, urllib.request.urlopen(url) as r:
                fp.write(r.read())
                return imageio.imread(fp) / 255

def read_image(filename):
        try:
            with open(filename, 'rb') as fp:
                return imageio.imread(fp) / 255
        except FileNotFoundError:
        	print("File Not found")

def compress_kmeans(im, k, T, name):
    height, width, depth = im.shape
    data = im.reshape((height * width, depth))
    clustering, centroids, score = lloyds_algorithm(data, k, 5)
    
    # make each entry of data to the value of it's cluster
    data_compressed = data
    
    for i in range(k): data_compressed[clustering == i] = centroids[i] 
    
    im_compressed = data_compressed.reshape((height, width, depth))
    
    # The following code should not be changed. 
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(im_compressed)
    plt.savefig("compressed.jpg")
    plt.show()
    
    original_size   = os.stat(name).st_size
    compressed_size = os.stat('compressed.jpg').st_size
    print("Original Size: \t\t", original_size)
    print("Compressed Size: \t", compressed_size)
    print("Compression Ratio: \t", round(original_size/compressed_size, 5))

def compress_facade(k=4, T=100):
    img_facade = download_image('https://users-cs.au.dk/rav/ml/handins/h4/nygaard_facade.jpg')
    compress_kmeans(img_facade, k, T, 'nygaard_facade.jpg')

def sample(means, covs, num):
    mean = means[num]
    cov = covs[num]
     
    fig, ax = plt.subplots(1, 10, figsize=(8, 1))
    
    for i in range(10):
        img = multivariate_normal.rvs(mean=mean, cov=cov) # draw random sample   
        ax[i].imshow(img.reshape(28, 28), cmap='gray') # draw the random sample
    plt.show()


if __name__ == '__main__':

    iris = sklearn.datasets.load_iris()
    X, labels = sklearn.datasets.load_iris(True)
    X = X[:,0:2] # reduce to 2d so you can plot if you want
    k = 3
    T = 100


    #2.1 silhoutte scores

    # print("2.1 : Silhouttes")
    # print("k\t EM\t\t\t LLOYD")
    # for k in range(2, 10):
    #     means,covs,probs_c, llh = em_algorithm(X, k, 50)

    #     em_clustering = compute_em_cluster(means, covs, probs_c, X)
    #     em_sc = silhouette(X, em_clustering)

    #     lloyd_clustering, _, _ =  lloyds_algorithm(X, k, 50)
    #     lloyd_sc = silhouette(X, lloyd_clustering)

    #     print(k,"\t", em_sc,"\t", lloyd_sc)



    #2.2 F1 score
    # print("\n\n2.2 : F1 scores")
    # print("k\t EM\t\t\t LLOYD")
    # for k in range(2, 10):
    #     means,covs,probs_c, llh = em_algorithm(X, k, 50)

    #     em_clustering = compute_em_cluster(means, covs, probs_c, X)
    #     F_individual, EM_F_overall, contingency = f1(em_clustering, labels)

    #     lloyd_clustering, _, _ =  lloyds_algorithm(X, k, 50)
    #     _, LLOYD_F_overall, contingency = f1(lloyd_clustering, labels)

    #     print(k,"\t", EM_F_overall,"\t", LLOYD_F_overall)




    #3. compress image
    # bucarest_img = read_image('steagul-Romaniei.jpg')
    # k = 5
    # T = 100
    # compress_kmeans(bucarest_img, k, T, 'steagul-Romaniei.jpg')





    #4. sampling from MNIST
    # mnist = input_data.read_data_sets("data/")

    # X = mnist.train.images
    # y = mnist.train.labels

    # # One cluster for each digit
    # k = 10

    # # Run EM algorithm on 1000 images from the MNIST dataset. 
    # expectation_maximization = EM(n_components=k, max_iter=10, init_params='kmeans', covariance_type='diag', verbose=1, verbose_interval =1).fit(X)

    # means = expectation_maximization.means_
    # covs = expectation_maximization.covariances_
        
    # fig, ax = plt.subplots(1, k, figsize=(8, 1))

    # for i in range(k):
    #     ax[i].imshow(means[i].reshape(28, 28), cmap='gray')
        
    # plt.show()
    #sample(means, covs, 0)

  