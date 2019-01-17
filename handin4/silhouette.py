def silhouette(data, clustering, verbose=0): 
    n, d = data.shape
    k = np.unique(clustering)[-1]+1

    def getDist(p1, p2):
        a = p1[0] - p2[0]
        b = p1[1] - p2[1]
        dist = math.sqrt(a * a + b * b)
        return dist
    
    def getAvgDistance(target, clusterIndex):
        totalDistance = 0
        indicies = np.argwhere(clustering == clusterIndex).flatten()
        
        if len(indicies) == 0:
            return -1
        
        for pointIndex in indicies:
            p1 = data[pointIndex]
            totalDistance += getDist(p1, target)
        
        return totalDistance / len(indicies)
    
    sumA = 0
    sumB = 0
    silh = None
    silhSum = 0
    silhNum = 0
    for i in range(n):
        x = data[i]
        clusterIndex = clustering[i]
        a = getAvgDistance(x, clusterIndex)
        
        if verbose >= 2:
            print('(', i, ') a:\t', a)
        
        minDist = -1

        for j in range(k):
            if j == clusterIndex:
                continue
            
            dist = getAvgDistance(x, j)
            if minDist == -1 or dist < minDist:
                minDist = dist
                
        b = minDist
        if verbose >= 2:
            print('(', i, ') b:\t', b)
        
        if a == -1 or b == -1:
            continue
        
        if verbose >= 1:
            print('(', i, ') silh:\t', (b - a) / max(a, b))
        silhSum += (b - a) / max(a, b)
        silhNum += 1
        
        sumA += a
        sumB += b
    
    if silhNum == 0:
        silh = 0
    else:
        silh = silhSum/silhNum
        if verbose >= 2:
            print("Avg A:", sumA/silhNum)
            print("Avg B:", sumB/silhNum)
        
    # END CODE

    return silh

def silhouette2(data, clustering): 
    n, d = data.shape
    k = np.unique(clustering)[-1]+1

    # YOUR CODE HERE
    silh = None
    
    def getAvgDistance(x, clusterIndex):
        totalDistance = 0
        indices = np.argwhere(clustering == clusterIndex).flatten()

        if len(indices) == 0:
            return -1

        for pointIndex in indices:
            p1 = data[pointIndex]
            dist = (x - clustering[k])**2
            distance = math.sqrt(np.sum(dist))
            totalDistance += distance

        return totalDistance / len(indices)
    
    a_sum = 0
    b_sum = 0
    final_sum = 0
    
    avg_dist_b = 0
    for j in range(n):
        point = data[j]
        clusterIndex = clustering[j]
        
        a = getAvgDistance(point,clusterIndex)
        
        min_dist_b =-1
        for i in range(k):
            if clusterIndex == i:
                continue
            
            dist = getAvgDistance(point,i)
            if min_dist_b == -1 or dist < min_dist_b:
                min_dist_b = dist           
     
    b = min_dist_b

    
    final_sum += (b-a)/max(a,b)
   
    a_sum += a
    b_sum += b
    
    silh = final_sum
            
    # END CODE

    return silh

clustering = compute_em_cluster(means,covs,probs_c,X)
silh = silhouette(X,clustering)
print(silh)




def silhouette3(data, clustering, verbose=0): 
    n, d = data.shape
    k = np.unique(clustering)[-1]+1
    
    a_sum = 0
    b_sum = 0
    silh = None
    silh_temp = 0
    num_silh = 0
    for i in range(n):
        x = data[i]
        clusterIndex = clustering[i]
        
        totalDistance = 0
        indices = np.argwhere(clustering == clusterIndex).flatten()
        
        for pointIndex in indices:
            p1 = data[pointIndex]
            dist = np.linalg.norm(p1 - clustering[i])**2
            totalDistance += dist
        
        a = totalDistance / len(indices)
        
        min_dist = -1

        for j in range(k):
            if j != clusterIndex:
                dist = np.linalg.norm(x[i] - clustering[i])**2
                if min_dist == -1 or dist < min_dist:
                    min_dist = dist
                
        b = min_dist
        
        if a == -1 or b == -1:
            continue
        
        silh_temp += (b - a) / max(a, b)
        num_silh +=1
        
        a_sum += a
        b_sum += b
        

    silh = silh_temp/num_silh
        
    # END CODE

    return silh


clustering, centroids, cost = lloyds_algorithm(X, 3, 100)
silh = silhouette(X,clustering)
print(silh)