import os
import numpy as np

def compress_kmeans(im, k, T, name):
    height, width, depth = im.shape
    data = im.reshape((height * width, depth))
    clustering, centroids, score = lloyds_algorithm(data, k, 5)
    means, covs, priors, llh = em_algorithm(data, k, 5, 0.001, means=centroids)
    
    centroids = means
    clustering = compute_em_cluster(means, covs, priors, data)
    
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
    img_facade = download_image('https://pixel.nymag.com/imgs/daily/intelligencer/2018/01/31/31-katy-perry-left-shark-superbowl.w700.h700.jpg')
    compress_kmeans(img_facade, k, T, 'nygaard_facade.jpg')

compress_facade()
