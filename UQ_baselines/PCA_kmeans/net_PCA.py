from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

class PCAKMeansNet(nn.Module):
    def __init__(self, X=None, input_dim=None, emb_dim=None, n_clusters=64):
        super(PCAKMeansNet, self).__init__()
        
        if X is not None:
            # Perform PCA
            self.pca = PCA(n_components=emb_dim, svd_solver='full')
            X_enc = self.pca.fit_transform(X)
            print(f"Embedding dimension: {emb_dim}")
            
            # Perform K-means clustering
            self.kmeans = KMeans(n_clusters=n_clusters)
            self.kmeans.fit(X_enc)
            
            # Store PCA components and K-means centroids as parameters
            self.register_buffer('pca_components', torch.tensor(self.pca.components_, dtype=torch.float32))
            self.register_buffer('centroids', torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32))
        else:
            self.register_buffer('pca_components', torch.zeros(emb_dim, input_dim, dtype=torch.float32))
            self.register_buffer('centroids', torch.zeros(n_clusters, emb_dim, dtype=torch.float32))
            
    def forward(self, x):
        with torch.no_grad():
            # Transform x using PCA components
            x_enc = torch.nn.functional.linear(x, self.pca_components)
            
            # Compute Euclidean distance to each centroid
            distances = torch.cdist(x_enc, self.centroids)
            
            # Return the minimum distance for each sample in the batch
            return distances.min(dim=1).values