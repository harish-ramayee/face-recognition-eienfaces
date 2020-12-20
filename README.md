# Face Recognition Using Eigenfaces

This project is my implementation of the 'Face Recognition using Eigenfaces' research paper by Matthew A. Turk and Alex P. Pentland, using the AR face database (down sampled) by A.M.Martinez.

This algorithm makes use of the concept of eigenvectors and PCA, wherein we find the principal components of the distribution of faces, or the
eigenvectors of the covariance matrix of the set of face images in the given AR down-sampled dataset. Every face image contributes more or less to each eigenvevtor
and this creates a ghostly face referred to as an eigenface. 
