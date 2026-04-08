# Computational-RA-Take-Home-Assessment
Korsunsky Lab — Brigham and Women's Hospital /HMS Coding Assessment

### Background
The Harmony algorithm performs iterative batch correction of single-cell genomics data. At its core, each iteration involves a sequence of matrix operations — clustering assignments, regression, and centroid updates — applied to a matrix of cells × embedding dimensions. As datasets grow to tens of millions of cells, runtime becomes a bottleneck, and parallelization is essential.

### Parallelization
The parallelization strategy implemented here uses OpenMP. This method is simpler to use compared to other tools such as Intel TBB. The loops that needed to be parallelized were also regular for loops (low complexity) which allowed for a simpler tool to be appropriate. 

It was important to consider that the two loop blocks could not be parallelized with each other. The first loop block computes the centroids (raw accumulated weighted embeddings per cluster) and the additional cluster_sums (total probability for cluster k). The second loop block normalizes the probabilities in this cluster_sums vector, hence the first loop block needs to be completed on all threads before the second loop block begins.

Another consideration was if the cells were divided among multiple threads, then there would need to be a way to prevent them from writing to the same index in the centroids matrix because different cells could still share the same k and d iteration.

This was handled by adding reduction (#pragma omp parallel for reduction(...)) to the first loop block. This gives each thread its own copy of the centroids matrix to which they can write to. Then at the end each centroid matrix across threads gets summed together. Then the second loop block is given another (#pragma omp parallel for) tag, creating a barrier that prevents this loop from executing before all previous threads are complete.
