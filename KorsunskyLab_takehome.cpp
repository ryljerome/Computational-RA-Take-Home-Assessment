#include <vector>
#include <omp.h>
#include <iostream>
#include <random>
#include <algorithm>

// OpenMP vs Intel TBB
// OpenMP is simpler to use and for regular loops whereas Intel TBB is for more complex tasks
// OpenMP involves minimal coding changes to the loops in order to parallelize them

// user-defined reduction for std::vector<float>
#pragma omp declare reduction(vec_plus : std::vector<float> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<float>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

// R: (NxK) matrix of soft cluster assignments (row-major)
//  R[i][k] = probability that cell i belongs to cluster k
// Z: (NxD) matrix of cell embeddings (row-major)
//  Z[i][d] = embedding coordinate d for cell i
// centroids: (KxD) output matrix of weighted cluster centroids (row-major)
//
// All matrices are passed as flat vectors in row-major order.
// Dimensions: N = number of cells, K = number od clusters, D = embedding dims.

// For each cluster, calculate the mean position -- cells with a high probability of belonging to the cluster (from the soft assignment) pull the centroid towards them
void compute_centroids(
    const std::vector<float>& R,    // N x K; the soft cluster assignments
    const std::vector<float>& Z,    // N x D; cell embeddings
    std::vector<float>& centroids,  // K x D; output
    int N, int K, int D)
{
    // Zero out centroids for fresh start
    std::fill(centroids.begin(), centroids.end(), 0.0f);

    // compute cluster_sums outside of loop to avoid having each thread compute it
    std::vector<float> cluster_sums(K, 0.0f);   // higher cluster_sums[k] means more cells belong to cluster k

    // Accumulate weighted embeddings
    #pragma omp parallel for    \
        reduction(vec_plus : centroids) \
        reduction(vec_plus : cluster_sums)   // reduction prevents different cells from writing to the same location in centroids matrix; reduction creates private copies for each thread
    for (int i = 0; i < N; ++i) {   // loops over i cells, give different cells to different threads
        for (int k = 0; k < K; ++k) {   // loops over each cluster probabilty for cell
            float r_ik = R[i * K + k];  // defining the probability that cell i belongs to cluster k
            cluster_sums[k] += r_ik;   // update weight for the current cluster

            for (int d = 0; d< D; ++d) {
                centroids[k * D + d] += r_ik * Z[i * D + d];    // the higher r_ik, the more the cell pulls the centroid toward it
            }
            }
    }

    // Normalize by total weight per cluster
    #pragma omp parallel for    // this loop can only happen after the first block is complete
    for (int k = 0; k < K; ++k) {
        float total_weight = cluster_sums[k];
        if (total_weight > 0) {
            for (int d = 0; d < D; ++d) {
                centroids[k * D + d] /= total_weight;
            }
        }
    }

}

// for (e.g. N = 1,000,000, K = 100, D = 50):

int main() {
    int D = 50;
    int N = 1000000;
    int K = 100;

    // random number generation:
    // need const std::vector<float>& R, const std::vector<float>& Z
    // vector for R needs to sum to 1
    // vector for Z is random floats
    // vector for centroids
    std::mt19937 rng(42);   // set seed
    std::normal_distribution<float> normal(0.0f, 1.0f);     // representative of embedding values
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);      // representative of probabilities 

    // generate Z embedding vector using normal dist
    std::vector<float> Z(N * D);
    for (int i = 0; i < N * D; ++i) {
        Z[i] = normal(rng);
    }

    // generate R
    std::vector<float> R(N * K);
    for (int i = 0; i < N; ++i) {
        float row_sum = 0.0f;       // keep track of row_sum, starting at 0
        for (int k = 0; k < K; ++k) {
            float val = uniform(rng);   // generate values from uniform distribution to get probabilities
            R[i * K + k] = val;
            row_sum += val;             // add new value to row_sum each loop
        }
        for (int k = 0; k < K; ++k) {   // once row is generated, normalize the row
            R[i * K + k] /= row_sum;
        }
    }

    // run with 1 thread, 4 threads, 8 threads
    std::vector<int> threads = {1, 4, 8};
    std::vector<double> timer_counts(3);    // vector to keep track of runtimes

    // run the function

    for (int t = 0; t < 3; ++t){
        // intialize centroids
        std::vector<float> centroids(K * D, 0.0f);

        omp_set_num_threads(threads[t]);     // set number of threads

        double start = omp_get_wtime();     // start timer
        compute_centroids(R, Z, centroids, N, K, D);    // run function
        double end = omp_get_wtime();       // stop timer

        // record the time
        timer_counts[t] = end - start;
    }

    // format timer_counts as table
    std::cout << "Threads | Time (s)\n";
    std::cout << "--------|---------\n";

    for (int t = 0; t < 3; ++t) {
        std::cout << threads[t] << "       | "
                  << timer_counts[t] << "\n";
    }

    return 0;

}

// command line:
//g++ KorsunskyLab_takehome.cpp -o take_home -fopenmp 2>&1
//.\take_home.exe
//git remote add origin git@github.com:ryljerome/Computational-RA-Take-Home-Assessment.git
//git branch -M main
//git push -u origin main