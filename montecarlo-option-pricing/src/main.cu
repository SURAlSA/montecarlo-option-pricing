#include "util.h"

// CUDA kernel for Monte Carlo simulation on GPU
// Each thread handles a specified number of simulation paths
__global__ void monteCarloMultiPathKernel(
    double *partialSums,      // stores sum of payoffs calculated by each thread
    float S0,                 // Initial stock price
    float K,                  // Strike price of option
    float r,                  // Risk-free interest rate
    float sigma,              // Volatility of stock
    float T,                  // Time to maturity (in years)
    int pathsPerThread,       // number of simulation paths each thread will compute
    unsigned long long seed   // Seed random number generator to ensure different sequences
)
{
    // Calculate a unique global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize pseudo-random number generator state for current thread
    // thread ID and seed are used to ensure each thread gets a unique sequence of random numbers
    curandState state;
    curand_init(seed, tid, 0, &state);

    // Variable to accumulate sum of payoffs for this thread
    double sum = 0.0;

    // Loop to run specified number of Monte Carlo paths for this thread
    for (int i = 0; i < pathsPerThread; i++) 
    {
        // Generate a random number from a standard normal distribution using thread's state
        float Z = curand_normal(&state);

        // Calculate stock price at maturity (ST) using geometric Brownian motion formula
        // This is core of Monte Carlo simulation.
        float ST = S0 * expf((r - 0.5f * sigma * sigma) * T + sigma * sqrtf(T) * Z);
        
        // Calculate payoff for a call option (max(ST - K, 0)) and add it to thread's sum
        sum += fmaxf(ST - K, 0.0f);
    }

    // Store final accumulated sum for this thread in global `partialSums` array
    partialSums[tid] = sum;
}

// -----------------------------------------------------------------------------

// CPU implementation of Monte Carlo simulation
// Used as a reference to compare against GPU performance and accuracy
double monteCarloCPU(int nPaths, float S0, float K, float r, float sigma, float T) 
{
    double sum = 0.0;
    for (int i = 0; i < nPaths; i++) 
    {
        // Generate two uniform random numbers (U1, U2) from [0, 1]
        float U1 = (float)rand() / RAND_MAX;
        float U2 = (float)rand() / RAND_MAX;

        // Use Box-Muller transform to convert uniform random numbers into a standard normal random number (Z)
        float Z = sqrtf(-2.0f * logf(U1)) * cosf(2.0f * M_PI * U2);
        
        // Calculate stock price at maturity (ST)
        float ST = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
        
        // Calculate payoff and add it to total sum
        sum += fmax(ST - K, 0.0);
    }
    // Return discounted average payoff, which is estimated option price
    return exp(-r * T) * (sum / nPaths);
}

// -----------------------------------------------------------------------------

// Function to calculate theoretical option price using Black-Scholes formula
// This is "correct" value Monte Carlo simulations should converge towards
double blackScholesCall(float S0, float K, float r, float sigma, float T) 
{
    // Calculate d1 and d2 values, which are key components of Black-Scholes formula
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    
    // Define a lambda function for cumulative distribution function (CDF) of standard normal distribution
    // This is used to calculate N(d1) and N(d2)
    auto N = [](double x){ return 0.5 * erfc(-x / sqrt(2.0)); };
    
    // Return final Black-Scholes call option price
    return S0 * N(d1) - K * exp(-r * T) * N(d2);
}

// -----------------------------------------------------------------------------

int main() {
    // Define financial parameters for option
    float S0 = 100.0f; // Initial stock price
    float K = 100.0f;  // Strike price
    float r = 0.05f;   // Risk-free rate
    float sigma = 0.2f;  // Volatility
    float T = 1.0f;    // Time to maturity

    // Define different numbers of simulation paths to test
    int pathCounts[] = {10000000, 50000000, 100000000};
    int nExperiments = sizeof(pathCounts) / sizeof(pathCounts[0]);
    
    // Calculate exact Black-Scholes price once for comparison
    double bsPrice = blackScholesCall(S0, K, r, sigma, T);

    // Open a CSV file to save results
    std::ofstream out("results.csv");
    out << "nPaths,CPUPrice,GPUPrice,BlackScholes,CPUTime,GPUTime\n";

    // Loop through each experiment (different number of paths)
    for (int e = 0; e < nExperiments; e++) 
    {
        int nPaths = pathCounts[e];

        // --- CPU Simulation Section ---
        
        // Start timing CPU simulation
        auto startCPU = std::chrono::high_resolution_clock::now();
        // Run CPU Monte Carlo simulation
        double cpuPrice = monteCarloCPU(nPaths, S0, K, r, sigma, T);
        // End timing
        auto endCPU = std::chrono::high_resolution_clock::now();
        // Calculate elapsed CPU time
        double cpuTime = std::chrono::duration<double>(endCPU - startCPU).count();

        // --- GPU Simulation Section ---

        // Define CUDA grid and block dimensions
        int threadsPerBlock = 256;
        int pathsPerThread = 1000; // Each thread simulates 1000 paths
        // Calculate total number of threads needed
        int nThreads = (nPaths + pathsPerThread - 1) / pathsPerThread;
        // Calculate number of blocks needed based on threads per block
        int nBlocks = (nThreads + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate memory on GPU to store partial sums
        double *d_partialSums;
        cudaMalloc(&d_partialSums, nThreads * sizeof(double));

        // Start timing GPU simulation
        auto startGPU = std::chrono::high_resolution_clock::now();
        
        // Launch CUDA kernel on GPU
        monteCarloMultiPathKernel<<<nBlocks, threadsPerBlock>>>(
            d_partialSums, S0, K, r, sigma, T, pathsPerThread, (unsigned long long)time(0));
            
        // Wait for GPU to finish its computation
        cudaDeviceSynchronize();
        
        // End timing
        auto endGPU = std::chrono::high_resolution_clock::now();

        // Allocate memory on CPU (host) to receive results from GPU
        double *h_partialSums = new double[nThreads];
        
        // Copy results (partial sums) from GPU memory to CPU memory
        cudaMemcpy(h_partialSums, d_partialSums, nThreads * sizeof(double), cudaMemcpyDeviceToHost);

        // Sum up partial sums from all threads to get total sum
        double sumGPU = 0.0;
        for (int i = 0; i < nThreads; i++) sumGPU += h_partialSums[i];
        
        // Calculate final GPU-estimated option price
        double gpuPrice = exp(-r * T) * (sumGPU / nPaths);
        // Calculate elapsed GPU time
        double gpuTime = std::chrono::duration<double>(endGPU - startGPU).count();

        // --- Output and Cleanup Section ---

        // Write results for current experiment to CSV file
        out << nPaths << "," << cpuPrice << "," << gpuPrice << "," << bsPrice
            << "," << cpuTime << "," << gpuTime << "\n";

        // Print results to console for real-time monitoring
        std::cout << "Paths: " << nPaths
                  << " | CPU: " << cpuPrice << " (" << cpuTime << "s)"
                  << " | GPU: " << gpuPrice << " (" << gpuTime << "s)"
                  << " | Black-Scholes: " << bsPrice << std::endl;

        // Free memory allocated on CPU and GPU
        delete[] h_partialSums;
        cudaFree(d_partialSums);
    }

    out.close();
    std::cout << "Simulation complete. Results written to results.csv\n";
    return 0;
}