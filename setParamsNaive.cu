#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int SIZE = 8;
const int N = 512 * 512;

void init(float *A, int size) {
  for (int i = 0; i < size; i++) {
    A[i] = static_cast<float>(i);
  }
}

void print(float *A) {
  for (int i = 0; i < SIZE; i++) {
    cout << A[i] << " ";
  }
  cout << "\n\n";
}

__global__ void kernel(float *array, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    array[index] += 1.f;
    //if (index == 0)
    //  printf("### array[%d] = %f\tArray size: %d\n", index, array[index], size);
  }
}

void runGraph(float* B_h, float* B_d, int size, cudaGraph_t& graph, cudaGraphExec_t& instance,
    cudaStream_t& stream) {
  cudaMemcpy3DParms dParams0, dParams2;
  cudaKernelNodeParams kParams;

  size_t nNodes;

  // Get the number of graph nodes first
  cudaGraphGetNodes(graph, NULL, &nNodes);
  // Then get nodes
  cudaGraphNode_t nodes[nNodes];
  cudaGraphGetNodes(graph, nodes, &nNodes);

  dim3 nthreads(size);
  
  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    auto ptr = &B_h[i*size];

    // #################### Update node 0 ####################
    checkCudaErrors(cudaGraphMemcpyNodeGetParams(nodes[0], &dParams0));
    //dParams0.srcArray = NULL;
    //dParams0.srcPos = make_cudaPos(0, 0, 0);
    dParams0.srcPtr = make_cudaPitchedPtr(ptr, size * sizeof(float), 1, 1);

    //dParams0.dstArray = NULL;
    //dParams0.dstPos = make_cudaPos(0, 0, 0);
    dParams0.dstPtr = make_cudaPitchedPtr(B_d, size * sizeof(float), 1, 1);
    dParams0.extent = make_cudaExtent(size * sizeof(float), 1, 1);
    //dParams0.kind = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaGraphExecMemcpyNodeSetParams(instance, nodes[0], &dParams0));

    // #################### Update node 1 ####################
    checkCudaErrors(cudaGraphKernelNodeGetParams(nodes[1], &kParams));
    //kParams.func = (void *)kernel;
    //kParams.gridDim = nblocks;
    kParams.blockDim = nthreads;
    //kParams.sharedMemBytes = 0;
    void *kernelArgs[2] = {(void *)&B_d, (void *)&size};
    kParams.kernelParams = kernelArgs;
    //kParams.extra = NULL;
    checkCudaErrors(cudaGraphExecKernelNodeSetParams(instance, nodes[1], &kParams));

    // #################### Update node 2 ####################
    cudaGraphMemcpyNodeGetParams(nodes[2], &dParams2);
    //dParams2.srcArray = NULL;
    //dParams2.srcPos = make_cudaPos(0, 0, 0);
    dParams2.srcPtr = make_cudaPitchedPtr(B_d, size * sizeof(float), 1, 1);
    //dParams2.dstArray = NULL;
    //dParams2.dstPos = make_cudaPos(0, 0, 0);
    dParams2.dstPtr = make_cudaPitchedPtr(ptr, size * sizeof(float), 1, 1);
    dParams2.extent = make_cudaExtent(size * sizeof(float), 1, 1);
    //dParams2.kind = cudaMemcpyDeviceToHost;
    checkCudaErrors(cudaGraphExecMemcpyNodeSetParams(instance, nodes[2], &dParams2));

    // Relaunch the graph with new parameters
    checkCudaErrors(cudaGraphLaunch(instance, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  high_resolution_clock::time_point end = high_resolution_clock::now();
  duration<double> time = duration_cast<duration<double>>(end - start);
  cout << "Graph runtime: " << time.count() << "\n";
  //print(&B_h[(N-1)*size]);
}

void runKernels(float* B_h, float* B_d, int size, cudaStream_t &stream) {
  dim3 nblocks(1);
  dim3 nthreads(size);

  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < N; i++) {
    auto ptr = &B_h[i*size];

    cudaMemcpyAsync(reinterpret_cast<void *>(B_d),
                    reinterpret_cast<void *>(ptr), size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    kernel<<<nblocks, nthreads, 0, stream>>>(B_d, size);
    cudaMemcpyAsync(reinterpret_cast<void *>(ptr),
                    reinterpret_cast<void *>(B_d), size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
  high_resolution_clock::time_point end = high_resolution_clock::now();
  duration<double> time = duration_cast<duration<double>>(end - start);
  cout << "Kernels runtime: " << time.count() << "\n";
  //print(&B_h[(N-1)*size]);
}

int main() {
  int size = SIZE;

  dim3 nblocks(1, 1, 1);
  dim3 nthreads(size, 1, 1);

  // Host array
  float *A_h;
  checkCudaErrors(
      cudaMallocHost(reinterpret_cast<void **>(&A_h), size * sizeof(float)));

  // Device array
  float *dArray;
  cudaMalloc(reinterpret_cast<void **>(&dArray), size * sizeof(float));

  init(A_h, size);
  cout << "Results after init:\n";
  print(A_h);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t graph;
  cudaGraphExec_t instance;

  // Create graph
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  cudaMemcpyAsync(reinterpret_cast<void *>(dArray),
                  reinterpret_cast<void *>(A_h), size * sizeof(float),
                  cudaMemcpyHostToDevice, stream);
  kernel<<<nblocks, nthreads, 0, stream>>>(dArray, size);
  cudaMemcpyAsync(reinterpret_cast<void *>(A_h),
                  reinterpret_cast<void *>(dArray), size * sizeof(float),
                  cudaMemcpyDeviceToHost, stream);

  cudaStreamEndCapture(stream, &graph);

  checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
  checkCudaErrors(cudaGraphLaunch(instance, stream));
  cudaStreamSynchronize(stream);

  cout << "First run:\n";
  print(A_h);

  size = 512;
  float *B_h;
  float *B_d;

  checkCudaErrors(
      cudaMallocHost(reinterpret_cast<void **>(&B_h), N * size * sizeof(float)));
  checkCudaErrors(
      cudaMalloc(reinterpret_cast<void **>(&B_d), size * sizeof(float)));

  for (int i = 0; i < N; i++)
    for (int j = 0; j < size; j++)
      B_h[i*size + j] = i;

  // Run graph
  runGraph(B_h, B_d, size, graph, instance, stream);
  // Run kernels
  runKernels(B_h, B_d, size, stream);

  cudaStreamDestroy(stream);

  cudaFreeHost(A_h);
  cudaFree(dArray);

  cudaFreeHost(B_h);
  cudaFree(B_d);

  return 0;
}
