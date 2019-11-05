#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>

using namespace std;

const int SIZE = 8;

void init(float *A, int size) {
  for (int i = 0; i < size; i++) {
    A[i] = static_cast<float>(i);
  }
}

void print(float *A, int size) {
  for (int i = 0; i < size; i++) {
    cout << A[i] << " ";
  }
  cout << "\n\n";
}

__global__ void kernel(float *array, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    array[index] += 1.f;
    if (index == 0)
      printf("### Array size: %d\n", size);
  }
}

template<typename T>
struct MemCpy {
  int *size;
  T *hstPtr;
  T *devPtr;
  MemCpy(){};
  MemCpy(int *s, T *h, T *d):
    size(s),
    hstPtr(h),
    devPtr(d){}
};

int main() {
  int size = SIZE;

  // Host array
  float *A_h;
  checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&A_h), size * sizeof(float)));

  // Device array
  float *dArray;
  cudaMalloc(reinterpret_cast<void **>(&dArray), size * sizeof(float));

  auto memcpy = MemCpy<float>(&size, A_h, dArray);

  init(A_h, size);
  cout << "Results after init:\n";
  print(memcpy.hstPtr, size);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaGraph_t graph;
  cudaGraphExec_t instance;


  // Create graph
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  cudaMemcpyAsync(reinterpret_cast<void *>(memcpy.devPtr), reinterpret_cast<void *>(memcpy.hstPtr),
                  *memcpy.size * sizeof(float), cudaMemcpyHostToDevice, stream);
  kernel<<<1, 32, 0, stream>>>(memcpy.devPtr, *memcpy.size);
  cudaMemcpyAsync(reinterpret_cast<void *>(memcpy.hstPtr), reinterpret_cast<void *>(memcpy.devPtr),
                  *memcpy.size * sizeof(float), cudaMemcpyDeviceToHost, stream);

  cudaStreamEndCapture(stream, &graph);

  checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));
  checkCudaErrors(cudaGraphLaunch(instance, stream));
  cudaStreamSynchronize(stream);

  cout << "Results from kernel:\n";
  print(memcpy.hstPtr, size);

  // New data: host array
  float *B_h;
  size = 2 * SIZE;
  checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&B_h), size * sizeof(float)));
  cudaMalloc(reinterpret_cast<void **>(&dArray), size * sizeof(float));
  memcpy.size = &size;
  memcpy.hstPtr = B_h;
  memcpy.devPtr = dArray;
  for (int i = 0; i < size; i++)
    B_h[i] = 1.f;
  cout << "Init new data:\n";
  print(memcpy.hstPtr, size);

  checkCudaErrors(cudaGraphLaunch(instance, stream));
  cudaStreamSynchronize(stream);
  cout << "Results from new data:\n";
  print(memcpy.hstPtr, size);

  //size_t nNodes;
  //// Get the number of graph nodes first
  //cudaGraphGetNodes(graph, NULL, &nNodes);
  //cout << "Number of nodes in graph: " << nNodes << "\n";
  //// Then get the nodes
  //cudaGraphNode_t nodes[nNodes];
  //cudaGraphGetNodes(graph, nodes, &nNodes);

  //// Get memcpy node parameters: cudaMemcpy3DParms
  //// https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaMemcpy3DParms.html#structcudaMemcpy3DParms
  //cudaMemcpy3DParms dparams;
  //cudaGraphMemcpyNodeGetParams(nodes[0], &dparams);

  //// Print the memcpy kind
  //// cudaMemcpyHostToHost = 0
  //// cudaMemcpyHostToDevice = 1
  //// cudaMemcpyDeviceToHost = 2
  //// cudaMemcpyDeviceToDevice = 3
  //// cudaMemcpyDefault = 4
  //cout << "kind: " << dparams.kind << "\n";

  //cout << "depth: " << dparams.extent.depth << "\t";
  //cout << "height: " << dparams.extent.height << "\t";
  //cout << "width: " << dparams.extent.width << "\n";
  //
  //cout << "A_h: " << A_h << "\t";
  //cout << "srcPtr: " << dparams.srcPtr.ptr << "\n";

  //cout << "dArray: " << dArray << "\t";
  //cout << "dstPtr: " << dparams.dstPtr.ptr << "\n";

  //// New data: host array
  //float *B_h;
  //size = 2 * SIZE;
  //checkCudaErrors(cudaMallocHost(reinterpret_cast<void **>(&B_h), size * sizeof(float)));
  //init(B_h, size);

  //// New data: Device array
  //cudaMalloc(reinterpret_cast<void **>(&dArray), size * sizeof(float));

  //// Update nodes[0] host2device memcpy parameter
  //dparams.srcPtr.ptr = B_h;
  //dparams.dstPtr.ptr = B_d;
  //dparams.extent.width = 2 * SIZE * sizeof(float);
  //cudaGraphMemcpyNodeSetParams(nodes[0], &dparams);

  //// Update nodes[1] kernel parameter: cudaKernelNodeParams
  //// https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaKernelNodeParams.html#structcudaKernelNodeParams
  //cudaKernelNodeParams kparams;
  //cudaGraphKernelNodeGetParams(nodes[1], &kparams);
  //cout << "blockDim.x: " << kparams.blockDim.x << "\t";
  //cout << "blockDim.y: " << kparams.blockDim.y << "\t";
  //cout << "blockDim.z: " << kparams.blockDim.z << "\n";
  //kparams.blockDim.x = 2 * SIZE;
  //cout << "blockDim.x: " << kparams.blockDim.x << "\t";
  //cout << "blockDim.y: " << kparams.blockDim.y << "\t";
  //cout << "blockDim.z: " << kparams.blockDim.z << "\n\n";
  //cudaGraphKernelNodeSetParams(nodes[1], &kparams);

  //// Update nodes[2] device2host memcpy parameter
  //cudaGraphMemcpyNodeGetParams(nodes[2], &dparams);
  //dparams.srcPtr.ptr = B_d;
  //dparams.dstPtr.ptr = B_h;
  //dparams.extent.width = 2 * SIZE * sizeof(float);
  //cudaGraphMemcpyNodeSetParams(nodes[2], &dparams);

  //print(B_h, size);

  //// cudaKernelNodeParams knp;
  //// cudaGraphKernelNodeGetParams(nodesQ[0], &knp);
  //// cudaGraphExecKernelNodeSetParams(graphExec, nodesP[0], &knp);
  //// cudaGraphLaunch(graphExec, stream);
  //// cudaDeviceSynchronize();

  cudaStreamDestroy(stream);

  //cudaFreeHost(A_h);
  //cudaFreeHost(B_h);
  //cudaFree(A_d);
  //cudaFree(B_d);

  return 0;
}
