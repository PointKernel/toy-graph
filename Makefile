CXX      = nvcc
#CXXFLAGS = -g -O3 -std=c++11 --compiler-options -Wall -arch=sm_60 -Wno-deprecated-gpu-targets -lcublas # Pascal
CXXFLAGS = -g -O3 -std=c++11 --compiler-options -Wall -arch=sm_70 -Wno-deprecated-gpu-targets -lcublas # Volta
TARGET1  = testGraph
TARGET2  = setParams
INCL     = -I${CUDA_HOME}/include -I${CUDA_HOME}/samples/common/inc

all: target1 target2

target1: $(TARGET1).o
	$(CXX) $(INCL) $(CXXFLAGS) $(TARGET1).o -o $(TARGET1)

target2: $(TARGET2).o
	$(CXX) $(INCL) $(CXXFLAGS) $(TARGET2).o -o $(TARGET2)

%.o: %.cu
	$(CXX) -c $(INCL) $(CXXFLAGS) $< -o $@

clean:
	rm -f *.o *.lst $(TARGET1) $(TARGET2) *.ptx *.cub
