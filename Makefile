CXX = g++
NVCC = nvcc
INCLUDES = -I./csrc -I./include
CXXFLAGS = -Wall -std=c++11
NVCCFLAGS = -dc
LIBS = 
CPP_SRCS = $(wildcard csrc/*.cpp)
CU_SRCS = $(wildcard csrc/*.cu)
OBJS = $(patsubst csrc/%.cpp,%.o,$(CPP_SRCS)) $(patsubst csrc/%.cu,%.o,$(CU_SRCS))
MAIN = main

.PHONY: depend clean

$(MAIN): $(OBJS)
	$(NVCC) $(INCLUDES) -o $(MAIN) $(OBJS) $(LIBS)

%.o: csrc/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: csrc/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(CPP_SRCS) $(CU_SRCS)
	$(CXX) -MM $(INCLUDES) $(CPP_SRCS) > ./Makefile.dep
	$(NVCC) -M $(INCLUDES) $(CU_SRCS) >> ./Makefile.dep
	
-include Makefile.dep