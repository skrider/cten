NVCC = nvcc
INCLUDES = -I./csrc -I./include
NVCCFLAGS = -dc
LIBS = 
CU_SRCS = $(wildcard csrc/*.cu)
OBJS = $(patsubst csrc/%.cu,%.o,$(CU_SRCS))
MAIN = main

.PHONY: depend clean

$(MAIN): $(OBJS)
	$(NVCC) $(INCLUDES) -o $(MAIN) $(OBJS) $(LIBS)

%.o: csrc/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(CU_SRCS)
	$(NVCC) -M $(INCLUDES) $(CU_SRCS) >> ./Makefile.dep
	
-include Makefile.dep