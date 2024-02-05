NVCC = nvcc
INCLUDES = -I./csrc -I./include -I./vendor/cutlass/include
NVCCFLAGS = -dc --expt-relaxed-constexpr -g -Xptxas -v
LIBS = 
CU_SRCS = $(wildcard csrc/*.cu)
OBJS = $(patsubst csrc/%.cu,%.o,$(CU_SRCS))
PTX = $(patsubst csrc/%.cu,%.ptx,$(CU_SRCS))
MAIN = main

.PHONY: depend clean format

$(MAIN): $(OBJS)
	$(NVCC) $(INCLUDES) -o $(MAIN) $(OBJS) $(LIBS)

%.o: csrc/%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

# ptx for inspection purposes
%.ptx: csrc/%.cu
	$(NVCC) --ptx --expt-relaxed-constexpr $(INCLUDES) -c $< -o $@

ptx: $(OBJS)
	$(NVCC) --ptx $(INCLUDES) -o $(MAIN) $(OBJS) $(LIBS)

clean:
	$(RM) *.o *.ptx *~ $(MAIN)

format:
	find csrc -type f | xargs clang-format -i --style=LLVM

depend: $(CU_SRCS)
	$(NVCC) -M $(INCLUDES) $(CU_SRCS) >> ./Makefile.dep
	
-include Makefile.dep