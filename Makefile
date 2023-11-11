CXX = g++
INCLUDES = -I./csrc -I./include
CXXFLAGS = -Wall -std=c++11
LIBS = 
SRCS = $(wildcard csrc/*.cpp)
OBJS = $(patsubst csrc/%.cpp,%.o,$(SRCS))
MAIN = main

.PHONY: depend clean

$(MAIN): $(OBJS) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LIBS)

%.o: csrc/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
	$(CXX) -MM $(INCLUDES) $^ > ./Makefile.dep
	
-include Makefile.dep
