CPP=nvcc
SRC=src/main.cc src/model.cc src/mnist.cc
BIN=main
CPPFLAGS=-std=c++14 -g -DARMA_DONT_USE_WRAPPER -lblas -llapack -fopenmp -l /opt/cuda

all:
	$(CPP) $(SRC) $(CPPFLAGS) -o $(BIN)

check: all
	./$(BIN)

clean:
	$(RM) $(BIN)

.PHONY: clean
