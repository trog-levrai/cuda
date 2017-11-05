CPP=nvcc --std=c++11
SRC=src/main.cc src/cuda_matrix.cu src/matrix_helper.cu
BIN=main
CPPFLAGS=-lcuda

all:
	$(CPP) $(SRC) $(CPPFLAGS) -o $(BIN)

check: all
	./$(BIN)

clean:
	$(RM) $(BIN)

.PHONY: clean
