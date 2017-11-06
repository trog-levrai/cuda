CPP=nvcc --std=c++11 -G
SRC=src/main.cc src/cuda_matrix.cu src/matrix_helper.cu src/model.cc
BIN=main
CPPFLAGS=-lcuda -lcublas

all:
	$(CPP) $(SRC) $(CPPFLAGS) -o $(BIN)

check: all
	./$(BIN)

clean:
	$(RM) $(BIN)

.PHONY: clean
