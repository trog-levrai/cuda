CPP=nvcc -Xptxas -O3 --std=c++11 --compiler-options "-O3" -Wno-deprecated-gpu-targets
#CPP=nvcc -G --std=c++11 --compiler-options "-g" -Wno-deprecated-gpu-targets
SRC=src/main.cc src/cuda_matrix.cu src/matrix_helper.cu src/model.cc src/mnist.cc
UNIT=src/unit_test.cc src/cuda_matrix.cu src/matrix_helper.cu src/model.cc
BIN=main
UBIN=unit
CPPFLAGS=-lcuda -lcublas

all:
	$(CPP) $(SRC) $(CPPFLAGS) -o $(BIN)

check: all
	./$(BIN)

unit: clean
	$(CPP) $(UNIT) $(CPPFLAGS) -o $(UBIN)
	./$(UBIN)

clean:
	$(RM) $(BIN) $(UBIN)

.PHONY: clean
