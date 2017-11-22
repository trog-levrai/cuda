CPP=nvcc --std=c++11 -G -g -lineinfo -Wno-deprecated-gpu-targets
SRC=src/main.cc src/cuda_matrix.cu src/matrix_helper.cu src/model.cc
UNIT=src/unit_test.cc src/cuda_matrix.cu src/matrix_helper.cu src/model.cc
BIN=main
UBIN=unit
CPPFLAGS=-lcuda -lcublas

all:
	$(CPP) $(SRC) $(CPPFLAGS) -o $(BIN)

check: all
	./$(BIN)

unit:
	$(CPP) $(UNIT) $(CPPFLAGS) -o $(UBIN)
	./$(UBIN)

clean:
	$(RM) $(BIN) $(UBIN)

.PHONY: clean
