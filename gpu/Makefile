CPP=nvcc --std=c++11
SRC=src/main.cc src/model.cc src/mnist.cc
BIN=main
CPPFLAGS=-lcuda

all:
	$(CPP) $(SRC) $(CPPFLAGS) -o $(BIN)

check: all
	./$(BIN)

clean:
	$(RM) $(BIN)

.PHONY: clean
