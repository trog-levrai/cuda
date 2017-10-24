CPP=g++
SRC=src/main.cc src/model.cc
BIN=main
CPPFLAGS=-std=c++14 -DARMA_DONT_USE_WRAPPER -lblas -llapack

all:
	$(CPP) $(SRC) $(CPPFLAGS) -o $(BIN)

check: all
	./$(BIN)

clean:
	$(RM) $(BIN)
