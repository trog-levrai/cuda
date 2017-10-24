CPP=g++
SRC=src/main.cc
BIN=main
CPPFLAGS=-DARMA_DONT_USE_WRAPPER -lopenblas -llapack

all:
	$(CPP) $(SRC) $(CPPFLAGS) -o $(BIN)

check: all
	./$(BIN)

clean:
	$(RM) $(BIN)
