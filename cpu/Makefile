CPP=g++
SRC=src/main.cc
BIN=main
CPPFLAGS=-llapack

all:
	$(CPP) $(SRC) $(CPPFLAGS) -o $(BIN)

check: all
	./$(BIN)

clean:
	$(RM) $(BIN)
