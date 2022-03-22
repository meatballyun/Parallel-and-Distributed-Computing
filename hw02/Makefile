all: main.exe 

CC=g++
CFLAGS=-Wall -O2 -march=native

%.o: src/%.cpp inc/%.hpp
	$(CC) -c $(CFLAGS) $< -Iinc -o $@

main.exe: src/main.cpp ANN.o MNIST.o
	$(CC) $(CFLAGS) $< -o $@ ANN.o MNIST.o

test: main.exe
	valgrind ./$< verify

mnist: main.exe
	valgrind ./$< mnist

clean: 
	rm *o *exe