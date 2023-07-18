all: build

build: lenet

readubyte.o: readubyte.h readubyte.cpp
	g++ -Wall -I/usr/local/cuda/include -o readubyte.o -c readubyte.cpp

lenet: lenet.cu readubyte.o
	/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -o lenet readubyte.o lenet.cu -I /usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudnn -lcublas

clean:
	rm -f *.o lenet
