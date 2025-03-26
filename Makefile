CC    = gcc -O0
MPICC = mpicc -O0
DEBUG = $(" ") #-g -fsanitize=address -Wall -Wextra -lefence #$(" ") #
LIBS  = -lm -lblas -llapacke -llapack  

all: q3 q3_parallel

q3: q3.c serial_gmres.o
	$(CC) -o q3 q3.c serial_gmres.o $(LIBS) $(DEBUG)

q3_parallel: q3_parallel.c serial_gmres.o parallel_gmres.o
	$(CC) -o q3_parallel q3_parallel.c parallel_gmres.o $(LIBS) $(DEBUG) -fopenmp

serial_gmres.o: serial_gmres.c serial_gmres.h
	$(CC) -c serial_gmres.c $(LIBS) $(DEBUG)

parallel_gmres.o: parallel_gmres.c parallel_gmres.h
	$(CC) -c parallel_gmres.c $(LIBS) $(DEBUG) -fopenmp

clean:
	rm *.o q3
