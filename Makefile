CC    = gcc -O3
MPICC = mpicc -O3
DEBUG = -g -fsanitize=address -Wall -Wextra -lefence #$(" ") #
LIBS  = -lm -lblas -llapacke -llapack  

all: q3

q3: q3.c serial_gmres.o
	$(CC) -o q3 q3.c serial_gmres.o $(LIBS) $(DEBUG)

serial_gmres.o: serial_gmres.c serial_gmres.h
	$(CC) -c serial_gmres.c $(LIBS) $(DEBUG)

clean:
	rm *.o q3
