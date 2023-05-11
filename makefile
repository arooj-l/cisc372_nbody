CC = nvcc
CFLAGS = -O3 -arch=sm_35
LIBS = -lm

SRCS = body.cu compute.cu
OBJS = $(SRCS:.cu=.o)

.PHONY: all clean

all: nbody

nbody: $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

%.o: %.cu compute.h config.h planets.h vector.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(OBJS) nbody
