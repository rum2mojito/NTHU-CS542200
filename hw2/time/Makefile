CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread
test: CFLAGS += -pthread -fsanitize=address -g -Wall
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
