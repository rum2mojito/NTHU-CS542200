CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread
hw2a_1: CFLAGS += -pthread -fsanitize=address -g -Wall
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp
test: CFLAGS += -pthread
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b hw2a_1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
