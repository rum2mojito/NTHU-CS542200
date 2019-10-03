CC = clang
CFLAGS = -fsanitize=address -g -Wall
TARGETS = oe

.MAIN: all
all: $(TARGETS)

.MAIN: clean
clean:
	rm -f $(TARGETS)

%: %.c
	$(CC) -o $@ $(CFLAGS) $^