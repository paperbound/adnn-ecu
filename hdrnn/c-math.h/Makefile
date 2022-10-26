LDLIBS := -lm
CFLAGS := -g -std=c99 -Wall

all: hdrnn

hdrnn:
	$(CC) $(CFLAGS) src/main.c  -o bin/hdrnn $(LDLIBS)

image-gen:
	$(CC) src/example.c -o bin/genimage

sample-image: image-gen
	./bin/genimage 1 tests/image1.pgm

clean:
	rm -rf bin/*
