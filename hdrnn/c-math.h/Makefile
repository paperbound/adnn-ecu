LDLIBS := -lm
CFLAGS := -std=c99 -Wall

all: CFLAGS += -O2
all: hdrnn

debug: CFLAGS += -O0 -g
debug: hdrnn

hdrnn:
	$(CC) $(CFLAGS) src/main.c  -o bin/hdrnn $(LDLIBS)

image-gen:
	$(CC) src/example.c -o bin/genimage

sample-image: image-gen
	./bin/genimage 1 images/1.pgm

clean:
	rm -rf bin/*
