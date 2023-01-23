/*
 * @author Prasanth Thomas Shaji
 *
 * Read & Write 28 x 28 PGM Images
 *
 */

#ifndef IMAGES_H
#define IMAGES_H

#include "mnist.h"

extern unsigned char test_image_char[NUM_TEST][SIZE];
extern float image[SIZE];

void write_image(char* filename, int t_index)
{
	FILE *fd;

	if ( (fd=fopen(filename, "wb")) == NULL )
	{
		printf("could not open file to write\n");
		exit(1);
	}

	fputs("P2\n", fd);
	fputs("# Created by dennis\n", fd);
	fprintf(fd, "%d %d\n", 28, 28);
	fprintf(fd, "%d\n", 255);

	for (int x=0; x<28; x++)
		for (int y=0; y<28; y++)
			fprintf(fd, "%d\n", test_image_char[t_index][x * 28 + y]);
	fclose(fd);

	printf("Image was saved successfully\n");
}

void load_infer_image(char* filename)
{
	FILE* fd;
	char buf[21];
	int val;
	int w, h, m;

	if ( (fd=fopen(filename, "r")) == NULL )
	{
		printf("could not open image file to read\n");
		exit(1);
	}

	fgets(buf, sizeof buf, fd);  // P2
	fgets(buf, sizeof buf, fd);  // Created by dennis
	fscanf(fd, "%d%d\n", &w, &h); // 28 28
	fscanf(fd, "%d", &m);         // 255

	for (int x=0; x<28; x++)
		for (int y=0; y<28; y++)
		{
			fscanf(fd, "%d", &val);
			image[x * 28 + y] = (float) val / 255.0;
		}
	fclose(fd);

}

#endif
