/*
 * Company:	Systhesis
 * Author: 	Chen
 * Date:	2018/06/07
 */
#ifndef __IMAGE_H_
#define __IMAGE_H_

typedef struct
{
    int w;
    int h;
    int c;
    float *data;
}image;

image load_image_color(char* filename,int w,int h);

void free_image(image m);

image letterbox_image(image im, int w, int h);

float get_pixel(image m, int x, int y, int c);
void set_pixel(image m, int x, int y, int c, float val);
void add_pixel(image m, int x, int y, int c, float val);

//image make_image(int w, int h, int c);

image resize_image(image im, int w, int h);

//void fill_image(image m, float s);

#endif
