<%include file="/port.mako"/>

typedef ${ctype} T;

KERNEL
void fftshift(GLOBAL T *data, int row_stride, int slice_stride, int half_x, int half_y, int y_offset)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    data += get_group_id(2) * slice_stride;
    if (gx < half_x && gy < half_y)
    {
        int addr00 = gy * row_stride + gx;
        T value00 = data[addr00];
        int addr01 = addr00 + half_x;
        T value01 = data[addr01];
        int addr10 = addr00 + y_offset;
        T value10 = data[addr10];
        int addr11 = addr10 + half_x;
        T value11 = data[addr11];

        data[addr11] = value00;
        data[addr10] = value01;
        data[addr01] = value10;
        data[addr00] = value11;
    }
}
