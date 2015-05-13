<%include file="/port.mako"/>

typedef ${ctype} T;

KERNEL REQD_WORK_GROUP_SIZE(${wgsx}, ${wgsy}, 1)
void fftshift(GLOBAL T *data, int stride, int halfx, int halfy, int yoffset)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    if (gx < halfx && gy < halfy)
    {
        int addr00 = gy * stride + gx;
        T value00 = data[addr00];
        int addr01 = addr00 + halfx;
        T value01 = data[addr01];
        int addr10 = addr00 + yoffset;
        T value10 = data[addr10];
        int addr11 = addr10 + halfx;
        T value11 = data[addr11];

        data[addr11] = value00;
        data[addr10] = value01;
        data[addr01] = value10;
        data[addr00] = value11;
    }
}
