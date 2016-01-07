<%include file="/port.mako"/>

typedef ${real_type} Real;
typedef ${real_type}2 Complex;

KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, 1)
void fourier_beam(
    GLOBAL Complex *data,
    int stride,
    Real amplitude,
    Real a,
    Real b,
    Real c,
    int width,
    int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height)
        return;
    Real u = x;   // No reflection needed, since only half of this axis is computed
    Real v = (y * 2 >= height) ? y - height : y;
    Real power = (a * v + b * u) * v + c * u * u;
    Real ft = amplitude * exp(power);
    int addr = y * stride + x;
    Complex value = data[addr];
    value.x *= ft;
    value.y *= ft;
    data[addr] = value;
}
