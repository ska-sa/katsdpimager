<%include file="/port.mako"/>

typedef ${real_type} Real;
typedef ${real_type}2 Complex;

KERNEL REQD_WORK_GROUP_SIZE(${wgs}, 1, 1)
void complex_to_real(
    GLOBAL Real * RESTRICT dest,
    const GLOBAL Complex * RESTRICT src,
    int N)
{
    int gid = get_global_id(0);
    if (gid < N)
        dest[gid] = src[gid].x;
}
