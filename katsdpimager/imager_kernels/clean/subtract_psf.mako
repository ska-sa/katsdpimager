<%include file="/port.mako"/>

#define WGSX ${wgsx}
#define WGSY ${wgsy}
#define NPOLS ${num_polarizations}
typedef ${real_type} T;

typedef struct
{
    T v[NPOLS];
} pixel;

KERNEL REQD_WORK_GROUP_SIZE(WGSX, WGSY, 1)
void subtract_psf(
    GLOBAL pixel * RESTRICT dirty,
    GLOBAL pixel * RESTRICT model,
    int image_width, int image_height, int image_stride,
    const GLOBAL pixel * RESTRICT psf,
    int psf_width, int psf_height, int psf_stride,
    const GLOBAL pixel * RESTRICT center,
    int centerx, int centery,
    int startx, int starty,
    float loop_gain)
{
    LOCAL_DECL pixel local_scale;
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    if (get_local_id(0) == 0 && get_local_id(0) == 0)
    {
        pixel scale = *center;
        for (int i = 0; i < NPOLS; i++)
            scale.v[i] *= loop_gain;
        local_scale = scale;
        if (gx == 0 && gy == 0)
        {
            int center_idx = centery * image_stride + centerx;
            pixel m = model[center_idx];
            model[center_idx] = m;
            for (int i = 0; i < NPOLS; i++)
                m.v[i] += scale.v[i];
            model[center_idx] = m;
        }
    }
    BARRIER();
    pixel scale = local_scale;

    if (gx < psf_width && gy < psf_height)
    {
        pixel p = psf[gy * psf_stride + gx];
        int x = startx + gx;
        int y = starty + gy;
        if (x >= 0 && x < image_width && y >= 0 && y < image_height)
        {
            int idx = y * image_stride + x;
            pixel d = dirty[idx];
            for (int i = 0; i < NPOLS; i++)
                d.v[i] -= scale.v[i] * p.v[i];
            dirty[idx] = d;
        }
    }
}
