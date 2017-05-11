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
    GLOBAL T * RESTRICT dirty,
    GLOBAL T * RESTRICT model,
    int image_row_stride, int image_pol_stride,
    int image_width, int image_height,
    const GLOBAL T * RESTRICT psf,
    int psf_row_stride, int psf_pol_stride,
    int psf_width, int psf_height,
    int psf_addr_offset,
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
            int center_idx = centery * image_row_stride + centerx;
            for (int i = 0; i < NPOLS; i++)
                model[i * image_pol_stride + center_idx] += scale.v[i];
        }
    }
    BARRIER();
    pixel scale = local_scale;

    if (gx < psf_width && gy < psf_height)
    {
        int x = startx + gx;
        int y = starty + gy;
        if (x >= 0 && x < image_width && y >= 0 && y < image_height)
        {
            int psf_addr = gy * psf_row_stride + gx + psf_addr_offset;
            int image_addr = y * image_row_stride + x;
            for (int i = 0; i < NPOLS; i++)
            {
                T p = psf[i * psf_pol_stride + psf_addr];
                dirty[i * image_pol_stride + image_addr] -= scale.v[i] * p;
            }
        }
    }
}
