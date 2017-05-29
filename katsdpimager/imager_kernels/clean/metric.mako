DEVICE_FN ${real_type} clean_metric(const GLOBAL ${real_type} * RESTRICT image,
                                    int pixel_offset, int pol_stride)
{
    ${real_type} pix = image[pixel_offset];
% if clean_mode == 1:
## CLEAN_SUMSQ
    ${real_type} value = pix * pix;
    for (int i = 1; i < ${num_polarizations}; i++)
    {
        pix = image[i * pol_stride + pixel_offset];
        value = fma(pix, pix, value);
    }
    return value;
% else:
    return fabs(pix);
% endif
}
