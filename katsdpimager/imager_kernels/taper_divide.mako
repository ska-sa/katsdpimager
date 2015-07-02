<%include file="/port.mako"/>

typedef ${real_type} Real;
typedef ${real_type}2 Complex;

/**
 * @param out       Output real image, with strides @a x_stride, @a y_stride and image centre in the middle
 * @param in        Input complex image, with stride @a x_stride, @a y_stride and image centre at the corners
 * @param x_stride, y_stride  Strides for @a out and @a in
 * @param kernel1d, lm_scale, lm_bias  See Python documentation
 * @param half_size Half the width/height of @a in and @a out images
 * @param x_offset  Product of @a half_size and @a x_stride
 * @param y_offset  Product of @a half_size and @a y_stride
 * @param lm_offset Product of @a half_size and @a lm_scale
 * @param w2        Twice the central W value for the slice
 */
KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, 1)
void taper_divide(
    GLOBAL Real * RESTRICT out,
    const GLOBAL Complex * RESTRICT in,
    int row_stride,
    int slice_stride,
    const GLOBAL Real * RESTRICT kernel1d,
    Real lm_scale,
    Real lm_bias,
    int half_size,
    int y_offset,
    Real lm_offset,
    Real w2)
{
    int x[2], y[2];
    x[0] = get_global_id(0);
    y[0] = get_global_id(1);
    int pol = get_global_id(2);
    if (x[0] < half_size && y[0] < half_size)
    {
        Real kernel_x[2], kernel_y[2], l2[2], m2[2];
        Complex value[2][2];
        int addr[2][2];
        x[1] = x[0] + half_size;
        y[1] = y[0] + half_size;
        // Load kernel samples
        for (int i = 0; i < 2; i++)
        {
            kernel_x[i] = kernel1d[x[i]];
            kernel_y[i] = kernel1d[y[i]];
        }
        // Load data, applying fftshift
        addr[0][0] = pol * slice_stride + y[0] * row_stride + x[0];
        addr[0][1] = addr[0][0] + half_size;
        addr[1][0] = addr[0][0] + y_offset;
        addr[1][1] = addr[0][1] + y_offset;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                value[i][j] = in[addr[1 - i][1 - j]];

        // Compute l^2, m^2 (first compute l and m)
        l2[0] = x[0] * lm_scale + lm_bias;
        m2[0] = y[0] * lm_scale + lm_bias;
        l2[1] = l2[0] + lm_offset;
        m2[1] = m2[0] + lm_offset;
        for (int i = 0; i < 2; i++)
        {
            l2[i] *= l2[i];
            m2[i] *= m2[i];
        }

        // Compute and write results
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                Real n = sqrt(1 - m2[i] - l2[j]);
                Real c, s;
                // TODO: add sincospi wrapper for OpenCL
                sincospi(w2 * (n - 1), &s, &c);
                // Multiply by e^(2pi i w (n-1))
                Real rotated = value[i][j].x * c - value[i][j].y * s;
                out[addr[i][j]] = rotated * n / (kernel_y[i] * kernel_x[j]);
            }
    }
}
