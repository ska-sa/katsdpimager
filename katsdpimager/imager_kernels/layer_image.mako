## Helper functions for layer_to_image and image_to_layer

<%def name="compute_lm()">
    Real l2[2], m2[2];
    l2[0] = x[0] * lm_scale + lm_bias;
    m2[0] = y[0] * lm_scale + lm_bias;
    l2[1] = l2[0] + lm_offset;
    m2[1] = m2[0] + lm_offset;
    for (int i = 0; i < 2; i++)
    {
        l2[i] *= l2[i];
        m2[i] *= m2[i];
    }
</%def>

<%def name="compute_n_rotate(i, j)">
    Real n = sqrt(1 - m2[${i}] - l2[${j}]);
    Complex rotate;
#ifdef CUDA
    sincospi(w2 * (n - 1), &rotate.y, &rotate.x);
#else
    rotate.x = cospi(w2 * (n - 1));
    rotate.y = sinpi(w2 * (n - 1));
#endif
</%def>

<%def name="kernel(kernel_name, image_const, layer_const, real_type)">
    typedef ${real_type} Real;
    typedef ${real_type}2 Complex;

    /**
     * @param image     Real image, with stride @a stride and image centre in the middle
     * @param layer     Complex layer, with stride @a x_stride, @a y_stride and image centre at the corners
     * @param image_start Pixel offset to the start of the image layer
     * @param stride    Row stride for @a image and @a layer
     * @param kernel1d, lm_scale, lm_bias  See Python documentation
     * @param half_size Half the width/height of @a layer and @a image
     * @param y_offset  Product of @a half_size and @a stride
     * @param lm_offset Product of @a half_size and @a lm_scale
     * @param w2        Twice the central W value for the slice
     */
    KERNEL REQD_WORK_GROUP_SIZE(${wgs_x}, ${wgs_y}, 1)
    void ${kernel_name}(
        ${image_const} GLOBAL Real * RESTRICT image,
        ${layer_const} GLOBAL Complex * RESTRICT layer,
        int image_start,
        int stride,
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
        if (x[0] < half_size && y[0] < half_size)
        {
            Real kernel_x[2], kernel_y[2];
            int addr[2][2];

            x[1] = x[0] + half_size;
            y[1] = y[0] + half_size;
            for (int i = 0; i < 2; i++)
            {
                kernel_x[i] = kernel1d[x[i]];
                kernel_y[i] = kernel1d[y[i]];
            }
            addr[0][0] = y[0] * stride + x[0];
            addr[0][1] = addr[0][0] + half_size;
            addr[1][0] = addr[0][0] + y_offset;
            addr[1][1] = addr[0][1] + y_offset;

            ${caller.body()}
        }
    }
</%def>
