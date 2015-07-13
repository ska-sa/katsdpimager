## Helper functions for taper_divide and taper_multiply

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
    // TODO: add sincospi wrapper for OpenCL
    sincospi(w2 * (n - 1), &rotate.y, &rotate.x);
</%def>

<%def name="taper_kernel()">
    int x[2], y[2];
    x[0] = get_global_id(0);
    y[0] = get_global_id(1);
    int pol = get_global_id(2);
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
        addr[0][0] = pol * slice_stride + y[0] * row_stride + x[0];
        addr[0][1] = addr[0][0] + half_size;
        addr[1][0] = addr[0][0] + y_offset;
        addr[1][1] = addr[0][1] + y_offset;

        ${caller.body()}
    }
</%def>
