<%include file="/port.mako">
<%namespace name="taper" file="taper.mako"/>

<%call expr="taper.taper_kernel('taper_multiply', 'const', '', real_type)">
    Complex value[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            value[i][j] = in[addr[i][j]];

    ${taper.compute_lm()}

    // Compute and write results
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            ${taper.compute_n_rotate(i, j)}
            value[i][j] *= kernel_y[i] * kernel_x[j] / n;
            // Multiply by e^(-2pi i w (n-1))
            Complex rotated;
            rotated.x = value[i][j] * rotate.x;
            rotated.y = value[i][j] * -rotate.y;
            out[addr[1 - i][1 - j]] = rotated;
        }
</%call>
