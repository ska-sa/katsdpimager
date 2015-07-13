<%include file="/port.mako"/>
<%namespace name="taper" file="taper.mako"/>

<%call expr="taper.taper_kernel('taper_divide', '', 'const', real_type)">
    Complex value[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            value[i][j] = layer[addr[1 - i][1 - j]];

    ${taper.compute_lm()}

    // Compute and write results
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            ${taper.compute_n_rotate('i', 'j')}
            // Multiply by e^(2pi i w (n-1))
            Real rotated = value[i][j].x * rotate.x - value[i][j].y * rotate.y;
            image[addr[i][j]] += rotated * n / (kernel_y[i] * kernel_x[j]);
        }
</%call>
