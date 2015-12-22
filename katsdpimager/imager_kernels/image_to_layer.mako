<%include file="/port.mako"/>
<%namespace name="layer_image" file="layer_image.mako"/>

<%call expr="layer_image.kernel('image_to_layer', 'const', '', real_type)">
    Real value[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            value[i][j] = image[image_start + addr[i][j]];

    ${layer_image.compute_lm()}

    // Compute and write results
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            ${layer_image.compute_n_rotate('i', 'j')}
            value[i][j] /= kernel_y[i] * kernel_x[j] * n;
            // Multiply by e^(-2pi i w (n-1))
            Complex rotated;
            rotated.x = value[i][j] * rotate.x;
            rotated.y = value[i][j] * -rotate.y;
            layer[addr[1 - i][1 - j]] = rotated;
        }
</%call>
