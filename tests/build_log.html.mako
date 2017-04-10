<%page expression_filter="h"/><!DOCTYPE html>
<html lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap.min.css">
<title>Build log for ${image.name | n}</title>
</head>
<body>
<div class="container-fluid">
<h1>Build log for ${image.name | n}</h1>
<h3>Commands</h3>
<pre>
% for cmd in build_info.cmds:
${' '.join(cmd)}
% endfor
</pre>
% if build_info.returncode != 0:
<p class="text-warning">Command failed with exit code ${build_info.returncode}.</p>
% endif
<p class="text-muted">
Elapsed time: ${"%.1f" % build_info.elapsed} seconds
</p>

<h3>Log</h3>
% if build_info.output != '' and build_info.output != '\n':
<pre>${build_info.output}</pre>
% else:
<p class="text-muted">No output</p>
% endif

% if build_info.returncode == 0:
<h3>Output files</h3>
<ul>
    <%
    filenames = []
    for mode in modes:
        for s in stokes:
            for channel in channels:
                filename = image.fits_filename(mode, s, channel, channel - channels[0])
                if filename not in filenames:
                    filenames.append(filename)
    %>
    % for filename in filenames:
    <li><a href="${filename | u}">${filename}</a></li>
    % endfor
</ul>
% endif
</div>
</body>
</html>
