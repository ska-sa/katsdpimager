## -*- coding: utf-8 -*-
<%page expression_filter="h"/><!DOCTYPE html>
<html lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap.min.css">
<title>Imaging report</title>
</head>
<body>
<div class="container-fluid">
<h1>Imaging report</h1>
<p>Revision ${revision}</p>
% for mode in modes:
% for channel in channels:
<div class="panel panel-default">
    <div class="panel-heading">${mode} — channel ${channel}</div>
    <div class="panel-body">
        <table class="table table-bordered">
            <head>
                <tr>
                    <th>Tool</th>
                    % for s in stokes:
                    <th>${s}</th>
                    % endfor
                </tr>
            </thead>
            <tbody>
                % for image in images:
                <tr>
                    <th scope="row">
                        <a href="${image.build_info_filename() | u}">
                            ${image.name | n}
                        </a>
                    </th>
                    % for s in stokes:
                    <td>
                        % if build_info[image].returncode == 0:
                        <a href="${image.svg_filename_full(mode, s, channel) | u}">
                            <img alt="Stokes ${s} for ${image.name | n} channel ${channel} (${mode})" src="${image.svg_filename_thumb(mode, s, channel) | u}"/>
                        </a>
                        % else:
                        &nbsp;
                        % endif
                    </td>
                    % endfor
                </tr>
                % endfor
            </tbody>
        </table>
    </div>
</div>
% endfor
% endfor
</body>
</div>
</html>
