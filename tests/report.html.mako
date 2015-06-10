<!DOCTYPE html>
<html lang="en">
<head>
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.4/css/bootstrap.min.css">
<title>Imaging report</title>
</head>
<body>
<div class="container">
<h1>Imaging report</h1>
<p>Revision ${revision | h}</p>
<table class="table table-bordered">
    <head>
        <tr>
            <th>Tool</th>
% for s in stokes:
            <th>${s | h}</th>
% endfor
        </tr>
    </thead>
    <tbody>
% for image in images:
        <tr>
            <th scope="row">${image.name}</th>
% for s in stokes:
            <td><img src="${image.output_filename(s) | u}"/></td>
% endfor
        </tr>
    </tbody>
% endfor
</table>
</body>
</div>
</html>
