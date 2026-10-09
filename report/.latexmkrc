# Compilación con LuaLaTeX y biber; los intermedios van a build/ y el PDF se copia a dist/.
$pdf_mode = 4;
$lualatex = 'lualatex -interaction=nonstopmode -halt-on-error -file-line-error %O %S';
$out_dir = 'build';
$bibtex_use = 2;
$success_cmd = 'cp build/informe.pdf dist/informe.pdf';
