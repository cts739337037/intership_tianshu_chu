# Disclaimer
This is an unofficial LaTeX template for KCL MSc Projects for the Informatics Department. 

How ever it has been approved by the supervisor for the MSc Projects for 2015/16 of the Informatics Department Hak-Keung Lam (hak-keung.lam@kcl.ac.uk).

It is derived from the Imperial College London Thesis template (which can be found [here](https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/public/study/admissions/pg/msc/math-fin/MScThesisTemplate.zip)).

Unfortunately it has no Licence given, therefore I publish this under the MIT License.

Changes have been made by Sebastian Zillessen (sebastian.zillessen@kcl.ac.uk) to adapt it to KCL.

The front page and the bibliography style of King's Harvard V1 have been adapted from the official resources of KCL which were ported to LaTex by [Andre Müller](https://github.com/mueller-andre) (andre.mueller@kcl.ac.uk). Many thanks!

Feel free to modify it or adapt it to your requirements, just make sure that you share your wisdom with everyone else by submitting a merge request.

I do not give any guarantee on this template that it fullfils the KCL regulations or any other regulations.


## Compilation

To compile the checked out template please run:

````
xelatex -file-line-error -interaction=nonstopmode Thesis.tex
bibtex Thesis
makeindex Thesis.acn 
makeindex Thesis.glo 
xelatex Thesis.tex
````

in your terminal. Alternatively (and recommended) use Texpad (https://www.texpadapp.com/) to open the project, it will recognise your required typesetting commands by default!

## How to contribute

Feel free to use the layout for your own projects. If you're making changes that could be useful to other users as well, 
please contribute by creating a pull request and state what you changed. 
The repository can be found here: https://github.com/sebastianzillessen/KCLInformaticsMScProjectLayout
