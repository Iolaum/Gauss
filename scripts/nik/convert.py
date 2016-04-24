import nbformat
import codecs
import argparse

from nbformat.v4 import new_code_cell, new_notebook

parser = argparse.ArgumentParser(
    description='Give input a .py file and convert it to ipynb file(Jupyter notebook file)'
)
parser.add_argument('-i', '--input', help='Input python file name', required=True)
parser.add_argument('-o', '--output', help='Output ipynb file name', required=True)
args = parser.parse_args()

## show values ##
print ("Input Python file: %s" % args.input)
print ("Output Jupyter notebook file: %s" % args.output)


sourceFile = args.input
destFile = args.output


def parsePy(fn):
    """ Generator that parses a .py file exported from a IPython notebook and
extracts code cells (whatever is between occurrences of "In[*]:").
Returns a string containing one or more lines
"""
    with open(fn, "r") as py_file:
        lines = []
        for l in py_file:
            l1 = l.strip()
            if l1.startswith('# In[') and l1.endswith(']:') and lines:
                yield "".join(lines)
                lines = []
                continue
            lines.append(l)
        if lines:
            yield "".join(lines)


# Create the code cells by parsing the file in input
cells = []
for c in parsePy(sourceFile):
    cells.append(new_code_cell(source=c))

# This creates a V4 Notebook with the code cells extracted above
nb0 = new_notebook(cells=cells,
                   metadata={'language': 'python',})

with codecs.open(destFile, encoding='utf-8', mode='w') as ipynb_file:
    nbformat.write(nb0, ipynb_file, 4)
