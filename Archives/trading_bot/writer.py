# Convertit tous les .py en UTF-8
import glob
import codecs

for filename in glob.glob('src/**/*.py', recursive=True):
    with codecs.open(filename, 'r', encoding='latin1') as f:
        content = f.read()
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
