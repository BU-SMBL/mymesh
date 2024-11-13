# Building the documentation

## Documentation build dependencies
```
pip install -U sphinx sphinx-design sphinx-copybutton sphinxcontrib-bibtex matplotlib pydata-sphinx-theme
```

## Build:
```
sphinx-build -M html docs/source/ docs/build/
```