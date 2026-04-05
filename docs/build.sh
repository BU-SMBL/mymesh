#!/bin/bash
docsdir="$(dirname "$0")"
sphinx-build -M html "$docsdir"/source/ "$docsdir"/build/
