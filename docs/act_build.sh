#!/bin/bash
docsdir="$(dirname "$0")"
sphinx-build -M html "$docsdir"/source/ "$docsdir"/build/
act pull_request -W "$docsdir"/../.github/workflows/documentation.yml --bind