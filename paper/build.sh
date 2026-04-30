#!/bin/bash
docker run --rm  --volume $(dirname "$0"):/data --user $(id -u):$(id -g) --env JOURNAL=joss openjournals/inara