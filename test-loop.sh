#!/bin/bash

set -xeo pipefail

cargo build --release

while true
do
    ./target/release/qdrant-catch-me-if-you-can | tee test.log
done
