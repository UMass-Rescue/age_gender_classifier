#! /usr/bin/env bash

set -a; source .env; set +a

mkdir -p ./mounted_volume
mkdir -p ./input_data

docker run -d \
    --name cs596 \
    --rm \
    -e POSTGRES_USER=$PG_USER \
    -e POSTGRES_PASSWORD=$PG_PASSWORD \
    -e POSTGRES_DB=$PG_DB \
    -e PGDATA=/var/lib/postgresql/data \
    -v ./mounted_volume:/var/lib/postgresql/data \
    -v ./input_data:/tmp/postgres \
    -h $PG_HOST \
    -p $PG_PORT:5432 \
    postgres:16.2
