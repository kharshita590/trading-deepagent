#!/bin/bash
docker-compose up -d --build
docker-compose exec api poetry install
