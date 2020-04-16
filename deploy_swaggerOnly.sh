docker-compose -f "docker-compose_swaggerOnly.yaml" down
rm -rf ./db/*
CURRENT_UID=$(id -u):$(id -g) docker-compose -f "docker-compose_swaggerOnly.yaml" up --abort-on-container-exit --force-recreate --build
