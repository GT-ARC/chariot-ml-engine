docker-compose -f "docker-compose.yaml" down
rm -rf ./db/*
CURRENT_UID=$(id -u):$(id -g) docker-compose -f "docker-compose.yaml" up --abort-on-container-exit --force-recreate --build
