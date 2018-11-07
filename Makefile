IMAGE=tyz910/classicai

run:
	docker run --rm -it -v ${CURDIR}:/app -w /app -p 8000:8000 ${IMAGE} python3 server.py

docker-build:
	docker build -t ${IMAGE} . && (docker ps -q -f status=exited | xargs docker rm) && (docker images -qf dangling=true | xargs docker rmi) && docker images

docker-push:
	docker push ${IMAGE}
