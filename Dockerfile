FROM java:8
VOLUME /tmp
EXPOSE 8085
ADD /build/libs/wekaservice.jar wekaservice.jar
ENTRYPOINT ["java","-jar","wekaservice.jar","--server.port=8085"]
