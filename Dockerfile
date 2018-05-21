FROM java:8
VOLUME /tmp
EXPOSE 8080
ADD /build/libs/wekaservice.jar wekaservice.jar
ENTRYPOINT ["java","-jar","wekaservice.jar"]
