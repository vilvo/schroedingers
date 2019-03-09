all: protoc build

CPPOPTS=-Wall -std=c++14

protoc:
	mkdir -p gen
	protoc -I ../proto/ --cpp_out=./gen/ ../proto/image.proto
	g++ ${CPPOPTS} `pkg-config --cflags --libs protobuf` -c ./gen/image.pb.cc -o image.pb.o

build:
	g++ ${CPPOPTS} streamer.cpp image.pb.o -lrealsense2 `pkg-config --libs --cflags opencv` `pkg-config --cflags --libs protobuf` -I./gen -o streamer

clean:
	rm -f *~ *.o streamer gen/*
	rmdir gen
