all: mjpegwriter build

CPPOPTS=-Wall -std=c++14
#MJPEGW-DIR=MJPEGWriter

mjpegwriter:
	@if [ ! -d MJPEGWriter ] ;\
	then \
	  git clone https://github.com/JPery/MJPEGWriter.git; \
	fi
	g++ ${CPPOPTS} -c ./MJPEGWriter/MJPEGWriter.cpp -o MJPEGWriter.o

build:
	g++ ${CPPOPTS} streamer.cpp MJPEGWriter.o -lrealsense2 `pkg-config --libs --cflags opencv` -lpthread -I./MJPEGWriter -I./gen -o streamer

clean:
	rm -f *~ *.o streamer
	rm -rf MJPEGWriter
