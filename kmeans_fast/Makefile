NCPU=$(shell nproc)
LDFLAGS += -lm -pthread
CXXFLAGS += -g -O3 -DNR_CPU=${NCPU} -Wall -std=c++11

all: makedirs kmeans_js_32 kmeans_js_64 js_search_32 js_search_64

build32/%.o: CXXFLAGS += -DTARGET_SIZE=32
build32/%.o: %.cpp
	${CXX} -c ${CXXFLAGS} $^ -o $@

build64/%.o: CXXFLAGS += -DTARGET_SIZE=64
build64/%.o: %.cpp
	${CXX} -c ${CXXFLAGS} $^ -o $@

kmeans_js_32: build32/kmeans_js.o
	${CXX} $^ -o $@ ${LDFLAGS}

kmeans_js_64: build64/kmeans_js.o
	${CXX} $^ -o $@ ${LDFLAGS}

js_search_32: build32/js_search.o
	${CXX} $^ -o $@ ${LDFLAGS}

js_search_64: build64/js_search.o
	${CXX} $^ -o $@ ${LDFLAGS}

clean:
	rm -f *.o build32/* build64/* kmeans_js_32 kmeans_js_64 js_search_32 js_search_64

makedirs:
	mkdir -p build32 build64
