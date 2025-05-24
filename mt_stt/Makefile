
# RhinoDevel, Marcel Timm, 2025mar19

CXX = g++
CXXFLAGS = -Wall -O2 -std=c++17 -fPIC -DNDEBUG

WHISPER_DIR = ./whisper.cpp
WHISPER_LIB_DIRS = -L$(WHISPER_DIR)/build/src
WHISPER_LIBS = -lwhisper
WHISPER_INCLUDES = -I$(WHISPER_DIR)/include -I$(WHISPER_DIR)/ggml/include

SRC = $(filter-out whisper.cpp, $(wildcard *.cpp))
OBJ = $(SRC:.cpp=.o)
LIBRARY = libmtstt.so

$(LIBRARY): $(OBJ)
	$(CXX) -shared -o $@ $^ $(WHISPER_LIB_DIRS) $(WHISPER_LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(WHISPER_INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJ) $(LIBRARY)

.PHONY: clean
