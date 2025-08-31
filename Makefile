CXX = nvcc

TARGET = montecarlo-option-pricing.exe

SRCS = src/main.cu src/util.cpp

OBJS = obj/main.o obj/util.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $(TARGET) $(OBJS)

obj/%.o: src/%.cpp
	$(CXX) -c $< -o $@
obj/%.o: src/%.cu
	$(CXX) -c $< -o $@

clean:
	rm -f $(TARGET) obj/*.o