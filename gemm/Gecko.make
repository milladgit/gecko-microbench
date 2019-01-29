# C compiler

DATA_TYPE=USE_DOUBLE

CC = pgc++
CC_FLAGS =-D$(DATA_TYPE) -m64 -std=c++11 -w -Mllvm -mp -acc -ta=tesla,multicore -Minfo=accel -O3 -std=c++11 -Mcuda 

#CC_FLAGS += -O0 -g

CC_FLAGS += -I$(GECKO_HOME)/ -L$(GECKO_HOME)/lib 
CC_LIB = -lgecko -lm


gecko-gemm: gecko-gemm.cpp
	$(CC) $(CC_FLAGS) $^ -o $@ $(CC_LIB)

clean:
	rm -f gecko-gemm *.o

