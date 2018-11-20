# C compiler
CC = pgc++
CC_FLAGS = -w -mp -acc -ta=tesla,multicore -Minfo=accel -O3 -std=c++11 -Mcuda

CC_FLAGS += -I$(GECKO_HOME)/ -L$(GECKO_HOME)/lib 
CC_LIB = -lgecko


simpleCUBLAS: simpleCUBLAS.cpp
	$(CC) $(CC_FLAGS) simpleCUBLAS.cpp -o simpleCUBLAS $(CC_LIB)

clean:
	rm -f simpleCUBLAS 

