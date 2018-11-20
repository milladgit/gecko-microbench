# C compiler
CC = pgc++
CC_FLAGS = -w -mp -acc -ta=tesla,multicore -Minfo=accel -O3 -std=c++11 -Mcuda

CC_FLAGS += -I$(GECKO_HOME)/ -L$(GECKO_HOME)/lib
CC_LIB = -lgecko


vector_add: vector_add.cpp
	$(CC) $(CC_FLAGS) vector_add.cpp -o vector_add $(CC_LIB)

clean:
	rm -f vector_add 

