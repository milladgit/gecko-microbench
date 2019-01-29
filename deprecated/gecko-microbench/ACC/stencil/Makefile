# C compiler
CC = pgc++
CC_FLAGS = -w -mp -acc -ta=tesla:managed,multicore -Minfo=accel -O3 -std=c++11 -Mcuda

#CC_FLAGS += -I$(GECKO_HOME)/ -L$(GECKO_HOME)/lib
#CC_LIB = -lgecko


stencil: stencil.cpp
	$(CC) $(CC_FLAGS) stencil.cpp -o stencil $(CC_LIB)

clean:
	rm -f stencil 

