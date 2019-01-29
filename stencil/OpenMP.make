# C compiler

# Borrowed following for BabelStream Benchmark

DATA_TYPE=USE_DOUBLE

ifndef COMPILER
define compiler_help
Set COMPILER to ensure correct flags are set.
Available compilers are:
  PGI GNU
endef
$(info $(compiler_help))
endif

COMPILER_ = $(CXX)
COMPILER_PGI = pgc++
COMPILER_GNU = g++

FLAGS_GNU =-std=c++11 -O3 -fopenmp 


FLAGS_PGI = -Mllvm -std=c++11 -O3 -mp 
ifeq ($(COMPILER), PGI)
define target_help
Set a TARGET to ensure PGI targets the correct offload device.
Available targets are:
  SNB, IVB, HSW
endef
ifndef TARGET
$(error $(target_help))
endif
TARGET_FLAGS_SNB     = -tp=sandybridge
TARGET_FLAGS_IVB     = -tp=ivybridge
TARGET_FLAGS_HSW     = -tp=haswell
ifeq ($(TARGET_FLAGS_$(TARGET)),)
$(error $(target_help))
endif

FLAGS_PGI += $(TARGET_FLAGS_$(TARGET))

endif


CXXFLAGS = -D$(DATA_TYPE) $(FLAGS_$(COMPILER))


omp-stencil: omp-stencil.cpp
	$(COMPILER_$(COMPILER)) $(CXXFLAGS) $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f omp-stencil *.o



