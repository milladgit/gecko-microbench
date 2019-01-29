# C compiler

# Borrowed following for BabelStream Benchmark

COMPILER=PGI
COMPILER_PGI = pgc++

DATA_TYPE=USE_DOUBLE

FLAGS_PGI = -Mllvm -std=c++11 -O3 -mp 
ifeq ($(COMPILER), PGI)
define target_help
Set a TARGET to ensure PGI targets the correct offload device.
Available targets are:
  SNB, IVB, HSW
  KEPLER, MAXWELL, PASCAL, VOLTA
  HAWAII
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


omp-gemm: omp-gemm.cpp
	$(COMPILER_$(COMPILER)) $(CXXFLAGS) $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f omp-gemm *.o



