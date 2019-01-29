# C compiler

# Borrowed following for BabelStream Benchmark

DATA_TYPE=USE_DOUBLE

COMPILER=PGI
COMPILER_PGI = pgc++

FLAGS_PGI =-Mllvm -std=c++11 -O3 -mp -acc
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
TARGET_FLAGS_SNB     = -ta=multicore -tp=sandybridge
TARGET_FLAGS_IVB     = -ta=multicore -tp=ivybridge
TARGET_FLAGS_HSW     = -ta=multicore -tp=haswell
TARGET_FLAGS_KEPLER  = -ta=nvidia:cc35
TARGET_FLAGS_MAXWELL = -ta=nvidia:cc50
TARGET_FLAGS_PASCAL  = -ta=nvidia:cc60
TARGET_FLAGS_VOLTA   = -ta=nvidia:cc70
TARGET_FLAGS_HAWAII  = -ta=radeon:hawaii
ifeq ($(TARGET_FLAGS_$(TARGET)),)
$(error $(target_help))
endif

FLAGS_PGI += $(TARGET_FLAGS_$(TARGET))

endif


CXXFLAGS = -D$(DATA_TYPE) $(FLAGS_$(COMPILER))


acc-stencil: acc-stencil.cpp
	$(COMPILER_$(COMPILER)) $(CXXFLAGS) $^ $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f acc-stencil *.o



