CXX := g++
NVCC = "$(shell which nvcc)"
NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' |  sed 's/,.*//'))

GEN_SM86 	= -gencode=arch=compute_86,code=\"sm_86,compute_86\" # Ampere RTX30XX
GEN_SM80 	= -gencode=arch=compute_80,code=\"sm_80,compute_80\" # Ampere A100
GEN_SM75 	= -gencode=arch=compute_75,code=\"sm_75,compute_75\" # Turing RTX20XX
GEN_SM70 	= -gencode=arch=compute_70,code=\"sm_70,compute_70\" # Volta V100
GEN_SM61 	= -gencode=arch=compute_61,code=\"sm_61,compute_61\" # Pascal GTX10XX
GEN_SM60 	= -gencode=arch=compute_60,code=\"sm_60,compute_60\" # Pascal P100

# Add your own SM target (default: GTX10XX):
SM_TARGETS 	= $(GEN_SM61)

CXXFLAGS    += -std=c++17
CXXFLAGS    += -Wall
CXXFLAGS    += -Wno-unused-local-typedefs -Wno-strict-aliasing -Wno-unused-function -Wno-format-security

NVCCFLAGS   += -std=c++17
NVCCFLAGS   += $(SM_TARGETS)
NVCCFLAGS   += --expt-extended-lambda --expt-relaxed-constexpr --use_fast_math --ptxas-options -v --relocatable-device-code true  -lnvToolsExt

# Optimization Flags
NVCCOPT     = -O3 --generate-line-info
CXXOPT      = -O3

# Debug Flags
NVCCDEBUG   = -O0 --debug --device-debug
CXXDEBUG    = -O0 -g

## Includes & Sources
CUDA_PATH	= "$(shell dirname $(NVCC))/.."
CUDA		= "$(CUDA_PATH)/include"
INC 		= -I./../include -I$(CUDA)

SOURCES 	= $(shell find . -name "*.cu")
EXE 		= $(patsubst %.cu,%,$(SOURCES))

all: $(EXE)
% : %.cu
	mkdir -p bin
	$(NVCC) -ccbin=$(CXX) $(NVCCFLAGS) ${NVCCOPT} --compiler-options "${CXXFLAGS} ${CXXOPT}" -o bin/$@ $< $(INC)

clean:
	rm -rf bin
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o