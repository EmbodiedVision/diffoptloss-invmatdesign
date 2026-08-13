CC      = c++
CFLAGS  = -Wall -O3 -march=native -std=c++17
COPT	= -DMKL_ILP64  -I"${CONDA_PREFIX}/include"
LIBS	= -L${CONDA_PREFIX}/lib -Wl,--start-group ${CONDA_PREFIX}/lib/libmkl_intel_ilp64.a ${CONDA_PREFIX}/lib/libmkl_intel_thread.a ${CONDA_PREFIX}/lib/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl -lz

ODIR    = /OBJ
CFILES  := $(wildcard SRC/*.cpp) cnpy/cnpy.cpp
OFILES  := $(addprefix OBJ/,$(notdir $(CFILES:.cpp=.o)))

fem3D:  $(OFILES)
	$(CC) $(CFLAGS) -o $@ $(OFILES) $(LIBS)
OBJ/%.o: SRC/%.cpp
	$(CC) $(CFLAGS) -c $(COPT) $< -o $@
OBJ/cnpy.o: cnpy/cnpy.cpp
	$(CC) $(CFLAGS) -c $(COPT) $< -o $@

fem3D_cluster: $(OFILES)
	$(CC) $(CFLAGS) -o $@ $(OFILES) $(LIBS)
	
clean:
	rm -f OBJ/*.o

