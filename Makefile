BUILDDIR=bin
TARGETS=$(BUILDDIR)/summa $(BUILDDIR)/summatau

.phony: all
all: $(TARGETS)

$(BUILDDIR)/summa: summa.cpp
	mkdir -p $(BUILDDIR)
	mpic++ summa.cpp -o bin/summa

$(BUILDDIR)/summatau: summa.cpp
	mkdir -p $(BUILDDIR)
	tau_cxx.sh summa.cpp -o bin/summatau

clean: 
	-rm bin/*
	-rm *.trc
	-rm profile*
	-rm *.edf
	-rm summaprof
	-rm summatrace
	-rm summa.o
	-rm summa.slog2
