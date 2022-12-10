#!/bin/bash

export TAU_MAKEFILE=/home/braet/tau-2.32/x86_64/lib/Makefile.tau-mpi-pdt-profile-trace
tau_cxx.sh summa.cpp -o summatrace >/dev/null 2>&1

export TAU_MAKEFILE=/home/braet/tau-2.32/x86_64/lib/Makefile.tau-mpi-pdt
tau_cxx.sh summa.cpp -o summaprof >/dev/null 2>&1

echo "##### TRACE #####"
mpirun --oversubscribe -n 100 ./summatrace 1000 10
echo "#### PROFILE ####"
mpirun --oversubscribe -n 100 ./summaprof 1000 10

tau_merge tautrace* summa.trc >/dev/null 2>&1
tau2slog2 summa.trc tau.edf -o summa.slog2 >/dev/null 2>&1 /dev/null

