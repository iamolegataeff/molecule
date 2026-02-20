//go:build cgo && (darwin || linux)

package main

/*
#cgo darwin CFLAGS: -DACCELERATE_NEW_LAPACK
#cgo darwin LDFLAGS: -framework Accelerate
#cgo linux CFLAGS: -I/usr/include/openblas
#cgo linux LDFLAGS: -lopenblas

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

static void mol_dgemv(int M, int N, const double *A, const double *x, double *y) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0, A, N, x, 1, 0.0, y, 1);
}
*/
import "C"
import "unsafe"

const hasBLAS = true

func blasDgemv(M, N int, A, x, y []float64) {
	C.mol_dgemv(C.int(M), C.int(N),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&y[0])))
}
