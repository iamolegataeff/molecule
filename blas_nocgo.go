//go:build !(cgo && (darwin || linux))

package main

const hasBLAS = false

func blasDgemv(M, N int, A, x, y []float64) {}
