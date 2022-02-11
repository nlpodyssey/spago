// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

// SinusoidalPositionalEncoder uses the sine and cosine functions of different frequencies to compose position embeddings so to
// incorporate a notion of word order in non-recurrent models (Vaswani et al., 2017).
type SinusoidalPositionalEncoder[T mat.DType] struct {
	// Size is the encoding vector size.
	Size int
	// Length is the max number of positions.
	Length int
	// Vectors contains the pre-computed encoding.
	Vectors []mat.Matrix[T]
}

// NewSinusoidalPositionalEncoder returns a new SinusoidalPositionalEncoder ready to use.
func NewSinusoidalPositionalEncoder[T mat.DType](size, length int) *SinusoidalPositionalEncoder[T] {
	pe := &SinusoidalPositionalEncoder[T]{
		Size:    size,
		Length:  length,
		Vectors: make([]mat.Matrix[T], length),
	}

	half := (size + (size % 2)) / 2

	// pre-compute the encoding for each position
	for pos := 0; pos < length; pos++ {
		data := make([]T, size, size)

		for i := 0; i < size; i++ {
			v := T(pos) / mat.Pow(10000, 2*T(i/2)/T(size))
			if i%2 == 0 {
				data[i/2] = mat.Sin(v)
			} else {
				data[half+i/2] = mat.Cos(v)
			}
		}

		pe.Vectors[pos] = mat.NewVecDense[T](data)
	}
	return pe
}

// Encode returns the positional encoding for the given positions.
func (r *SinusoidalPositionalEncoder[T]) Encode(xs ...int) []mat.Matrix[T] {
	ys := make([]mat.Matrix[T], len(xs))
	for i, x := range xs {
		ys[i] = r.Vectors[x]
	}
	return ys
}
