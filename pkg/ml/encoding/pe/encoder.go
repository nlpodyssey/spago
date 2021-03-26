// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

// SinusoidalPositionalEncoder uses the sine and cosine functions of different frequencies to compose position embeddings so to
// incorporate a notion of word order in non-recurrent models (Vaswani et al., 2017).
type SinusoidalPositionalEncoder struct {
	// Size is the encoding vector size.
	Size int
	// Length is the max number of positions.
	Length int
	// Vectors contains the pre-computed encoding.
	Vectors []mat.Matrix
}

// NewSinusoidalPositionalEncoder returns a new SinusoidalPositionalEncoder ready to use.
func NewSinusoidalPositionalEncoder(size, length int) *SinusoidalPositionalEncoder {
	pe := &SinusoidalPositionalEncoder{
		Size:    size,
		Length:  length,
		Vectors: make([]mat.Matrix, length),
	}

	half := (size + (size % 2)) / 2

	// pre-compute the encoding for each position
	for pos := 0; pos < length; pos++ {
		data := make([]mat.Float, size, size)

		for i := 0; i < size; i++ {
			v := mat.Float(pos) / mat.Pow(10000, 2*mat.Float(i/2)/mat.Float(size))
			if i%2 == 0 {
				data[i/2] = mat.Sin(v)
			} else {
				data[half+i/2] = mat.Cos(v)
			}
		}

		pe.Vectors[pos] = mat.NewVecDense(data)
	}
	return pe
}

// Encode returns the positional encoding for the given positions.
func (r *SinusoidalPositionalEncoder) Encode(xs ...int) []mat.Matrix {
	ys := make([]mat.Matrix, len(xs))
	for i, x := range xs {
		ys[i] = r.Vectors[x]
	}
	return ys
}
