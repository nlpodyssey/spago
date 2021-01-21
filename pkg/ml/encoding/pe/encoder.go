// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
)

// PositionalEncoder uses the sine and cosine functions of different frequencies to compose position embeddings so to
// incorporate a notion of word order in non-recurrent models (Vaswani et al., 2017).
// The wavelengths form a geometric progression from 2π to 10000·2π. so to easily learn to attend by relative positions.
// Each dimension of the positional encoding corresponds to a sinusoid:
//    PE(pos,2i) = sin(pos/10000**(2i/size))
//    PE(pos,2i+1) = cos(pos/10000**(2i+1/size))
// where pos is the position (up to a maximal length) and i is the dimension (up to size).
type PositionalEncoder struct {
	// Size is the encoding vector size.
	Size int
	// Length is the max number of positions.
	Length int
	// Cache contains the pre-computed encoding.
	Cache []mat.Matrix
}

var log10000 = mat.Log(10000)

// NewPositionalEncoder returns a new PositionalEncoder ready to use.
func NewPositionalEncoder(size, length int) *PositionalEncoder {
	pe := &PositionalEncoder{
		Size:   size,
		Length: length,
		Cache:  make([]mat.Matrix, length),
	}
	// pre-compute the encoding for each position, calculating it in natural log-space
	for pos := 0; pos < length; pos++ {
		data := make([]mat.Float, size, size)
		for i := 0; i < size-1; i += 2 {
			data[i] = mat.Sin(mat.Float(pos) * mat.Exp(mat.Float(i)*-log10000/mat.Float(size)))
			data[i+1] = mat.Cos(mat.Float(pos) * mat.Exp(mat.Float(i)*-log10000/mat.Float(size)))
		}
		pe.Cache[pos] = mat.NewVecDense(data)
	}
	return pe
}

// EncodingAt returns the positional encoding at the given position.
func (r *PositionalEncoder) EncodingAt(pos int) mat.Matrix {
	return r.Cache[pos]
}
