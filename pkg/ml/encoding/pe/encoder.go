// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"math"
	"saientist.dev/spago/pkg/mat"
)

// PositionalEncoder uses the sine and cosine functions of different frequencies to compose position embeddings so to
// incorporate a notion of word order in non-recurrent models (Vaswani et al., 2017).
// Each dimension of the positional encoding corresponds to a sinusoid no notion of word order.
type PositionalEncoder struct {
	// Size is the encoding vector size.
	Size int
	// Length is the max number of positions.
	Length int
	// cache contains the pre-computed encoding.
	cache []*mat.Dense
}

// New returns a new PositionalEncoder ready to use.
func New(size, length int) *PositionalEncoder {
	pe := &PositionalEncoder{
		Size:   size,
		Length: length,
		cache:  make([]*mat.Dense, length),
	}
	// pre-compute the encoding for each position, calculating it in natural log-space
	for i := 0; i < length; i++ {
		data := make([]float64, size, size)
		for k := 0; k < size; k += 2 {
			divTerm := math.Exp(float64(k) * -math.Log(10000.0) / float64(size))
			data[k] = math.Sin(float64(i) * divTerm)
			data[k+1] = math.Cos(float64(i) * divTerm)
		}
		pe.cache[i] = mat.NewVecDense(data)
	}
	return pe
}

// EncodingAt returns the positional encoding at the given position.
func (r *PositionalEncoder) EncodingAt(pos int) *mat.Dense {
	return r.cache[pos]
}
