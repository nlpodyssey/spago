// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

import (
	"github.com/nlpodyssey/spago/pkg/mat"
)

// AxialPositionalEncoder uses part of the original encodings to save memory space.
type AxialPositionalEncoder struct {
	// Size is the encoding vector size.
	Size int
	// D is the index where vector is split (D < Size).
	D int
	// Length is the max number of positions.
	Length int
	// The dimensions of the axis, such as Width X Height = Length.
	Width, Height int
	// cache contains the pre-computed encoding.
	cache []*mat.Dense
}

// NewAxialPositionalEncoder returns a new AxialPositionalEncoder ready to use.
func NewAxialPositionalEncoder(size, d, length, width, height int) *AxialPositionalEncoder {
	pe := &AxialPositionalEncoder{
		Size:   size,
		D:      d,
		Length: length,
		Width:  width,
		Height: height,
	}
	if !(pe.Height*pe.Width == pe.Length) {
		panic("pe: new axial encoding with invalid factorization dimensions")
	}
	if !(pe.D < pe.Size) {
		panic("pe: new axial encoding with invalid size")
	}
	max := pe.Height
	if pe.Width > pe.Height {
		max = pe.Width
	}
	pe.cache = NewPositionalEncoder(size, max).cache
	return pe
}

// EncodingAt returns the positional encoding at the given position.
func (r *AxialPositionalEncoder) EncodingAt(pos int) *mat.Dense {
	data := make([]float64, r.Size)
	for i := 0; i < r.D; i++ {
		data[i] = r.cache[pos%r.Width].Data()[i]
	}
	for i := r.D; i < r.Size; i++ {
		data[i] = r.cache[pos/r.Height].Data()[i]
	}
	return mat.NewVecDense(data)
}
