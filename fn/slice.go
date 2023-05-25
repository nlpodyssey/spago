// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
)

// Slice is a function to extract a portion of a matrix.
type Slice[O DualValue] struct {
	x       O
	fromRow int
	fromCol int
	toRow   int
	toCol   int
}

// NewSlice returns a new Slice Function.
func NewSlice[O DualValue](x O, fromRow, fromCol, toRow, toCol int) *Slice[O] {
	return &Slice[O]{
		x:       x,
		fromRow: fromRow,
		fromCol: fromCol,
		toRow:   toRow,
		toCol:   toCol,
	}
}

// Operands returns the list of operands.
func (s *Slice[O]) Operands() []O {
	return []O{s.x}
}

// Forward computes the output of the function.
func (s *Slice[O]) Forward() (mat.Matrix, error) {
	return s.x.Value().Slice(s.fromRow, s.fromCol, s.toRow, s.toCol), nil
}

// Backward computes the backward pass.
func (s *Slice[O]) Backward(gy mat.Matrix) error {
	lx := s.toRow - s.fromRow
	ly := s.toCol - s.fromCol
	if !(gy.Rows() == lx && gy.Cols() == ly) {
		return fmt.Errorf("fn: matrices have incompatible dimensions")
	}
	if s.x.RequiresGrad() {
		gx := s.x.Value().ZerosLike()
		for i := 0; i < lx; i++ {
			for j := 0; j < ly; j++ {
				gx.SetScalar(gy.ScalarAt(i, j), i+s.fromRow, j+s.fromCol)
			}
		}
		s.x.AccGrad(gx)
	}
	return nil
}
