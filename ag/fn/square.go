// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fn

import (
	"github.com/nlpodyssey/spago/mat"
)

// Square is an operator to perform element-wise square.
type Square[T mat.DType, O Operand[T]] struct {
	*Prod[T, O]
}

// NewSquare returns a new Prod Function with both operands set to the given value x.
func NewSquare[T mat.DType, O Operand[T]](x O) *Square[T, O] {
	return &Square[T, O]{Prod: &Prod[T, O]{x1: x, x2: x}}
}

// Operands returns the list of operands.
func (r *Square[T, O]) Operands() []O {
	return []O{r.x1}
}
