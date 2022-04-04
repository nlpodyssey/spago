// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package approxlinear

import (
	"github.com/nlpodyssey/gomaddness"
	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
	"sync"
)

type AMM[T mat.DType, O fn.Operand[T]] struct {
	m        *gomaddness.Maddness[T]
	x        O
	operands []O
}

func NewAMM[T mat.DType, O fn.Operand[T]](m *gomaddness.Maddness[T], x O) *AMM[T, O] {
	return &AMM[T, O]{
		m:        m,
		x:        x,
		operands: []O{x},
	}
}

// Operands returns the list of operands.
func (amm *AMM[_, O]) Operands() []O {
	return amm.operands
}

func (amm *AMM[T, _]) Forward() mat.Matrix[T] {
	maddness := amm.m
	x := amm.x.Value().Data()
	q := maddness.Quantize(x)
	lutIndices := maddness.LookupTableIndices(q)

	y := mat.NewDirtyVecDense[T](len(maddness.LookupTables))
	yData := y.Data()

	var wg sync.WaitGroup
	wg.Add(len(yData))

	for i := range yData {
		go func(i int) {
			yData[i] = maddness.DotProduct(lutIndices, i)
			wg.Done()
		}(i)
	}

	wg.Wait()

	return y
}

func (*AMM[T, _]) Backward(mat.Matrix[T]) {
	panic("AMM.Backward is not implemented")
}
