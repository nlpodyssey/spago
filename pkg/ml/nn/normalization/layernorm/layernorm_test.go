// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layernorm

import (
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/ag"
	"testing"
)

func TestModel_Forward(t *testing.T) {

	model := newTestModel()
	g := ag.NewGraph()

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]float64{0.4, 0.8, -0.7, -0.5}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{-0.4, -0.6, -0.2, -0.9}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{0.4, 0.4, 0.2, 0.8}), true)

	y := model.NewProc(g).Forward(x1, x2, x3)

	_ = y

	// TODO: write tests
}

func newTestModel() *Model {
	model := New(4)
	model.W.Value().SetData([]float64{0.4, 0.0, -0.3, 0.8})
	model.B.Value().SetData([]float64{0.9, 0.2, -0.9, 0.2})
	return model
}
