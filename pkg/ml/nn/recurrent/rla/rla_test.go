// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rla

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_ForwardWithPrev(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)

	// == Forward
	x0 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9, 1.0}), true)
	_ = proc.Forward(x0)
	s0 := proc.LastState()

	assert.InDeltaSlice(t, []mat.Float{0.88, -1.1, -0.45, 0.41}, s0.Y.Value().Data(), 1.0e-05)

	x1 := g.NewVariable(mat.NewVecDense([]mat.Float{0.8, -0.3, 0.5, 0.3}), true)
	_ = proc.Forward(x1)
	s1 := proc.LastState()

	assert.InDeltaSlice(t, []mat.Float{0.5996537, -0.545537, -0.63689751, 0.453609420}, s1.Y.Value().Data(), 1.0e-05)
}

func newTestModel() *Model {
	model := New(Config{
		InputSize: 4,
	})
	model.Wv.Value().SetData([]mat.Float{
		0.5, 0.6, -0.8, 0.7,
		-0.4, 0.1, 0.7, -0.7,
		0.3, 0.8, -0.9, 0.0,
		0.5, -0.4, -0.5, -0.3,
	})
	model.Bv.Value().SetData([]mat.Float{0.4, 0.0, -0.3, 0.3})
	model.Wk.Value().SetData([]mat.Float{
		0.7, -0.2, -0.1, 0.2,
		-0.1, -0.1, 0.3, -0.2,
		0.6, 0.1, 0.9, 0.3,
		0.3, 0.6, 0.4, 0.2,
	})
	model.Bk.Value().SetData([]mat.Float{0.8, -0.2, -0.5, -0.9})
	model.Wq.Value().SetData([]mat.Float{
		-0.8, -0.6, 0.2, 0.5,
		0.7, -0.6, -0.3, 0.6,
		-0.3, 0.3, 0.4, -0.8,
		0.8, 0.2, 0.4, 0.3,
	})
	model.Bq.Value().SetData([]mat.Float{0.3, 0.5, -0.7, -0.6})
	return model
}
