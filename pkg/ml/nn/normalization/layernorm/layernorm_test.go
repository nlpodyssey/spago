// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layernorm

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}

	// == Forward
	x := g.NewVariable(mat.NewVecDense([]mat.Float{0.4, 0.8, -0.7, -0.5}), true)
	y := nn.ToNode(nn.Reify(ctx, model).(*Model).Forward(x))

	assert.InDeltaSlice(t, []mat.Float{1.157863, 0.2, -0.561554, -0.444658}, y.Value().Data(), 1.0e-06)

	// == Backward
	y.PropagateGrad(mat.NewVecDense([]mat.Float{-1.0, -0.2, 0.4, 0.6}))
	g.BackwardAll()

	assert.InDeltaSlice(t, []mat.Float{-0.496261, 0.280677, -0.408772, 0.624355}, x.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{-0.644658, -0.257863, -0.45126, -0.483493}, model.W.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{-1.0, -0.2, 0.4, 0.6}, model.B.Grad().Data(), 1.0e-06)
}

func newTestModel() *Model {
	model := New(4)
	model.W.Value().SetData([]mat.Float{0.4, 0.0, -0.3, 0.8})
	model.B.Value().SetData([]mat.Float{0.9, 0.2, -0.9, 0.2})
	return model
}
