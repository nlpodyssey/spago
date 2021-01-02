// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModelReLU_Forward(t *testing.T) {
	g := ag.NewGraph()
	m := New(ag.OpReLU)
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	p := nn.Reify(ctx, m).(*Model)

	// == Forward
	x := g.NewVariable(mat.NewVecDense([]mat.Float{0.1, -0.2, 0.3, 0.0}), true)
	y := nn.ToNode(p.Forward(x))

	assert.InDeltaSlice(t, []mat.Float{0.1, 0.0, 0.3, 0.0}, y.Value().Data(), 1.0e-05)

	// == Backward
	g.Backward(y, ag.OutputGrad(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0})))

	assert.InDeltaSlice(t, []mat.Float{-1.0, 0.0, 0.8, 0.0}, x.Grad().Data(), 1.0e-6)
}

func TestModelSwish_Forward(t *testing.T) {
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}

	beta := nn.NewParam(mat.NewScalar(2.0))
	model := New(ag.OpSwish, beta)
	p := nn.Reify(ctx, model).(*Model)

	// == Forward
	x := g.NewVariable(mat.NewVecDense([]mat.Float{0.1, -0.2, 0.3, 0.0}), true)
	y := nn.ToNode(p.Forward(x))

	assert.InDeltaSlice(t, []mat.Float{0.0549833997, -0.080262468, 0.1936968919, 0.0}, y.Value().Data(), 1.0e-6)

	// == Backward
	g.Backward(y, ag.OutputGrad(mat.NewVecDense([]mat.Float{-1.0, 0.5, 0.8, 0.0})))

	assert.InDeltaSlice(t, []mat.Float{-0.5993373119, 0.1526040208, 0.6263414804, 0.0}, x.Grad().Data(), 1.0e-6)
	assert.InDeltaSlice(t, []mat.Float{0.0188025145}, beta.Grad().Data(), 1.0e-6)
}
