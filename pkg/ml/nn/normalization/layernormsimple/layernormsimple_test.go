// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layernormsimple

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := New()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}

	// == Forward
	x1 := g.NewVariable(mat.NewVecDense([]mat.Float{1.0, 2.0, 0.0, 4.0}), true)
	x2 := g.NewVariable(mat.NewVecDense([]mat.Float{3.0, 2.0, 1.0, 6.0}), true)
	x3 := g.NewVariable(mat.NewVecDense([]mat.Float{6.0, 2.0, 5.0, 1.0}), true)

	y := nn.Reify(ctx, model).(*Model).Forward(x1, x2, x3)

	assert.InDeltaSlice(t, []mat.Float{-0.5070925528, 0.1690308509, -1.1832159566, 1.5212776585}, y[0].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{0.0, -0.5345224838, -1.0690449676, 1.6035674515}, y[1].Value().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{1.2126781252, -0.7276068751, 0.7276068751, -1.2126781252}, y[2].Value().Data(), 1.0e-06)

	// == Backward

	y[0].PropagateGrad(mat.NewVecDense([]mat.Float{-1.0, -0.2, 0.4, 0.6}))
	y[1].PropagateGrad(mat.NewVecDense([]mat.Float{-0.3, 0.1, 0.7, 0.9}))
	y[2].PropagateGrad(mat.NewVecDense([]mat.Float{0.3, -0.4, 0.7, -0.8}))
	g.BackwardAll()

	assert.InDeltaSlice(t, []mat.Float{-0.5640800969, -0.1274975561, 0.4868088507, 0.2047688023}, x1.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{-0.3474396144, -0.0878144080, 0.2787152951, 0.1565387274}, x2.Grad().Data(), 1.0e-06)
	assert.InDeltaSlice(t, []mat.Float{-0.1440946948, 0.0185468419, 0.1754816581, -0.0499338051}, x3.Grad().Data(), 1.0e-06)
}
