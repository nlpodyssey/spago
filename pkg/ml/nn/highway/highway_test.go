// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package highway

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

	x := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9, 1.0}), true)
	y := nn.ToNode(nn.Reify(ctx, model).(*Model).Forward(x))

	assert.InDeltaSlice(t, []mat.Float{-0.456097, -0.855358, -0.79552, 0.844718}, y.Value().Data(), 1.0e-05)

	// == Backward

	g.Backward(y, ag.OutputGrad(mat.NewVecDense([]mat.Float{0.57, 0.75, -0.15, 1.64})))

	assert.InDeltaSlice(t, []mat.Float{0.822396, 0.132595, -0.437002, 0.446894}, x.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []mat.Float{
		-0.327765, -0.368736, -0.368736, 0.409706,
		-0.094803, -0.106653, -0.106653, 0.118504,
		0.013931, 0.015672, 0.015672, -0.017413,
		-0.346622, -0.389949, -0.389949, 0.433277,
	}, model.WIn.Grad().(*mat.Dense).Data(), 1.0e-06)

	assert.InDeltaSlice(t, []mat.Float{
		0.409706, 0.118504, -0.017413, 0.433277,
	}, model.BIn.Grad().Data(), 1.0e-06)

	assert.InDeltaSlice(t, []mat.Float{
		-0.023020, -0.025897, -0.025897, 0.028775,
		-0.015190, -0.017088, -0.017088, 0.018987,
		0.011082, 0.012467, 0.012467, -0.013853,
		0.097793, 0.110017, 0.110017, -0.122241,
	}, model.WT.Grad().(*mat.Dense).Data(), 1.0e-06)

	assert.InDeltaSlice(t, []mat.Float{
		0.028775, 0.018987, -0.013853, -0.122241,
	}, model.BT.Grad().Data(), 1.0e-06)
}

func newTestModel() *Model {

	model := New(4, ag.OpTanh)

	model.WIn.Value().SetData([]mat.Float{
		0.5, 0.6, -0.8, -0.6,
		0.7, -0.4, 0.1, -0.8,
		0.7, -0.7, 0.3, 0.5,
		0.8, -0.9, 0.0, -0.1,
	})

	model.WT.Value().SetData([]mat.Float{
		0.1, 0.4, -1.0, 0.4,
		0.7, -0.2, 0.1, 0.0,
		0.7, 0.8, -0.5, -0.3,
		-0.9, 0.9, -0.3, -0.3,
	})

	model.BIn.Value().SetData([]mat.Float{0.4, 0.0, -0.3, 0.8})
	model.BT.Value().SetData([]mat.Float{0.9, 0.2, -0.9, 0.2})

	return model
}
