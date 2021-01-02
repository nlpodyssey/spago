// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tpr

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, 0.9, 0.1}), true)
	_ = proc.Forward(x)
	st := proc.LastState()

	assert.InDeltaSlice(t, []mat.Float{0.050298, 0.029289, 0.321719, 0.187342, 0.149808, 0.087235}, st.Y.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.569546, 0.748381, 0.509998, 0.345246}, st.AS.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.291109, 0.391740, 0.394126}, st.AR.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.142810, 0.913446, 0.425346}, st.S.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.352204, 0.205093}, st.R.Value().Data(), 0.000001)

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]mat.Float{0.57, 0.75, -0.15, 1.64, 0.45, 0.11}), false)

	mse := losses.MSE(g, st.Y, gold, false)
	q1 := losses.OneHotQuantization(g, st.AR, 0.001)
	q2 := losses.OneHotQuantization(g, st.AS, 0.001)
	q := g.Add(q1, q2)
	loss := g.Add(mse, q)
	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{
		-0.083195466589325, -0.079995855904333, -0.000672136225078, 0.023205789428363,
	}, x.Grad().Data(), 0.000001)
}

func TestModel_ForwardWithPrev(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()

	yPrev := g.NewVariable(mat.NewVecDense([]mat.Float{0.211, -0.451, 0.499, -1.333, -0.11645, 0.366}), true)
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)
	proc.SetInitialState(&State{Y: yPrev})

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, 0.9, 0.1}), true)
	_ = proc.Forward(x)
	st := proc.LastState()

	assert.InDeltaSlice(t, []mat.Float{
		0.05472795, 0.0308627,
		0.2054040, 0.1158336,
		0.0429874, 0.0242419,
	}, st.Y.Value().Data(), 0.000001)

	assert.InDeltaSlice(t, []mat.Float{0.3104128, 0.8803527, 0.3561176, 0.5755996}, st.AS.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.0754811, 0.6198861, 0.3573797}, st.AR.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.169241341812798, 0.635193673105892, 0.132934724456263}, st.S.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.323372243322051, 0.182359559619209}, st.R.Value().Data(), 0.000001)

	// == Backward

	gold := g.NewVariable(mat.NewVecDense([]mat.Float{0.57, 0.75, -0.15, 1.64, 0.45, 0.11}), false)

	mse := losses.MSE(g, st.Y, gold, false)
	q1 := losses.OneHotQuantization(g, st.AR, 0.001)
	q2 := losses.OneHotQuantization(g, st.AS, 0.001)
	q := g.Add(q1, q2)
	loss := g.Add(mse, q)
	g.Backward(loss)

	assert.InDeltaSlice(t, []mat.Float{
		-0.060099369011985, -0.048029952866947, -0.028715724278403, 0.004889227782339,
	}, x.Grad().Data(), 0.000001)
}

func TestModel_ForwardSeq(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}
	proc := nn.Reify(ctx, model).(*Model)
	proc.SetInitialState(&State{
		Y: g.NewVariable(mat.NewVecDense([]mat.Float{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}), true),
	})

	// == Forward

	x := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, 0.9, 0.1}), true)
	_ = proc.Forward(x)
	s := proc.LastState()

	assert.InDeltaSlice(t, []mat.Float{0.05029859664638596, 0.02928963193170334,
		0.3217195687341599, 0.18734216025343006,
		0.1498086999769255, 0.08723586690378164,
	}, s.Y.Value().Data(), 0.000001)

	assert.InDeltaSlice(t, []mat.Float{0.5695462239392289, 0.7483817216070642, 0.5099986668799654, 0.3452465393936807}, s.AS.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.2911098274338801, 0.3917409692534855, 0.3941263315682394}, s.AR.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.14281092586919966, 0.9134463492922567, 0.4253462437008787}, s.S.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.3522041212200695, 0.20509377523768507}, s.R.Value().Data(), 0.000001)

	x2 := g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, 0.9, 0.1}), true)
	_ = proc.Forward(x2)
	s2 := proc.LastState()

	assert.InDeltaSlice(t, []mat.Float{0.03398428524859144, 0.019818448417970973,
		0.38891858550151426, 0.22680373793859504,
		0.1921681864263287, 0.1120657757668473,
	}, s2.Y.Value().Data(), 0.000001)

	assert.InDeltaSlice(t, []mat.Float{0.5639735444997409, 0.8022024627153614, 0.5576652280441475, 0.271558247560365}, s2.AS.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.2978298580977239, 0.4601236969137651, 0.4189711189333963}, s2.AR.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.08876417178261771, 1.0158235160864522, 0.5019275758288234}, s2.S.Value().Data(), 0.000001)
	assert.InDeltaSlice(t, []mat.Float{0.38286038799323796, 0.2232708087054098}, s2.R.Value().Data(), 0.000001)

	// == Backward

	s.Y.PropagateGrad(mat.NewVecDense([]mat.Float{-0.2, -0.3, -0.4, 0.6, 0.3, 0.3}))
	s2.Y.PropagateGrad(mat.NewVecDense([]mat.Float{0.6, -0.3, -0.8, 0.2, 0.4, -0.8}))

	g.BackwardAll()

	assert.InDeltaSlice(t, []mat.Float{
		-0.020471392359016696, -0.021638276740337678, -0.004140271053657623, -0.019166708609272186,
	}, x.Grad().Data(), 0.000001)

	assert.InDeltaSlice(t, []mat.Float{
		-0.07442057206008514, -0.06614504823252586, -0.06037883696058007, 0.00909050757456592,
	}, x2.Grad().Data(), 0.000001)
}

func newTestModel() *Model {
	params := New(
		4, // in
		4, // nSymbols
		3, // dSymbols
		3, // nRoles
		2, // dRoles
	)
	params.WInS.Value().SetData([]mat.Float{
		0.2, 0.1, 0.3, -0.4,
		0.3, -0.1, 0.9, 0.3,
		0.4, 0.2, -0.3, 0.1,
		0.6, 0.5, -0.4, 0.5,
	})
	params.WInR.Value().SetData([]mat.Float{
		0.3, 0.5, -0.5, -0.5,
		0.5, 0.4, 0.1, 0.3,
		0.6, 0.7, 0.8, 0.6,
	})
	params.WRecS.Value().SetData([]mat.Float{
		0.4, 0.2, -0.4, 0.5, 0.2, -0.5,
		-0.2, 0.7, 0.8, -0.5, 0.5, 0.7,
		0.4, -0.1, 0.1, 0.7, -0.1, 0.3,
		0.3, 0.2, -0.7, -0.8, -0.3, 0.6,
	})
	params.WRecR.Value().SetData([]mat.Float{
		0.4, 0.8, -0.4, 0.7, 0.2, -0.5,
		-0.2, 0.7, 0.8, -0.5, 0.3, 0.7,
		0.3, -0.1, 0.1, 0.3, -0.1, 0.2,
	})
	params.BS.Value().SetData([]mat.Float{0.3, 0.4, 0.8, 0.6})
	params.BR.Value().SetData([]mat.Float{0.3, 0.2, -0.1})
	params.S.Value().SetData([]mat.Float{
		0.3, -0.2, -0.1, 0.5,
		0.6, 0.7, 0.5, -0.6,
		0.4, 0.2, 0.5, -0.6,
	})
	params.R.Value().SetData([]mat.Float{
		0.4, 0.3, 0.3,
		0.3, 0.2, 0.1,
	})
	return params
}
