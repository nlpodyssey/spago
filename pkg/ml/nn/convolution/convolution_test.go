// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution

import (
	"brillion.io/spago/pkg/mat"
	"brillion.io/spago/pkg/ml/act"
	"brillion.io/spago/pkg/ml/ag"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()

	// == Forward

	x1 := g.NewVariable(mat.NewDense(4, 4, []float64{
		0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.3,
		0.5, -0.6, -0.4, 0.6,
		-0.3, 0.9, 0.5, 0.5,
	}), true)

	x2 := g.NewVariable(mat.NewDense(4, 4, []float64{
		-0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.9,
		0.5, 0.2, 0.2, 0.9,
		0.9, 0.3, 0.2, 0.7,
	}), true)

	x3 := g.NewVariable(mat.NewDense(4, 4, []float64{
		0.2, 0.5, 0.9, 0.8,
		0.4, -0.5, -0.3, -0.2,
		0.5, 0.6, -0.9, 0.0,
		0.3, 0.9, 0.2, 0.1,
	}), true)

	y := model.NewProc(g).Forward(x1, x2, x3)[0]

	if !floats.EqualApprox(y.Value().Data(), []float64{
		0.6291451614, 0.4218990053, 0.0399786803,
		0.8956928738, -0.0698858903, 0.8004990218,
		0.9892435057, 0.8956928738, 0.8144140938,
	}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	g.Backward(y, mat.NewDense(3, 3, []float64{
		-0.3, 0.5, 0.6,
		0.9, 0.1, 0.0,
		0.3, 0.4, -1.0,
	}))

	if !floats.EqualApprox(x1.Grad().Data(), []float64{
		-0.0906264549, 0.2780014712, 0.1351202657, -0.2396164092,
		0.0346045512, 0.0474957703, 0.2632078528, 0.1797123069,
		0.0565978474, 0.1202209141, -0.1701488472, 0.1346918736,
		0.0019257558, 0.0256538689, -0.0772907921, -0.1010189052,
	}, 1.0e-05) {
		t.Error("x1 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{
		0.0906264549, -0.2598761803, -0.1762203271, 0.1797123069,
		-0.125231006, -0.0772950395, 0.5195622367, 0.5391369208,
		0.0323825767, 0.1424459832, 0.2816533916, -0.1010189052,
		0.0012838372, 0.0215960094, 0.0038384025, -0.3030567155,
	}, 1.0e-05) {
		t.Error("x2 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x3.Grad().Data(), []float64{
		-0.0725011639, 0.1100243729, 0.3629165936, 0.1797123069,
		0.0349337573, 0.06664127, 0.3962620523, 0.3594246138,
		0.038159844, 0.1602420681, -0.0512568027, -0.1010189052,
		0.0012838372, 0.0196702536, -0.0198897106, -0.2020378103,
	}, 1.0e-05) {
		t.Error("x3 gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.K[0].Grad().Data(), []float64{
		0.4361460918, 0.3557904551,
		-0.385442345, -0.4771584238,
	}, 1.0e-05) {
		t.Error("K 1 gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.K[1].Grad().Data(), []float64{
		0.3698844136, 0.3073631249,
		-0.2445673659, -0.7294329628,
	}, 1.0e-05) {
		t.Error("K 2 gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.K[2].Grad().Data(), []float64{
		1.083537722, 0.5723401861,
		-0.3032622381, -0.1473428208,
	}, 1.0e-05) {
		t.Error("K 3 gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.B[0].Grad().Data(), []float64{
		0.8550443848,
	}, 1.0e-05) {
		t.Error("B 1 gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.B[1].Grad().Data(), []float64{
		0.8550443848,
	}, 1.0e-05) {
		t.Error("B 2 gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.B[2].Grad().Data(), []float64{
		0.8550443848,
	}, 1.0e-05) {
		t.Error("B 3 gradients don't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(2, 2, 3, act.Tanh)
	model.K[0].Value().SetData([]float64{
		0.5, -0.4,
		0.3, 0.3,
	})
	model.K[1].Value().SetData([]float64{
		-0.5, 0.3,
		0.2, 0.9,
	})
	model.K[2].Value().SetData([]float64{
		0.4, 0.3,
		0.2, 0.6,
	})
	model.B[0].Value().SetData([]float64{0.0})
	model.B[1].Value().SetData([]float64{0.2})
	model.B[2].Value().SetData([]float64{0.5})
	return model
}
