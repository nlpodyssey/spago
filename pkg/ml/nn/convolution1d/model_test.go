// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution1d

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

//gocyclo:ignore
func TestModel_Forward(t *testing.T) {
	t.Run("float32", testModelForward[float32])
	t.Run("float64", testModelForward[float64])
}

func testModelForward[T mat.DType](t *testing.T) {
	model := newTestModel[T]()
	g := ag.NewGraph[T]()

	// == Forward

	x1 := g.NewVariable(mat.NewDense(2, 4, []T{
		0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.3,
	}), true)

	x2 := g.NewVariable(mat.NewDense(2, 4, []T{
		-0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.9,
	}), true)

	x3 := g.NewVariable(mat.NewDense(2, 4, []T{
		0.2, 0.5, 0.9, 0.8,
		0.4, -0.5, -0.3, -0.2,
	}), true)

	y := nn.ReifyForTraining(model, g).Forward(x1, x2, x3)

	assert.InDeltaSlice(t, []T{
		0.62914516, 0.42189900, 0.03997868,
	}, y[0].Value().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.77788806, 0.9775871, 0.99681227,
	}, y[1].Value().Data(), 1.0e-05)

	y[0].PropagateGrad(mat.NewDense(1, 3, []T{
		-0.3, 0.5, 0.6,
	}))

	y[1].PropagateGrad(mat.NewDense(1, 3, []T{
		-0.3, 0.5, -0.6,
	}))

	g.BackwardAll()

	assert.InDeltaSlice(t, []T{
		0.30437, 0.66660786,
		-0.31560957, -0.20753658,
	}, model.K[0].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.37687117, 0.66660786,
		-0.31560957, -0.56696117,
	}, model.K[1].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.70838666, 0.7585069,
		-0.4577138, -0.15248194,
	}, model.K[2].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.82878876,
	}, model.B[0].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.82878876,
	}, model.B[1].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.82878876,
	}, model.B[2].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.023386842, -0.0038212487,
		-0.05327147, 0.032253552,
	}, model.K[3].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.023999983, -0.0038212487,
		-0.05327147, 0.03454506,
	}, model.K[4].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.016049873, -0.04234343,
		-0.057321873, 0.053348884,
	}, model.K[5].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.10012464,
	}, model.B[3].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.10012464,
	}, model.B[4].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.10012464,
	}, model.B[5].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.13801329, 0.19209248, 0.15132189, -0.24267177,
		0.052244477, 0.0015920401, 0.3153144, 0.17818466,
	}, x1.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.09062646, -0.3191097, -0.16513953, 0.17780274,
		-0.07179071, -0.015045494, 0.4774822, 0.54104656,
	}, x2.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		-0.108041294, 0.045592614, 0.37506783, 0.17742082,
		-0.059944, -0.11689296, 0.38337404, 0.35636932,
	}, x3.Grad().Data(), 1.0e-05)
}

func TestDepthwise_Forward(t *testing.T) {
	t.Run("float32", testDepthwiseForward[float32])
	t.Run("float64", testDepthwiseForward[float64])
}

func testDepthwiseForward[T mat.DType](t *testing.T) {
	model := newTestModel2[T]()
	g := ag.NewGraph[T]()

	// == Forward

	x1 := g.NewVariable(mat.NewDense(2, 4, []T{
		0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.3,
	}), true)

	x2 := g.NewVariable(mat.NewDense(2, 4, []T{
		-0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.9,
	}), true)

	x3 := g.NewVariable(mat.NewDense(2, 4, []T{
		0.2, 0.5, 0.9, 0.8,
		0.4, -0.5, -0.3, -0.2,
	}), true)

	y := nn.ReifyForTraining(model, g).Forward(x1, x2, x3)

	assert.InDeltaSlice(t, []T{
		0.09, -0.3, -0.22,
	}, y[0].Value().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.14, 0.06, -0.66,
	}, y[1].Value().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []T{
		0.51, 0.69, 0.92,
	}, y[2].Value().Data(), 1.0e-05)
}

func newTestModel[T mat.DType]() *Model[T] {
	model := New[T](Config{
		KernelSizeX:    2,
		KernelSizeY:    2,
		YStride:        1,
		InputChannels:  3,
		OutputChannels: 2,
		Mask:           []int{1, 1, 1},
		DepthWise:      false,
		Activation:     ag.OpTanh,
	})
	model.K[0].Value().SetData([]T{
		0.5, -0.4,
		0.3, 0.3,
	})
	model.K[1].Value().SetData([]T{
		-0.5, 0.3,
		0.2, 0.9,
	})
	model.K[2].Value().SetData([]T{
		0.4, 0.3,
		0.2, 0.6,
	})
	model.B[0].Value().SetData([]T{0.0})
	model.B[1].Value().SetData([]T{0.2})
	model.B[2].Value().SetData([]T{0.5})
	model.K[3].Value().SetData([]T{
		0.4, 0.8,
		-0.9, 0.4,
	})
	model.K[4].Value().SetData([]T{
		0.0, 0.5,
		0.3, -0.5,
	})
	model.K[5].Value().SetData([]T{
		0.3, 0.6,
		0.2, 0.8,
	})
	model.B[3].Value().SetData([]T{0.4})
	model.B[4].Value().SetData([]T{0.1})
	model.B[5].Value().SetData([]T{0.5})
	return model
}

func newTestModel2[T mat.DType]() *Model[T] {
	model := New[T](Config{
		KernelSizeX:    2,
		KernelSizeY:    2,
		YStride:        1,
		InputChannels:  3,
		OutputChannels: 3,
		Mask:           []int{1, 1, 1},
		DepthWise:      true,
		Activation:     ag.OpIdentity,
	})
	model.K[0].Value().SetData([]T{
		0.5, -0.4,
		0.3, 0.3,
	})
	model.K[1].Value().SetData([]T{
		-0.5, 0.3,
		0.2, 0.9,
	})
	model.K[2].Value().SetData([]T{
		0.4, 0.3,
		0.2, 0.6,
	})
	model.B[0].Value().SetData([]T{0.0})
	model.B[1].Value().SetData([]T{0.2})
	model.B[2].Value().SetData([]T{0.5})
	return model
}
