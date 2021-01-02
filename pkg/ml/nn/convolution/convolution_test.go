// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package convolution

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

//gocyclo:ignore
func TestModel_Forward(t *testing.T) {
	model := newTestModel()
	g := ag.NewGraph()
	ctx := nn.Context{Graph: g, Mode: nn.Training}

	// == Forward

	x1 := g.NewVariable(mat.NewDense(4, 4, []mat.Float{
		0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.3,
		0.5, -0.6, -0.4, 0.6,
		-0.3, 0.9, 0.5, 0.5,
	}), true)

	x2 := g.NewVariable(mat.NewDense(4, 4, []mat.Float{
		-0.2, 0.1, 0.5, 0.8,
		0.4, -0.3, -0.2, -0.9,
		0.5, 0.2, 0.2, 0.9,
		0.9, 0.3, 0.2, 0.7,
	}), true)

	x3 := g.NewVariable(mat.NewDense(4, 4, []mat.Float{
		0.2, 0.5, 0.9, 0.8,
		0.4, -0.5, -0.3, -0.2,
		0.5, 0.6, -0.9, 0.0,
		0.3, 0.9, 0.2, 0.1,
	}), true)

	y := nn.Reify(ctx, model).(*Model).Forward(x1, x2, x3)

	assert.InDeltaSlice(t, []mat.Float{
		0.6291451614, 0.4218990053, 0.0399786803,
		0.8956928738, -0.0698858903, 0.8004990218,
		0.9892435057, 0.8956928738, 0.8144140938,
	}, y[0].Value().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.7778880666, 0.9775871874, 0.9968122755,
		0.4853810906, 0.0299910032, 0.049958375,
		0.9934620209, -0.0996679946, 0.7931990971,
	}, y[1].Value().Data(), 1.0e-05)

	y[0].PropagateGrad(mat.NewDense(3, 3, []mat.Float{
		-0.3, 0.5, 0.6,
		0.9, 0.1, 0.0,
		0.3, 0.4, -1.0,
	}))

	y[1].PropagateGrad(mat.NewDense(3, 3, []mat.Float{
		-0.3, 0.5, -0.6,
		-0.2, 0.0, 0.1,
		0.3, 0.6, 0.0,
	}))

	g.BackwardAll()

	assert.InDeltaSlice(t, []mat.Float{
		0.4361460918, 0.3557904551,
		-0.385442345, -0.4771584238,
	}, model.K[0].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.3698844136, 0.3073631249,
		-0.2445673659, -0.7294329628,
	}, model.K[1].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		1.083537722, 0.5723401861,
		-0.3032622381, -0.1473428208,
	}, model.K[2].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.8550443848,
	}, model.B[0].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.8550443848,
	}, model.B[1].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.8550443848,
	}, model.B[2].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-0.4589582127, -0.227843921,
		0.3638506439, 0.4843712647,
	}, model.K[3].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.0636604256, 0.0718576652,
		0.0719689855, 0.2137251507,
	}, model.K[4].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.2512514644, -0.5181427625,
		0.3122710023, 0.083947163,
	}, model.K[5].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.4446945415,
	}, model.B[3].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.4446945415,
	}, model.B[4].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.4446945415,
	}, model.B[5].Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-0.1380132736, 0.1920924921, 0.1513219132, -0.2426717471,
		0.0800724775, -0.1421413609, 0.3154099326, 0.2579849708,
		0.1957547684, 0.2998123793, 0.215307598, 0.17459204,
		-0.0015932117, -0.5074179426, 0.1603251177, -0.1010189052,
	}, x1.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		0.0906264549, -0.3191097037, -0.1651395044, 0.1778027207,
		-0.1607711201, -0.0878535422, 0.5073356622, 0.590921715,
		-0.0134817352, 0.2208414849, 0.6085984037, -0.1508941132,
		0.0024568264, 0.1978529598, -0.2931814848, -0.3030567155,
	}, x2.Grad().Data(), 1.0e-05)

	assert.InDeltaSlice(t, []mat.Float{
		-0.1080412779, 0.0455926386, 0.3750678293, 0.1774208035,
		-0.0346239638, -0.1154286619, 0.4431526591, 0.4162195256,
		0.0087566253, 0.2184951473, 0.3251171452, -0.0212185723,
		0.00206583, 0.1416061797, 0.455342109, -0.2020378103,
	}, x3.Grad().Data(), 1.0e-05)
}

func newTestModel() *Model {
	model := New(Config{
		KernelSizeX:    2,
		KernelSizeY:    2,
		XStride:        1,
		YStride:        1,
		InputChannels:  3,
		OutputChannels: 2,
		Mask:           []int{1, 1, 1},
		Activation:     ag.OpTanh,
	})
	model.K[0].Value().SetData([]mat.Float{
		0.5, -0.4,
		0.3, 0.3,
	})
	model.K[1].Value().SetData([]mat.Float{
		-0.5, 0.3,
		0.2, 0.9,
	})
	model.K[2].Value().SetData([]mat.Float{
		0.4, 0.3,
		0.2, 0.6,
	})
	model.B[0].Value().SetData([]mat.Float{0.0})
	model.B[1].Value().SetData([]mat.Float{0.2})
	model.B[2].Value().SetData([]mat.Float{0.5})
	model.K[3].Value().SetData([]mat.Float{
		0.4, 0.8,
		-0.9, 0.4,
	})
	model.K[4].Value().SetData([]mat.Float{
		0.0, 0.5,
		0.3, -0.5,
	})
	model.K[5].Value().SetData([]mat.Float{
		0.3, 0.6,
		0.2, 0.8,
	})
	model.B[3].Value().SetData([]mat.Float{0.4})
	model.B[4].Value().SetData([]mat.Float{0.1})
	model.B[5].Value().SetData([]mat.Float{0.5})
	return model
}

func newTestModel2() *Model {
	model := New(Config{
		KernelSizeX:    2,
		KernelSizeY:    2,
		XStride:        1,
		YStride:        1,
		InputChannels:  3,
		OutputChannels: 2,
		Mask:           []int{0, 1, 1},
		Activation:     ag.OpTanh,
	})
	model.K[0].Value().SetData([]mat.Float{
		0.5, -0.4,
		0.3, 0.3,
	})
	model.K[1].Value().SetData([]mat.Float{
		-0.5, 0.3,
		0.2, 0.9,
	})
	model.K[2].Value().SetData([]mat.Float{
		0.4, 0.3,
		0.2, 0.6,
	})
	model.B[0].Value().SetData([]mat.Float{0.0})
	model.B[1].Value().SetData([]mat.Float{0.2})
	model.B[2].Value().SetData([]mat.Float{0.5})
	model.K[3].Value().SetData([]mat.Float{
		0.4, 0.8,
		-0.9, 0.4,
	})
	model.K[4].Value().SetData([]mat.Float{
		0.0, 0.5,
		0.3, -0.5,
	})
	model.K[5].Value().SetData([]mat.Float{
		0.3, 0.6,
		0.2, 0.8,
	})
	model.B[3].Value().SetData([]mat.Float{0.4})
	model.B[4].Value().SetData([]mat.Float{0.1})
	model.B[5].Value().SetData([]mat.Float{0.5})
	return model
}
