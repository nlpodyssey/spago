package sgu

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	model := New(Config{
		Dim:     16,
		DimSeq:  2,
		InitEps: 0.001,
	})
	defer model.Close()

	model.Norm.W.Value().SetData([]mat.Float{
		0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7,
	})
	model.Norm.B.Value().SetData([]mat.Float{
		0.02, 0.04, 0.06, 0.08, 0.01, 0.03, 0.05, 0.07,
	})

	model.Proj.W.Value().SetData([]mat.Float{
		0.41, 0.42,
		0.43, 0.44,
	})
	model.Proj.B.Value().SetData([]mat.Float{
		0.48, 0.49,
	})

	g := ag.NewGraph()
	defer g.Clear()

	proc := nn.ReifyForInference(model, g).(*Model)

	xs := []ag.Node{
		g.NewVariable(mat.NewVecDense([]mat.Float{
			0.572342, 0.70716673, 0.8478436, 0.9926679, 1.2340385, 1.2887437, 1.4375468, 1.5856494, 1.7324576, 1.8776046, 2.0209146, 2.162365, 2.3020453, 2.44012, 2.5767968, 2.7122996,
		}), false),
		g.NewVariable(mat.NewVecDense([]mat.Float{
			0.572342, 0.70716673, 0.84784335, 0.9926679, 1.2340384, 1.2887436, 1.4375465, 1.585649, 1.7324572, 1.8776044, 2.0209143, 2.1623647, 2.3020446, 2.4401197, 2.5767963, 2.712299,
		}), false),
	}
	ys := proc.Forward(xs...)

	assert.InDeltaSlice(t, []mat.Float{
		0.1373449, 0.10625449, 0.17634945, 0.4072929, 0.62621367, 0.86293584, 1.3986156, 2.244735,
	}, ys[0].Value().Data(), 0.001)

	assert.InDeltaSlice(t, []mat.Float{
		0.13644764, 0.1020883, 0.17371385, 0.41388524, 0.6401866, 0.8875985, 1.4471399, 2.3320909,
	}, ys[1].Value().Data(), 0.001)
}
