package main

import (
	"fmt"
	"log"
	"math"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/initializers"
	"github.com/nlpodyssey/spago/losses"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/optimizers"
	"github.com/nlpodyssey/spago/optimizers/adam"
)

// SineModel defines a simple neural network for sine function approximation
type SineModel struct {
	nn.Module
	Layers nn.ModuleList[nn.StandardModel]
}

// NewSineModel creates a new model for sine approximation
func NewSineModel[T float.DType]() *SineModel {
	model := &SineModel{}

	// Create layers with proper activation modules
	model.Layers = nn.ModuleList[nn.StandardModel]{
		linear.New[T](1, 64),
		activation.New(activation.ReLU),
		linear.New[T](64, 64),
		activation.New(activation.ReLU),
		linear.New[T](64, 64),
		activation.New(activation.ReLU),
		linear.New[T](64, 1),
	}

	return model
}

// Forward performs the forward pass
func (m *SineModel) Forward(xs ...mat.Tensor) []mat.Tensor {
	// ModuleList.Forward handles the sequential processing
	return m.Layers.Forward(xs...)
}

// InitRandom initializes the model weights
func (m *SineModel) InitRandom(seed uint64) *SineModel {
	r := rand.NewLockedRand(seed)

	// Initialize only the linear layers
	nn.ForEachParam(m, func(param *nn.Param) {
		initializers.XavierUniform(param.Value().(mat.Matrix), 1.0, r)
	})

	return m
}

// GenerateBatch creates a batch of training data more efficiently
func GenerateBatch[T float.DType](batchSize int, rng *rand.LockedRand) ([]mat.Tensor, []mat.Tensor) {
	// Pre-allocate arrays
	xData := make([]mat.Tensor, batchSize)
	yData := make([]mat.Tensor, batchSize)

	// Fill arrays
	for i := 0; i < batchSize; i++ {
		var x float64
		if i%2 == 0 {
			x = rng.Float64() * math.Pi
		} else {
			x = rng.Float64() * 2 * math.Pi
		}
		if i%3 == 0 {
			x = rng.Float64() * -1 * math.Pi
		}
		y := math.Cos(x)
		xData[i] = mat.Scalar(x)
		yData[i] = mat.Scalar(y)
	}

	return xData, yData
}

func main() {
	const (
		epochs    = 1000
		batchSize = 32
		seed      = 42
	)

	// Create and initialize model
	model := NewSineModel[float64]()
	model.InitRandom(seed)

	// Setup optimizer
	conf := adam.NewDefaultConfig()
	conf.StepSize = 0.0001
	optimizer := optimizers.New(nn.Parameters(model), adam.New(conf))

	// Training loop
	rng := rand.NewLockedRand(seed)
	for epoch := 0; epoch < epochs; epoch++ {
		var epochLoss float64

		// Training phase
		for b := 0; b < 100; b++ { // 100 batches per epoch
			inputs, targets := GenerateBatch[float64](batchSize, rng)

			// Forward pass
			predictions := model.Forward(inputs...)

			// Calculate loss
			loss := losses.MSESeq(predictions, targets, true)

			// Backward pass
			if err := ag.Backward(loss); err != nil {
				log.Fatal(err)
			}

			// Update weights
			if err := optimizer.Optimize(); err != nil {
				log.Fatal(err)
			}

			epochLoss += float.ValueOf[float64](loss.Value().Item())
		}

		// Print progress every 100 epochs
		if (epoch+1)%10 == 0 {
			fmt.Printf("Epoch %d: Avg Loss %.6f\n", epoch+1, epochLoss/100)

			if epochLoss/100 < 0.00001 {
				break
			}
		}
	}

	// Evaluation
	fmt.Println("\nEvaluation:")
	testInputs := []float64{-math.Pi, -math.Pi / 2, 0, math.Pi / 2, math.Pi}
	for _, x := range testInputs {
		input := mat.NewDense[float64](mat.WithBacking([]float64{x}))
		pred := model.Forward(input)
		fmt.Printf("sin(%.3f) = %.3f (predicted) vs %.3f (actual)\n",
			x, pred[0].Value().Item().F64(), math.Cos(x))
	}
}
