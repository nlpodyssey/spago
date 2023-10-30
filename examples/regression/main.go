package main

import (
	"fmt"
	"log"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/initializers"
	"github.com/nlpodyssey/spago/losses"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/optimizers"
	"github.com/nlpodyssey/spago/optimizers/sgd"
)

const (
	epochs   = 100  // number of training epochs
	examples = 1000 // number of training examples
)

// Linear defines a Linear module
type Linear struct {
	nn.Module
	W *nn.Param
	B *nn.Param
}

// NewLinear creates a new Linear module with the specified input and output dimensions
func NewLinear[T float.DType](in, out int) *Linear {
	return &Linear{
		W: nn.NewParam(mat.NewDense[T](mat.WithShape(out, in))),
		B: nn.NewParam(mat.NewDense[T](mat.WithShape(out))),
	}
}

// InitRandom initializes the Linear module with random weights using the Xavier uniform distribution
func (m *Linear) InitRandom(seed uint64) *Linear {
	initializers.XavierUniform(m.W.Value().(mat.Matrix), 1.0, rand.NewLockedRand(seed))
	return m
}

// Forward applies the forward pass of the Linear module to the input x
func (m *Linear) Forward(x mat.Tensor) mat.Tensor {
	return ag.Add(ag.Mul(m.W, x), m.B)
}

type T = float64

func main() {
	m := NewLinear[T](1, 1).InitRandom(42)

	strategy := sgd.New[T](sgd.NewConfig(0.001, 0.9, true))
	optimizer := optimizers.New(nn.Parameters(m), strategy)

	normalize := func(x T) T { return x / T(examples) }
	objective := func(x T) T { return 3*x + 1 }
	criterion := losses.MSE

	learn := func(input, expected T) (T, error) {
		x, target := mat.Scalar(input), mat.Scalar(expected)
		y := m.Forward(x)
		loss := criterion(y, target, true)
		if err := ag.Backward(loss); err != nil {
			return 0, err
		}
		return float.ValueOf[T](loss.Value().Item()), nil
	}

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 0; i < examples; i++ {
			x := normalize(T(i))
			loss, err := learn(x, objective(x))
			if err != nil {
				log.Fatal(err)
			}
			if i%100 == 0 {
				fmt.Println(loss)
			}
		}
		if err := optimizer.Optimize(); err != nil {
			log.Fatal(err)
		}
	}

	fmt.Printf("\n\nTraining completed!\n\n")

	fmt.Printf("Model parameters:\n")
	fmt.Printf("W: %.2f | B: %.2f\n\n", m.W.Value().Item().F64(), m.B.Value().Item().F64())

	// -- Enable this code to save the trained model to a file --
	// fmt.Printf("Saving the trained model to the file...\n")
	// err := nn.DumpToFile(m, "model.bin")
	// if err != nil {
	// 	log.Fatal(err)
	// }
}
