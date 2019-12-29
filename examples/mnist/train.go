// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"brillion.io/spago/examples/mnist/internal/mnist"
	"brillion.io/spago/pkg/ml/act"
	"brillion.io/spago/pkg/ml/initializers"
	"brillion.io/spago/pkg/ml/nn"
	"brillion.io/spago/pkg/ml/nn/perceptron"
	"brillion.io/spago/pkg/ml/nn/stack"
	"brillion.io/spago/pkg/ml/optimizers/gd"
	"brillion.io/spago/pkg/ml/optimizers/gd/adam"
	"brillion.io/spago/third_party/GoMNIST"
	"golang.org/x/exp/rand"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
)

func main() {
	// go tool pprof http://localhost:6060/debug/pprof/profile
	go func() { log.Println(http.ListenAndServe("localhost:6060", nil)) }()

	modelPath := os.Args[1]

	var datasetPath string
	if len(os.Args) > 2 {
		datasetPath = os.Args[2]
	} else {
		// assuming default path
		datasetPath = "third_party/GoMNIST/data"
	}

	hiddenSize := 100
	batchSize := 50
	epochs := 20
	rndSrc := rand.NewSource(743)

	// read dataset
	trainSet, testSet, err := GoMNIST.Load(datasetPath)
	if err != nil {
		panic("Error reading MNIST data.")
	}

	// new model initialized with random weights
	model := initRandom(NewMLP(784, hiddenSize, 10), rand.NewSource(1))

	// new optimizer with an arbitrary update method
	//updater := sgd.New(sgd.NewConfig(0.1, 0.0, false)) // sgd
	//updater := sgd.New(sgd.NewConfig(0.1, 0.9, true))  // sgd with nesterov momentum
	updater := adam.New(adam.NewDefaultConfig())
	optimizer := gd.NewOptimizer(updater, nil)
	// ad-hoc trainer
	trainer := mnist.NewTrainer(model, optimizer, epochs, batchSize, true, trainSet, testSet, modelPath, rndSrc)
	trainer.Enjoy() // :)
}

func NewMLP(in, hidden, out int) *stack.Model {
	return stack.New(
		perceptron.New(in, hidden, act.ReLU),
		perceptron.New(hidden, out, act.Identity), // The CrossEntropy loss doesn't require explicit Softmax activation
	)
}

// initRandom initializes the model using the Xavier (Glorot) method.
func initRandom(model *stack.Model, source rand.Source) *stack.Model {
	for i, layer := range model.Layers {
		var gain float64
		if i == len(model.Layers)-1 { // last layer
			gain = initializers.Gain(act.SoftMax)
		} else {
			gain = initializers.Gain(act.ReLU)
		}
		layer.ForEachParam(func(param *nn.Param) {
			if param.Type() == nn.Weights {
				initializers.XavierUniform(param.Value(), gain, source)
			}
		})
	}
	return model
}
