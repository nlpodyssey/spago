// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"golang.org/x/exp/rand"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"saientist.dev/spago/examples/progsum/internal"
	"saientist.dev/spago/pkg/ml/act"
	"saientist.dev/spago/pkg/ml/initializers"
	"saientist.dev/spago/pkg/ml/nn"
	"saientist.dev/spago/pkg/ml/nn/perceptron"
	"saientist.dev/spago/pkg/ml/nn/rec/lstm"
	"saientist.dev/spago/pkg/ml/nn/stack"
	"saientist.dev/spago/pkg/ml/optimizers/gd"
	"saientist.dev/spago/pkg/ml/optimizers/gd/adam"
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
		datasetPath = "examples/progsum/data"
	}

	hiddenSize := 100
	batchSize := 1
	epochs := 4
	rndSrc := rand.NewSource(743)

	// read dataset
	trainSet, testSet, err := internal.Load(datasetPath)
	if err != nil {
		log.Fatal("error reading 'progressive sum' data")
	}

	// new model initialized with random weights
	model := initRandom(stack.New(
		lstm.New(1, hiddenSize),
		perceptron.New(hiddenSize, 11, act.Identity), // The CrossEntropy loss doesn't require explicit Softmax activation
	), rand.NewSource(1))

	// new optimizer with an arbitrary update method
	updater := adam.New(adam.NewDefaultConfig())
	//updater := sgd.New(sgd.NewConfig(0.001, 0.9, true))
	optimizer := gd.NewOptimizer(updater, nil)
	// ad-hoc trainer
	trainer := internal.NewTrainer(model, optimizer, epochs, batchSize, false, trainSet, testSet, modelPath, rndSrc)
	trainer.Enjoy() // :)
}

// initRandom initializes the model using the Xavier (Glorot) method.
func initRandom(model *stack.Model, source rand.Source) *stack.Model {
	for i, layer := range model.Layers {
		var gain float64
		if i == len(model.Layers)-1 { // last layer
			gain = initializers.Gain(act.SoftMax)
		} else {
			gain = initializers.Gain(act.Tanh)
		}

		layer.ForEachParam(func(param *nn.Param) {
			if param.Type() == nn.Weights {
				initializers.XavierUniform(param.Value(), gain, source)
			}
		})
	}
	return model
}
