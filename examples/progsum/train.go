// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/nlpodyssey/spago/examples/progsum/internal"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/perceptron"
	"github.com/nlpodyssey/spago/pkg/ml/nn/rec/lstm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/adam"
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
		datasetPath = "examples/progsum/data"
	}

	hiddenSize := 100
	batchSize := 1
	epochs := 4
	rndGen := rand.NewLockedRand(743)

	// read dataset
	trainSet, testSet, err := internal.Load(datasetPath)
	if err != nil {
		log.Fatal("error reading 'progressive sum' data")
	}

	model := stack.New(
		lstm.New(1, hiddenSize),
		perceptron.New(hiddenSize, 11, ag.Identity), // The CrossEntropy loss doesn't require explicit Softmax activation
	)

	// initialized the new model with random weights
	initRandom(model, rndGen)

	// new optimizer with an arbitrary update method
	updater := adam.New(adam.NewDefaultConfig())
	//updater := sgd.New(sgd.NewConfig(0.001, 0.9, true))
	optimizer := gd.NewOptimizer(updater, nil)
	// ad-hoc trainer
	trainer := internal.NewTrainer(model, optimizer, epochs, batchSize, false, trainSet, testSet, modelPath, rndGen)
	trainer.Enjoy() // :)
}

// initRandom initializes the model using the Xavier (Glorot) method.
func initRandom(model *stack.Model, rndGen *rand.LockedRand) {
	for i, layer := range model.Layers {
		var gain float64
		if i == len(model.Layers)-1 { // last layer
			gain = initializers.Gain(ag.Softmax)
		} else {
			gain = initializers.Gain(ag.Tanh)
		}
		layer.ForEachParam(func(param *nn.Param) {
			if param.Type() == nn.Weights {
				initializers.XavierUniform(param.Value(), gain, rndGen)
			}
		})
	}
}
