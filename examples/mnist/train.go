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
	"saientist.dev/spago/examples/mnist/internal/mnist"
	"saientist.dev/spago/pkg/ml/ag"
	"saientist.dev/spago/pkg/ml/optimizers/gd"
	"saientist.dev/spago/pkg/ml/optimizers/gd/adam"
	"saientist.dev/spago/third_party/GoMNIST"
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

	batchSize := 50
	epochs := 20
	rndSrc := rand.NewSource(743)

	// read dataset
	trainSet, testSet, err := GoMNIST.Load(datasetPath)
	if err != nil {
		panic("Error reading MNIST data.")
	}

	// new model initialized with random weights
	model := mnist.NewMLP(
		784, // input
		100, // hidden
		10,  // output
		ag.ReLU,
		ag.Identity, // The CrossEntropy loss doesn't require explicit Softmax activation
	)
	mnist.InitMLP(model, rand.NewSource(1))

	// new optimizer with an arbitrary update method
	//updater := sgd.New(sgd.NewConfig(0.1, 0.0, false)) // sgd
	//updater := sgd.New(sgd.NewConfig(0.1, 0.9, true))  // sgd with nesterov momentum
	updater := adam.New(adam.NewDefaultConfig())
	optimizer := gd.NewOptimizer(updater, nil)
	// ad-hoc trainer
	trainer := mnist.NewTrainer(
		model,
		optimizer,
		epochs,
		batchSize,
		true,
		mnist.Dataset{Set: trainSet, NormalizeVec: true},
		mnist.Dataset{Set: testSet, NormalizeVec: true},
		modelPath,
		rndSrc,
	)
	trainer.Enjoy() // :)
}
