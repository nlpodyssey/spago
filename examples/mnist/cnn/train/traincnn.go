// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/saientist/spago/examples/mnist/internal/mnist"
	"github.com/saientist/spago/pkg/ml/ag"
	"github.com/saientist/spago/pkg/ml/optimizers/gd"
	"github.com/saientist/spago/pkg/ml/optimizers/gd/adam"
	"github.com/saientist/spago/third_party/GoMNIST"
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

	batchSize := 50
	epochs := 20
	rndSrc := rand.NewSource(743)

	// read dataset
	trainSet, testSet, err := GoMNIST.Load(datasetPath)
	if err != nil {
		panic("Error reading MNIST data.")
	}

	// new model initialized with random weights
	model := mnist.NewCNN(
		9,   // kernelSizeX
		9,   // kernelSizeY
		1,   // inputChannels
		10,  // outputChannels
		5,   // maxPoolingRows
		5,   // maxPoolingCols
		160, // hidden
		10,  // out
		ag.ReLU,
		ag.Identity, // The CrossEntropy loss doesn't require explicit Softmax activation
	)
	mnist.InitCNN(model, rand.NewSource(1))

	trainer := mnist.NewTrainer(
		model,
		gd.NewOptimizer(adam.New(adam.NewDefaultConfig()), nil),
		epochs,
		batchSize,
		true,
		mnist.Dataset{Set: trainSet, FeaturesAsVector: false},
		mnist.Dataset{Set: testSet, FeaturesAsVector: false},
		modelPath,
		rndSrc,
	)
	trainer.Enjoy() // :)
}
