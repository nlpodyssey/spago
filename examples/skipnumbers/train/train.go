// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/nlpodyssey/spago/examples/skipnumbers/skipnumbers"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/rec/lstmsc"
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
		datasetPath = "examples/skipnumbers/data"
	}

	hiddenSize := 200
	batchSize := 1
	epochs := 10

	// read dataset
	trainSet, testSet, err := skipnumbers.Load(datasetPath)
	if err != nil {
		log.Fatal("error during data-set reading")
	}

	// new model
	model := skipnumbers.NewModel(
		lstmsc.New(
			10,         // in
			hiddenSize, // out
			10,         // k,
			0.5,        // lambda,
			50,         // intermediate layer
		),
		linear.New(hiddenSize, 10), // The CrossEntropy loss doesn't require explicit Softmax activation
	)

	// initialize model with random weights
	model.Init()

	// new optimizer with an arbitrary update method
	updater := adam.New(adam.NewDefaultConfig())
	//updater := sgd.New(sgd.NewConfig(0.001, 0.9, true))
	optimizer := gd.NewOptimizer(updater)
	// ad-hoc trainer
	trainer := skipnumbers.NewTrainer(model, optimizer, epochs, batchSize, false, trainSet, testSet, modelPath, 42)
	trainer.Enjoy() // :)
}
