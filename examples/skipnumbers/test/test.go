// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"github.com/nlpodyssey/spago/examples/skipnumbers/skipnumbers"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn/perceptron"
	"github.com/nlpodyssey/spago/pkg/ml/nn/rec/lstmsc"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
	_ "net/http/pprof"
	"os"
)

func main() {

	modelPath := os.Args[1]

	var datasetPath string
	if len(os.Args) > 2 {
		datasetPath = os.Args[2]
	} else {
		// assuming default path
		datasetPath = "examples/skipnumbers/data"
	}

	// read dataset
	_, testSet, err := skipnumbers.Load(datasetPath)
	if err != nil {
		log.Fatal("error during data-set reading")
	}

	// Warning: use the same model and hyper-params as for the training phase
	hiddenSize := 200
	model := skipnumbers.NewModel(
		lstmsc.New(10, hiddenSize, 10, 0.5, 50),
		perceptron.New(hiddenSize, 10, ag.Softmax),
	)

	err = utils.DeserializeFromFile(modelPath, model)
	if err != nil {
		panic("skipnumbers: error during model deserialization.")
	}

	fmt.Println("Start evaluation...")
	precision := skipnumbers.NewEvaluator(model).Evaluate(testSet).Precision()
	fmt.Printf("Accuracy: %.2f\n", 100*precision)
}
