// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"bufio"
	"log"
	"os"
	"strings"

	"github.com/gosuri/uiprogress"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/matutils"
	"github.com/nlpodyssey/spago/utils"
)

// Load inserts the pre-trained embeddings into the model.
func (m *Model[T]) Load(filename string) {
	count, err := utils.CountLines(filename)
	if err != nil {
		log.Fatal(err)
	}

	uip := uiprogress.New()
	bar := uip.AddBar(count)
	bar.AppendCompleted().PrependElapsed()
	uip.Start() // start bar rendering
	defer uip.Stop()

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	lineCount := 0
	for scanner.Scan() {
		lineCount++
		bar.Incr()
		line := strings.Trim(scanner.Text(), " ")
		if lineCount == 1 && strings.Count(line, " ") == 1 {
			// TODO: use the header information
			continue // skip header
		}
		key := utils.BeforeSpace(line)
		strVec := utils.AfterSpace(line)
		data, err := matutils.StrToFloatSlice[T](strVec)
		if err != nil {
			log.Fatal(err)
		}
		vector := mat.NewVecDense[T](data)
		m.SetEmbedding(key, vector)
		mat.ReleaseDense(vector)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}
