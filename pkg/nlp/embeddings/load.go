// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"bufio"
	"github.com/gosuri/uiprogress"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/utils"
	"log"
	"os"
	"strings"
)

// Load inserts the pre-trained embeddings into the model.
func (m *Model) Load(filename string) {
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
		data, err := floatutils.StrToFloatSlice(strVec)
		if err != nil {
			log.Fatal(err)
		}
		vector := mat.NewVecDense(data)
		m.SetEmbedding(key, vector)
		mat.ReleaseDense(vector)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}
