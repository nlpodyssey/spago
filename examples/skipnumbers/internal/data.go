// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"bufio"
	"encoding/json"
	"os"
)

type Example struct {
	xs []int // input sequence
	y  int   // target label
}

func Load(path string) (train []Example, test []Example, err error) {
	train, err = load(path + "/train.jsonl")
	if err != nil {
		return nil, nil, err
	}
	test, err = load(path + "/test.jsonl")
	if err != nil {
		return nil, nil, err
	}
	return
}

func load(filename string) (dataset []Example, err error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		items := make([]int, 0)
		err := json.Unmarshal([]byte(line), &items)
		if err != nil {
			return nil, err
		}
		xs := make([]int, 0, 20)
		y := items[20]
		for i, item := range items {
			if i < 20 {
				xs = append(xs, item)
			}
		}
		dataset = append(dataset, Example{
			xs: xs,
			y:  y,
		})
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return
}
