// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"bufio"
	"encoding/json"
	"os"
)

type Step struct {
	Input  float64
	Target int // progressive sum
}

type Sequence = []Step

func Load(path string) (train []Sequence, test []Sequence, err error) {
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

func load(filename string) (dataset []Sequence, err error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		items := make([][]int, 0)
		err := json.Unmarshal([]byte(line), &items)
		if err != nil {
			return nil, err
		}

		xs := make(Sequence, 0, 11)
		for _, item := range items {
			xs = append(xs, Step{
				Input:  float64(item[0]),
				Target: item[1],
			})
		}
		dataset = append(dataset, xs)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return
}
