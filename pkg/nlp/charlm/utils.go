// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"golang.org/x/exp/rand"
)

func splitByRune(str string) []string {
	out := make([]string, 0)
	for _, item := range str {
		out = append(out, string(item))
	}
	return out
}

// sample extracts the next character from the probability multinomial distribution.
// Note that the softmax must NOT have been applied to the prediction values.
func sample(prediction []float64, temperature float64) int {
	for i := range prediction {
		prediction[i] *= 1.0 / temperature
	}
	prediction = f64utils.SoftMax(prediction)
	p := rand.Float64() // TODO: use a local random generator?
	for i, x := range prediction {
		p -= x
		if p < 0 {
			return i
		}
	}
	return 0 // TODO: should panic here?
}
