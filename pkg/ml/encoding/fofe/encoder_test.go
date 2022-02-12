// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fofe

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"testing"
)

func TestEncode(t *testing.T) {
	t.Run("float32", testEncode[float32])
	t.Run("float64", testEncode[float64])
}

func testEncode[T mat.DType](t *testing.T) {

	vocabulary := map[string]int{"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

	var xs []int
	for _, c := range "acdeedca" {
		id, _ := vocabulary[string(c)]
		xs = append(xs, id)
	}

	z := Encode[T](0.5, len(vocabulary), xs)

	gold := map[int]T{
		0: 1.00781250,
		2: 0.51562500,
		3: 0.28125000,
		4: 0.18750000,
	}

	z[len(z)-1].DoNonZero(func(i, _ int, v T) {
		if gold[i] != v {
			t.Errorf("Found %f for the id %d. Expected %f.", v, i, gold[i])
		}
	})
}

func TestBiEncode(t *testing.T) {
	t.Run("float32", testBiEncode[float32])
	t.Run("float64", testBiEncode[float64])
}

func testBiEncode[T mat.DType](t *testing.T) {

	vocabulary := map[string]int{"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}

	var xs []int
	for _, c := range "acdeedca" {
		id, _ := vocabulary[string(c)]
		xs = append(xs, id)
	}

	l2r, r2l := BiEncode[T](0.5, len(vocabulary), xs)

	gold := map[int]T{
		0: 1.00781250,
		2: 0.51562500,
		3: 0.28125000,
		4: 0.18750000,
	}

	l2r[len(l2r)-1].DoNonZero(func(i, _ int, v T) {
		if gold[i] != v {
			t.Errorf("Left to right: Found %f for the id %d. Expected %f.", v, i, gold[i])
		}
	})

	r2l[0].DoNonZero(func(i, _ int, v T) {
		if gold[i] != v {
			t.Errorf("Right to left: Found %f for the id %d. Expected %f.", v, i, gold[i])
		}
	})
}
