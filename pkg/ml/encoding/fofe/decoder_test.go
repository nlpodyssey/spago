// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fofe

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"testing"
)

func TestDecode(t *testing.T) {
	reverseVocabulary := []string{
		0: "a",
		1: "b",
		2: "c",
		3: "d",
		4: "e",
	}

	z := mat.NewVecDense([]mat.Float{
		0: 1.00781250,
		1: 0,
		2: 0.51562500,
		3: 0.28125000,
		4: 0.18750000,
	})

	decoding := Decode[mat.Float](0.5, z)

	var xs string
	for _, id := range decoding {
		c := reverseVocabulary[id]
		xs += c
	}

	if xs != "acdeedca" {
		t.Errorf("The decoded string doesn't match the expected value.")
	}
}
