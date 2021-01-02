// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fofe

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"testing"
)

func TestDecode(t *testing.T) {
	reverseVocabulary := map[int]string{0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

	z := mat.NewSparseFromMap(len(reverseVocabulary), 1, map[mat.Coordinate]mat.Float{
		{I: 0, J: 0}: 1.00781250,
		{I: 2, J: 0}: 0.51562500,
		{I: 3, J: 0}: 0.28125000,
		{I: 4, J: 0}: 0.18750000,
	})

	decoding := Decode(0.5, z)

	var xs string
	for _, id := range decoding {
		c, _ := reverseVocabulary[id]
		xs += c
	}

	if xs != "acdeedca" {
		t.Errorf("The decoded string doesn't match the expected value.")
	}
}
