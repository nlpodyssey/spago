// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fofe

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"testing"
)

func TestDecode(t *testing.T) {
	reverseVocabulary := map[int]string{0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}
	z := mat.NewVecSparse(len(reverseVocabulary))
	z.SetVec(0, 1.00781250)
	z.SetVec(2, 0.51562500)
	z.SetVec(3, 0.28125000)
	z.SetVec(4, 0.18750000)

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
