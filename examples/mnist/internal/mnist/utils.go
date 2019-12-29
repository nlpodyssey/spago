// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mnist

import (
	"brillion.io/spago/pkg/mat"
	"brillion.io/spago/third_party/GoMNIST"
)

// Normalize converts the image to a Dense matrix, with values scaled to the range [0, 1]
func normalize(img GoMNIST.RawImage) *mat.Dense {
	data := make([]float64, 784)
	for i := 0; i < len(data); i++ {
		data[i] = float64(img[i]) / 255 // scale to the range [0, 1]
	}
	return mat.NewVecDense(data)
}
