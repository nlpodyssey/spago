// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mnist

import (
	"github.com/saientist/spago/pkg/mat"
	"github.com/saientist/spago/third_party/GoMNIST"
)

type Dataset struct {
	*GoMNIST.Set
	FeaturesAsVector bool
}

// The image features and its corresponding label
type Example struct {
	Features *mat.Dense
	Label    int
}

// Get returns the i-th normalized examples
func (s *Dataset) GetExample(i int) *Example {
	img := normalize(s.Images[i])
	label := s.Labels[i]
	if s.FeaturesAsVector {
		return &Example{
			Features: img,
			Label:    int(label),
		}
	} else {
		return &Example{
			Features: img.View(28, 28),
			Label:    int(label),
		}
	}
}

func GetAllExamples(dataset *GoMNIST.Set) []*Example {
	examples := make([]*Example, dataset.Count())
	for i := 0; i < dataset.Count(); i++ {
		img, label := dataset.Get(i)
		examples[i] = &Example{
			Features: normalize(img),
			Label:    int(label),
		}
	}
	return examples
}

// normalize converts the image to a Dense matrix, with values scaled to the range [0, 1]
func normalize(img GoMNIST.RawImage) *mat.Dense {
	data := make([]float64, 784)
	for i := 0; i < len(data); i++ {
		data[i] = float64(img[i]) / 255 // scale to the range [0, 1]
	}
	return mat.NewVecDense(data)
}
