// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package data

import "github.com/saientist/spago/pkg/mat/rnd"

// GenerateBatches generates a list of batches so that the classes distribution among them is approximately the same.
// The class is given by the callback for each i-th element up to size.
// The size of each batch depends on number of classes (batchFactor * nClasses).
// Each batch consists in a list of indices.
func GenerateBatches(size, batchFactor int, class func(i int) int) [][]int {
	groupsByClass := make(map[int][]int)
	for i := 0; i < size; i++ {
		c := class(i)
		groupsByClass[c] = append(groupsByClass[c], i)
	}
	nClasses := len(groupsByClass)
	batchSize := batchFactor * nClasses
	batchList := make([][]int, 0)
	for k := 0; k < size; k++ {
		if k%batchSize == 0 {
			batchList = append(batchList, []int{})
		}
	}
	distribution := make([]float64, nClasses)
	for i := 0; i < nClasses; i++ {
		distribution[i] = float64(len(groupsByClass[i])) / float64(size)
	}
	k := 0
	for k < size {
		class := rnd.WeightedChoice(distribution)
		if len(groupsByClass[class]) > 0 {
			var exampleIndex int
			exampleIndex, groupsByClass[class] = groupsByClass[class][0], groupsByClass[class][1:] // pop
			index := k % len(batchList)
			batchList[index] = append(batchList[index], exampleIndex)
			k++
		}
	}
	return batchList
}
