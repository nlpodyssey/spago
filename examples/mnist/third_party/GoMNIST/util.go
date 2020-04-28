// Copyright 2013 Petar Maymounkov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package GoMNIST

import (
	"path"
)

// Set represents a data set of image-label pairs held in memory
type Set struct {
	NRow   int
	NCol   int
	Images []RawImage
	Labels []Label
}

// ReadSet reads a set from the images file iname and the corresponding labels file lname
func ReadSet(iname, lname string) (set *Set, err error) {
	set = &Set{}
	if set.NRow, set.NCol, set.Images, err = ReadImageFile(iname); err != nil {
		return nil, err
	}
	if set.Labels, err = ReadLabelFile(lname); err != nil {
		return nil, err
	}
	return
}

// Count returns the number of points available in the data set
func (s *Set) Count() int {
	return len(s.Images)
}

// Get returns the i-th image and its corresponding label
func (s *Set) Get(i int) (RawImage, Label) {
	return s.Images[i], s.Labels[i]
}

// Sweeper is an iterator over the points in a data set
type Sweeper struct {
	set *Set
	i   int
}

// Next returns the next image and its label in the data set.
// If the end is reached, present is set to false.
func (sw *Sweeper) Next() (image RawImage, label Label, present bool) {
	if sw.i >= len(sw.set.Images) {
		return nil, 0, false
	}
	return sw.set.Images[sw.i], sw.set.Labels[sw.i], true
}

// Sweep creates a new sweep iterator over the data set
func (s *Set) Sweep() *Sweeper {
	return &Sweeper{set: s}
}

// Load reads both the training and the testing MNIST data sets, given
// a local directory dir, containing the MNIST distribution files.
func Load(dir string) (train, test *Set, err error) {
	if train, err = ReadSet(path.Join(dir, "train-images-idx3-ubyte.gz"), path.Join(dir, "train-labels-idx1-ubyte.gz")); err != nil {
		return nil, nil, err
	}
	if test, err = ReadSet(path.Join(dir, "t10k-images-idx3-ubyte.gz"), path.Join(dir, "t10k-labels-idx1-ubyte.gz")); err != nil {
		return nil, nil, err
	}
	return
}
