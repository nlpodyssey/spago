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
	"fmt"
	"testing"
)

func TestReadLabelFile(t *testing.T) {
	ll, err := ReadLabelFile("data/t10k-labels-idx1-ubyte.gz")
	if err != nil {
		t.Fatalf("read (%s)", err)
	}
	if len(ll) != 10000 {
		t.Errorf("unexpected count %d", len(ll))
	}
}

func TestReadImageFile(t *testing.T) {
	nrow, ncol, imgs, err := ReadImageFile("data/t10k-images-idx3-ubyte.gz")
	if err != nil {
		t.Fatalf("read (%s)", err)
	}
	if len(imgs) != 10000 {
		t.Errorf("unexpected count %d", len(imgs))
	}
	fmt.Printf("%d images, %dx%d format\n", len(imgs), nrow, ncol)
}

func TestLoad(t *testing.T) {
	train, test, err := Load("./data")
	if err != nil {
		t.Fatalf("load (%s)", err)
	}
	println(train.Count(), test.Count())
}
