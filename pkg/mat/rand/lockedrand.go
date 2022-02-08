// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat"
	"golang.org/x/exp/rand"
	"sync"
)

// LockedRand is an implementation of rand.Rand that is concurrency-safe.
// It is just a wrap of the standard rand.Rand with its operations protected by a sync.Mutex.
type LockedRand[T mat.DType] struct {
	lk sync.Mutex
	r  *rand.Rand
}

// NewLockedRand creates a new LockedRand that implements all Rand functions that is safe
// for concurrent use.
func NewLockedRand[T mat.DType](seed uint64) *LockedRand[T] {
	return &LockedRand[T]{
		r: rand.New(rand.NewSource(seed)),
	}
}

// Seed uses the provided seed value to initialize the generator to a deterministic state.
// Seed should not be called concurrently with any other Rand method.
func (lr *LockedRand[_]) Seed(seed uint64) {
	lr.lk.Lock()
	lr.r.Seed(seed)
	lr.lk.Unlock()
}

// TwoInt63 generates 2 random int64 without locking twice.
func (lr *LockedRand[_]) TwoInt63() (n1, n2 int64) {
	lr.lk.Lock()
	n1 = lr.r.Int63()
	n2 = lr.r.Int63()
	lr.lk.Unlock()
	return
}

// Int63 returns a non-negative pseudo-random 63-bit integer as an int64.
func (lr *LockedRand[_]) Int63() (n int64) {
	lr.lk.Lock()
	n = lr.r.Int63()
	lr.lk.Unlock()
	return
}

// Uint32 returns a pseudo-random 32-bit value as a uint32.
func (lr *LockedRand[_]) Uint32() (n uint32) {
	lr.lk.Lock()
	n = lr.r.Uint32()
	lr.lk.Unlock()
	return
}

// Uint64 returns a pseudo-random 64-bit value as a uint64.
func (lr *LockedRand[_]) Uint64() (n uint64) {
	lr.lk.Lock()
	n = lr.r.Uint64()
	lr.lk.Unlock()
	return
}

// Uint64n returns, as a uint64, a pseudo-random number in [0,n).
// It is guaranteed more uniform than taking a Source value mod n
// for any n that is not a power of 2.
func (lr *LockedRand[_]) Uint64n(n uint64) uint64 {
	lr.lk.Lock()
	n = lr.r.Uint64n(n)
	lr.lk.Unlock()
	return n
}

// Int31 returns a non-negative pseudo-random 31-bit integer as an int32.
func (lr *LockedRand[_]) Int31() (n int32) {
	lr.lk.Lock()
	n = lr.r.Int31()
	lr.lk.Unlock()
	return
}

// Int returns a non-negative pseudo-random int.
func (lr *LockedRand[_]) Int() (n int) {
	lr.lk.Lock()
	n = lr.r.Int()
	lr.lk.Unlock()
	return
}

// Int63n returns, as an int64, a non-negative pseudo-random number in [0,n).
// It panics if n <= 0.
func (lr *LockedRand[_]) Int63n(n int64) (r int64) {
	lr.lk.Lock()
	r = lr.r.Int63n(n)
	lr.lk.Unlock()
	return
}

// Int31n returns, as an int32, a non-negative pseudo-random number in [0,n).
// It panics if n <= 0.
func (lr *LockedRand[_]) Int31n(n int32) (r int32) {
	lr.lk.Lock()
	r = lr.r.Int31n(n)
	lr.lk.Unlock()
	return
}

// Intn returns, as an int, a non-negative pseudo-random number in [0,n).
// It panics if n <= 0.
func (lr *LockedRand[_]) Intn(n int) (r int) {
	lr.lk.Lock()
	r = lr.r.Intn(n)
	lr.lk.Unlock()
	return
}

// NormFloat returns a normally distributed value T in the range
// [-math.MaxFloat(T), +math.MaxFloat(T)] with
// standard normal distribution (mean = 0, stddev = 1).
func (lr *LockedRand[T]) NormFloat() (n T) {
	lr.lk.Lock()
	n = T(lr.r.NormFloat64())
	lr.lk.Unlock()
	return
}

// Float returns, as a T, a pseudo-random number in [0.0,1.0).
func (lr *LockedRand[T]) Float() (n T) {
	lr.lk.Lock()
	switch any(T(0)).(type) {
	case float32:
		n = T(lr.r.Float32())
	case float64:
		n = T(lr.r.Float64())
	default:
		panic(fmt.Sprintf("rand: unexpected type %T", T(0)))
	}
	lr.lk.Unlock()
	return
}

// Perm returns, as a slice of n ints, a pseudo-random permutation of the integers [0,n).
func (lr *LockedRand[_]) Perm(n int) (r []int) {
	lr.lk.Lock()
	r = lr.r.Perm(n)
	lr.lk.Unlock()
	return
}

// Read generates len(p) random bytes and writes them into p. It
// always returns len(p) and a nil error.
// Read should not be called concurrently with any other Rand method.
func (lr *LockedRand[_]) Read(p []byte) (n int, err error) {
	lr.lk.Lock()
	n, err = lr.r.Read(p)
	lr.lk.Unlock()
	return
}

// Shuffle pseudo-randomizes the order of elements using the default Source.
// n is the number of elements. Shuffle panics if n < 0.
// swap swaps the elements with indexes i and j.
func (lr *LockedRand[_]) Shuffle(i int, swap func(i int, j int)) {
	lr.lk.Lock()
	lr.r.Shuffle(i, swap)
	lr.lk.Unlock()
}
