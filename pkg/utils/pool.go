// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package utils

// Pool is a naive implementation of a pool, based on broadcast channel.
// You can use this pool in the same way you use sync.Pool.
type Pool struct {
	pool chan interface{}

	// New optionally specifies a function to generate
	// a value when Get would otherwise return nil.
	New func() interface{}
}

// NewPool returns a new pool ready to use. The pool can contain up to a max number of items.
// Set the property New to generate new items when needed during the Get.
func NewPool(max int) *Pool {
	return &Pool{
		pool: make(chan interface{}, max),
	}
}

// Get selects an item from the Pool, removes it from the Pool, and returns it to the caller.
// If the poll is empty, Get returns the result of calling p.New.
func (p *Pool) Get() interface{} {
	var x interface{}
	select {
	case x = <-p.pool:
		return x
	default:
		return p.New()
	}
}

// Put adds x to the pool.
func (p *Pool) Put(x interface{}) {
	select {
	case p.pool <- x:
	default: // nothing to do
	}
}
