// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/utils/processingqueue"
	"log"
	"runtime"
	"sync"
)

// ProcessingMode regulates the different usage of some operations (e.g. Dropout, BatchNorm, etc.),
// depending on whether you're doing training or inference.
// Failing to set the right mode will yield inconsistent inference results.
type ProcessingMode uint8

const (
	// Training is to be used during the training phase of a model. For example, dropouts are enabled.
	Training ProcessingMode = iota
	// Inference keeps weights fixed while using the model and disables some operations (e.g. skip dropout).
	Inference
)

// The Graph a.k.a. expression graph or computational graph is the centerpiece of the spaGO machine learning framework.
// It takes the form of a directed graph with no directed cycles (DAG).
type Graph[T mat.DType] struct {
	// to avoid data race during concurrent computations (mu2 is used in Constant())
	mu, mu2 sync.Mutex
	// maxID is the id of the last inserted node (corresponds of len(nodes)-1)
	maxID int
	// the time-step is useful to perform truncated back propagation (default 0)
	curTimeStep int
	// nodes contains the list of nodes of the graph. The indices of the list are the nodes ids.
	// The nodes are inserted one at a time in order of creation.
	nodes []Node[T]
	// constants maps scalar values that that doesn't require gradients to a Node. It is used in the Constant() method.
	constants map[T]Node[T]
	// incrementalForward reports whether to compute the forward during the graph definition.
	incrementalForward bool
	// cache of the support structures created during the last groupNodesByHeight() computation.
	// Before using it you have to check if the maxID of the graph matches the maxID of the cache.
	// Otherwise, the cache must be invalidated and the values recalculated.
	cache struct {
		// the maxID when this cache was created.
		maxID int
		// nodes grouped by height
		nodesByHeight [][]Node[T]
		// the nodes height. The index corresponds to the node ID.
		height []int
	}
	// mode defines whether the graph is being used in training or inference (default inference).
	mode ProcessingMode
	// randGen is the generator of random numbers
	randGen *rand.LockedRand[T]
	// processingQueue allows proper handling for computationally heavy operations
	// such as forward and backward steps.
	// The default size is defaultProcessingQueueSize.
	processingQueue processingqueue.ProcessingQueue
}

// defaultProcessingQueueSize is the default size of Graph.processingQueue on a new Graph.
var defaultProcessingQueueSize = runtime.NumCPU()

// GraphOption allows to configure a new Graph with your specific needs.
type GraphOption[T mat.DType] func(*Graph[T])

// WithRand sets the generator of random numbers.
func WithRand[T mat.DType](rand *rand.LockedRand[T]) GraphOption[T] {
	return func(g *Graph[T]) {
		g.randGen = rand
	}
}

// WithRandSeed set a new generator of random numbers with the given seed.
func WithRandSeed[T mat.DType](seed uint64) GraphOption[T] {
	return func(g *Graph[T]) {
		g.randGen = rand.NewLockedRand[T](seed)
	}
}

// WithIncrementalForward sets whether to compute the forward during the graph definition (default true).
// When enabled it lets you access to the Value() resulting from the computation.
// There are particular cases where you don't need intermediate values and computing the forward after
// the graph definition can be more efficient though.
func WithIncrementalForward[T mat.DType](value bool) GraphOption[T] {
	return func(g *Graph[T]) {
		g.incrementalForward = value
	}
}

// WithConcurrentComputations sets the maximum number of concurrent computations handled by the Graph
// for heavy tasks such as forward and backward steps.
// The value 1 corresponds to sequential execution.
func WithConcurrentComputations[T mat.DType](value int) GraphOption[T] {
	if value < 1 {
		panic("ag: WithConcurrentComputations value must be greater than zero")
	}
	return func(g *Graph[T]) {
		g.processingQueue = processingqueue.New(value)
	}
}

// WithMode sets whether the graph is being used in training or inference.
func WithMode[T mat.DType](mode ProcessingMode) GraphOption[T] {
	return func(g *Graph[T]) {
		g.mode = mode
	}
}

// NewGraph returns a new initialized graph.
// It can take an optional random generator of type rand.WithRand.
func NewGraph[T mat.DType](opts ...GraphOption[T]) *Graph[T] {
	g := &Graph[T]{
		maxID:              -1,
		curTimeStep:        0,
		nodes:              nil,
		constants:          map[T]Node[T]{},
		incrementalForward: true,
		mode:               Inference,
		processingQueue:    processingqueue.New(defaultProcessingQueueSize),
	}
	g.clearCache()
	for _, opt := range opts {
		opt(g)
	}
	if g.randGen == nil {
		g.randGen = rand.NewLockedRand[T](1) // set default random generator
	}
	return g
}

// IncrementalForwardEnabled returns whether the computation happens during the graph definition.
// See ag.WithIncrementalForward() option.
func (g *Graph[_]) IncrementalForwardEnabled() bool {
	return g.incrementalForward
}

// Mode returns whether the graph is being used in training or inference.
func (g *Graph[_]) Mode() ProcessingMode {
	return g.mode
}

// Clear cleans the graph. This is a destructive operation.
// It is not mandatory to call this method, but it is strongly recommended to do so when you finish using the graph.
// The cleaning of the graph improves the memory management and therefore the efficiency of execution.
// Clear releases the matrices underlying the nodes so to reduce the need of future new time-consuming allocations.
// It is important to stress that calling g.Clean(), the "value" and "grad" of the operators nodes are freed (set to nil).
// Whoever is using the Value() or Grad() properties of a node, does so at his own risk. It is therefore recommended to
// make always a copy of the return value of Value() or Grad().
// Alternatively, you can use the convenient graph's methods g.GetCopiedValue(node) and g.GetCopiedGrad(node).
func (g *Graph[T]) Clear() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.nodes == nil {
		return
	}
	g.maxID = -1
	g.curTimeStep = 0
	g.clearCache()
	g.releaseMemory()

	for _, node := range g.nodes {
		if node, ok := node.(*Operator[T]); ok {
			*node = Operator[T]{}
			getOperatorPool[T]().Put(node)
		}
	}

	g.nodes = nil
}

// clearCache cleans the cache.
func (g *Graph[_]) clearCache() {
	g.cache.maxID = -1
	g.cache.nodesByHeight = nil
	g.cache.height = nil
}

// ClearForReuse does the same thing as Clear(), with the difference that the graph structure (i.e.
// how nodes are connected to each other) is maintained.
// This allows you to efficiently use the graph as if it were "pre-computed" (see the ForwardAll()
// method for this usage).
func (g *Graph[_]) ClearForReuse() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.nodes == nil {
		return
	}
	g.releaseMemory()
}

// releaseMemory clears the values and the gradients of operator nodes.
// Since the values and the gradients within the nodes are handled through a pool of dense matrices,
// releasing them allows the memory to be reused without being reallocated, improving performance.
func (g *Graph[T]) releaseMemory() {
	for _, node := range g.nodes {
		if node, ok := node.(*Operator[T]); ok {
			g.releaseValue(node)
			g.releaseGrad(node)
		}
	}
}

// releaseValue set the node value to nil release the memory.
func (g *Graph[T]) releaseValue(node *Operator[T]) {
	if node.value == nil {
		return
	}
	mat.ReleaseMatrix(node.value)
	node.value = nil
}

// releaseGrad set the node gradient to nil and release the memory.
func (g *Graph[T]) releaseGrad(node *Operator[T]) {
	node.ZeroGrad()
}

// ZeroGrad sets the gradients of all nodes to zero.
func (g *Graph[_]) ZeroGrad() {
	for _, node := range g.nodes {
		node.ZeroGrad()
	}
}

// NewVariable creates and returns a new node.
func (g *Graph[T]) NewVariable(value mat.Matrix[T], requiresGrad bool) Node[T] {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &Variable[T]{
		graph:        g,
		timeStep:     g.curTimeStep,
		id:           g.newID(),
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requiresGrad,
	}

	g.nodes = append(g.nodes, newNode)
	return newNode
}

// NewVariableWithName creates and returns a new node.
func (g *Graph[T]) NewVariableWithName(value mat.Matrix[T], requiresGrad bool, name string) Node[T] {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &Variable[T]{
		graph:        g,
		timeStep:     g.curTimeStep,
		id:           g.newID(),
		name:         name,
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requiresGrad,
	}

	g.nodes = append(g.nodes, newNode)
	return newNode
}

// NewScalar creates a variable node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func (g *Graph[T]) NewScalar(value T) Node[T] {
	return g.NewVariable(mat.NewScalar(value), false)
}

// NewScalarWithName creates a variable node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func (g *Graph[T]) NewScalarWithName(value T, name string) Node[T] {
	return g.NewVariableWithName(mat.NewScalar(value), false, name)
}

// Constant returns a scalar Node that that doesn't require gradients.
// For the same value, a previously created Node is returned without creating a new one.
// Useful for example in the case of epsilon and number like 0.0 or 1.0.
func (g *Graph[T]) Constant(value T) Node[T] {
	g.mu2.Lock()
	defer g.mu2.Unlock()
	if node, ok := g.constants[value]; ok {
		return node
	}
	node := g.NewVariableWithName(mat.NewScalar(value), false, fmt.Sprint(value))
	g.constants[value] = node
	return node
}

// NewOperator creates a new operator along with its forward pass.
// Please note that operations must be performed among nodes belonging to the same graph; it panics otherwise.
func (g *Graph[T]) NewOperator(f fn.Function[T], operands ...Node[T]) Node[T] {
	for _, o := range operands {
		if o.Graph() != g {
			panic("ag: operations cannot be executed among nodes of different graphs. " +
				"You may consider wrapping the nodes you need with NewWrap().")
		}
	}
	var value mat.Matrix[T] = nil
	if g.incrementalForward {
		// the calculation is out of the lock, so it can run concurrently with other operators
		g.processingQueue.Run(func() {
			value = f.Forward()
		})
	}
	requiresGrad := false
	for _, operand := range operands {
		if operand.RequiresGrad() {
			requiresGrad = true
			break
		}
	}

	newNode := getOperatorPool[T]().Get().(*Operator[T])

	g.mu.Lock()
	defer g.mu.Unlock()

	*newNode = Operator[T]{
		graph:        g,
		timeStep:     g.curTimeStep,
		id:           g.newID(),
		function:     f,
		operands:     operands,
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requiresGrad,
	}

	g.nodes = append(g.nodes, newNode)
	return newNode
}

// NewWrap creates a new wrapper Node for the given value, attaching it to
// the graph.
func (g *Graph[T]) NewWrap(value GradValue[T]) Node[T] {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &Wrapper[T]{
		GradValue: value,
		timeStep:  g.curTimeStep,
		graph:     g,
		id:        g.newID(),
		wrapGrad:  true,
	}

	g.nodes = append(g.nodes, newNode)
	return newNode
}

// NewWrapNoGrad is similar to NewWrap, but it disables automatic
// differentiation on the new node.
func (g *Graph[T]) NewWrapNoGrad(value GradValue[T]) Node[T] {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &Wrapper[T]{
		GradValue: value,
		graph:     g,
		timeStep:  g.curTimeStep,
		id:        g.newID(),
		wrapGrad:  false,
	}

	g.nodes = append(g.nodes, newNode)
	return newNode
}

// ForwardOption allows to adapt the Forward() to your specific needs.
type ForwardOption[T mat.DType] func(*forwardHandler[T])

// Range allows you to limit the forward computation within a time-step range.
// By default, the forward computes from the first node at time-step 0 to the last node at the current time-step.
func Range[T mat.DType](fromTimeStep, toTimeStep int) ForwardOption[T] {
	if fromTimeStep < 0 {
		log.Fatalf("ag: expected fromTimeStep equal to or greater than zero. Found %d.", fromTimeStep)
	}
	if toTimeStep > -1 && toTimeStep < fromTimeStep {
		log.Fatalf("ag: expected toTimeStep equal to or greater than `%d` (fromTimeStep). Found `%d`.",
			fromTimeStep, toTimeStep)
	}
	return func(f *forwardHandler[T]) {
		f.fromTimeStep = fromTimeStep
		f.toTimeStep = toTimeStep
	}
}

// Forward computes the results of the entire Graph.
// Usually you don't need to execute Forward() manually in the define-by-run configuration (default).
// If you do, all values will be recalculated. You can also choose through the Range option to recalculate only a portion of nodes.
// Instead, it is required to obtain the value of the nodes in case the Graph has been created with WithIncrementalForward(false).
func (g *Graph[T]) Forward(opts ...ForwardOption[T]) {
	handler := &forwardHandler[T]{
		g:            g,
		fromTimeStep: 0,
		toTimeStep:   -1, // unlimited
	}
	for _, opt := range opts {
		opt(handler)
	}

	// Free the values that are about to be recalculated so that memory is not wasted
	for _, node := range g.nodes {
		if op, ok := node.(*Operator[T]); ok {
			if op.timeStep >= handler.fromTimeStep && (handler.toTimeStep == -1 || op.timeStep <= handler.toTimeStep) {
				g.releaseValue(op)
			}
		}
	}

	if g.processingQueue.Size() > 1 {
		handler.runConcurrent()
	} else {
		handler.runSerial()
	}
}

// BackwardOption allows to adapt the Backward() to your specific needs.
type BackwardOption[T mat.DType] func(*backwardHandler[T])

// Truncate is an option that sets the number of back steps for the
// Truncated Back-Propagation.
func Truncate[T mat.DType](backSteps int) BackwardOption[T] {
	return func(f *backwardHandler[T]) {
		f.stopAtTimeStep = f.node.TimeStep() - backSteps
	}
}

// OutputGrad is an option that sets the output gradients which are the starting
// point for the back-propagation (Backward).
func OutputGrad[T mat.DType](grad mat.Matrix[T]) BackwardOption[T] {
	return func(f *backwardHandler[T]) {
		f.outputGrad = grad
	}
}

// Backward performs the back-propagation.
// It visits each node in reverse topological order, to propagate the gradients from the given node all the way
// back to the leaf. Note that the gradients are summed to the existing ones. Unless that's what you want, make sure
// all nodes have zero gradients.
//
// The back-propagation starts from the node's output gradients, following these mutually exclusive rules:
//   a) the node has gradients (probably assigned externally via node.PropagateGrads()), use those;
//   b) the output gradients are passed through the backward options, use those;
//   c) the output gradients are automatically assigned by finding the derivative of the node with respect
//      to the node itself (dy/dy = 1).
//
// If the optional back steps are set, a Truncated Back-Propagation Through Time is carried out, that is:
// the visit ends as soon as it is encountered a node with time-step less or equal to the number of back steps.
// The TBTT can perform without the need to recalculate the values of previous nodes (Williams and Peng, 1990).
func (g *Graph[T]) Backward(node Node[T], opts ...BackwardOption[T]) {
	if node.Graph() != g {
		panic("ag: backward cannot be executed among nodes of different graphs")
	}

	handler := &backwardHandler[T]{
		g:              g,
		node:           node,
		outputGrad:     nil,
		stopAtTimeStep: -1, // no stop
	}
	for _, opt := range opts {
		opt(handler)
	}
	if !node.HasGrad() {
		handler.propagateOutputGrad()
	}
	if g.processingQueue.Size() > 1 {
		handler.runConcurrent()
	} else {
		handler.runSerial()
	}
}

// BackwardAll performs full back-propagation from the last node of the graph.
// It requires the root nodes to have assigned gradients already.
func (g *Graph[T]) BackwardAll() {
	handler := &backwardHandler[T]{
		g:              g,
		node:           g.nodes[g.maxID],
		outputGrad:     nil,
		stopAtTimeStep: -1, // no stop
	}
	if g.processingQueue.Size() > 1 {
		handler.runConcurrent()
	} else {
		handler.runSerial()
	}
}

// GetCopiedValue returns a copy of the value of a Node. If the value is nil, GetCopiedValue returns nil as well.
// The returned value is a copy, so it is safe to use even after the graph has been cleared calling Graph.Clear().
// It is important to remember that the Value() property of a Node is a weak access, as the matrix derived from
// graph's operations can be freed.
func (g *Graph[T]) GetCopiedValue(node Node[T]) mat.Matrix[T] {
	if node.Value() == nil {
		return nil
	}
	return node.Value().Clone()
}

// GetCopiedGrad returns a copy of the gradients of a Node. If the gradients are nil, GetCopiedGrad returns nil as well.
// The returned value is a copy, so it is safe to use even after the graph has been cleared calling Graph.Clear().
// It is important to remember that the Grad() property of a Node is a weak access, as the matrix derived from
// graph's operations can be freed.
func (g *Graph[T]) GetCopiedGrad(node Node[T]) mat.Matrix[T] {
	if node.Grad() == nil {
		return nil
	}
	return node.Grad().Clone()
}

// ReplaceValue replaces the current value of a variable Node with the given value.
// It panics if node is not a variable.
func (g *Graph[T]) ReplaceValue(node Node[T], value mat.Matrix[T]) {
	if node, ok := node.(*Variable[T]); !ok {
		panic("ag: invalid node. Only variables are allowed to change their value.")
	} else {
		node.value = value
	}
}

// IncTimeStep increments the value of the graph's TimeStep by one.
func (g *Graph[_]) IncTimeStep() {
	g.curTimeStep++
}

// TimeStep is an integer value associated with the graph, which can be useful
// to perform truncated back propagation. This value is 0 for a new Graph, and
// can be incremented calling IncTimeStep.
func (g *Graph[_]) TimeStep() int {
	return g.curTimeStep
}

// Nodes returns the nodes of the graph.
func (g *Graph[T]) Nodes() []Node[T] {
	return g.nodes
}

// ConcurrentComputations returns the maximum number of concurrent computations handled by the Graph
// for heavy tasks such as forward and backward steps.
func (g *Graph[_]) ConcurrentComputations() int {
	return g.processingQueue.Size()
}

// newID generates and returns a new incremental sequential ID.
func (g *Graph[_]) newID() int {
	g.maxID++
	return g.maxID
}

func (g *Graph[T]) groupNodesByHeight() [][]Node[T] {
	if g.cache.maxID == g.maxID {
		return g.cache.nodesByHeight
	}
	groups := g.cache.nodesByHeight
	height := make([]int, len(g.nodes))
	copy(height[:len(g.cache.height)], g.cache.height)

	startIndex := g.cache.maxID + 1
	for _, node := range g.nodes[startIndex:] {
		h := 0
		if node, ok := node.(*Operator[T]); ok {
			for _, operand := range node.operands {
				if operand, ok := operand.(*Operator[T]); ok {
					if height[operand.id] >= h {
						h = height[operand.id] + 1
					}
				}
			}
		}
		height[node.ID()] = h
		if h == len(groups) {
			groups = append(groups, make([]Node[T], 0, 1))
		}
		groups[h] = append(groups[h], node)
	}

	// update cache and return
	g.cache.maxID = g.maxID
	g.cache.nodesByHeight = groups
	g.cache.height = height
	return groups
}

// MarshalBinary satisfies encoding.BinaryMarshaler interface and prevents
// a Graph to be encoded to binary representation.
// This is relevant in the context of a Graph being part of a nn.Model: when
// serializing a model to binary, we want to skip the Graph, since it is part
// of the runtime context only.
func (g *Graph[_]) MarshalBinary() ([]byte, error) {
	return []byte{}, nil
}

// UnmarshalBinary satisfies encoding.BinaryUnmarshaler interface.
func (g *Graph[_]) UnmarshalBinary(_ []byte) error {
	return nil
}
