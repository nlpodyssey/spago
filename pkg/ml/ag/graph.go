// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"github.com/nlpodyssey/spago/pkg/utils/processingqueue"
	"log"
	"runtime"
	"sync"
)

// The Graph a.k.a. expression graph or computational graph is the centerpiece of the spaGO machine learning framework.
// It takes the form of a directed graph with no directed cycles (DAG).
type Graph struct {
	// to avoid data race during concurrent computations (mu2 is used in Constant())
	mu, mu2 sync.Mutex
	// maxID is the id of the last inserted node (corresponds of len(nodes)-1)
	maxID int
	// the time-step is useful to perform truncated back propagation (default 0)
	curTimeStep int
	// nodes contains the list of nodes of the graph. The indices of the list are the nodes ids.
	nodes []Node
	// constants maps scalar values that that doesn't require gradients to a Node. It is used in the Constant() method.
	constants map[mat.Float]Node
	// IncrementalForward sets whether to compute the forward during the graph definition (default true).
	incrementalForward bool
	// cache of the support structures created during the last groupNodesByHeight() computation.
	// Before using it you have to check if the maxID of the graph matches the maxID of the cache.
	// Otherwise the cache must be invalidated and the values recalculated.
	cache struct {
		// the maxID when this cache was created.
		maxID int
		// nodes grouped by height
		nodesByHeight [][]Node
		// the nodes height. The index corresponds to the node ID.
		height []int
	}
	// randGen is the generator of random numbers
	randGen *rand.LockedRand
	// processingQueue allows proper handling for computationally heavy operations
	// such as forward and backward steps.
	// The default size is defaultProcessingQueueSize.
	processingQueue processingqueue.ProcessingQueue
}

// defaultProcessingQueueSize is the default size of Graph.processingQueue on a new Graph.
var defaultProcessingQueueSize = runtime.NumCPU()

// GraphOption allows to configure a new Graph with your specific needs.
type GraphOption func(*Graph)

// Rand sets the generator of random numbers.
func Rand(rand *rand.LockedRand) GraphOption {
	return func(g *Graph) {
		g.randGen = rand
	}
}

// RandSeed set a new generator of random numbers with the given seed.
func RandSeed(seed uint64) GraphOption {
	return func(g *Graph) {
		g.randGen = rand.NewLockedRand(seed)
	}
}

// IncrementalForward sets whether to compute the forward during the graph definition (default true).
// When enabled it lets you access to the Value() resulting from the computation.
// There are particular cases where you don't need intermediate values and computing the forward after
// the graph definition can be more efficient though.
func IncrementalForward(value bool) GraphOption {
	return func(g *Graph) {
		g.incrementalForward = value
	}
}

// ConcurrentComputations sets the maximum number of concurrent computations handled by the Graph
// for heavy tasks such as forward and backward steps.
// The value 1 corresponds to sequential execution.
func ConcurrentComputations(value int) GraphOption {
	if value < 1 {
		panic("ag: ConcurrentComputations value must be greater than zero")
	}
	return func(g *Graph) {
		g.processingQueue = processingqueue.New(value)
	}
}

// NewGraph returns a new initialized graph.
// It can take an optional random generator of type rand.Rand.
func NewGraph(opts ...GraphOption) *Graph {
	g := &Graph{
		maxID:              -1,
		curTimeStep:        0,
		nodes:              nil,
		constants:          map[mat.Float]Node{},
		incrementalForward: true,
		processingQueue:    processingqueue.New(defaultProcessingQueueSize),
	}
	g.clearCache()
	for _, opt := range opts {
		opt(g)
	}
	if g.randGen == nil {
		g.randGen = rand.NewLockedRand(1) // set default random generator
	}
	return g
}

// IncrementalForwardEnabled returns whether the computation happens during the graph definition.
// See ag.IncrementalForward() option.
func (g *Graph) IncrementalForwardEnabled() bool {
	return g.incrementalForward
}

// Clear cleans the graph. This is a destructive operation.
// It is not mandatory to call this method, but it is strongly recommended to do so when you finish using the graph.
// The cleaning of the graph improves the memory management and therefore the efficiency of execution.
// Clear releases the matrices underlying the nodes so to reduce the need of future new time-consuming allocations.
// It is important to stress that calling g.Clean(), the "value" and "grad" of the operators nodes are freed (set to nil).
// Whoever is using the Value() or Grad() properties of a node, does so at his own risk. It is therefore recommended to
// make always a copy of the return value of Value() or Grad().
// Alternatively, you can use the convenient graph's methods g.GetCopiedValue(node) and g.GetCopiedGrad(node).
func (g *Graph) Clear() {
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
		if node, ok := node.(*Operator); ok {
			*node = Operator{}
			operatorPool.Put(node)
		}
	}

	g.nodes = nil
}

// clearCache cleans the cache.
func (g *Graph) clearCache() {
	g.cache.maxID = -1
	g.cache.nodesByHeight = nil
	g.cache.height = nil
}

// ClearForReuse does the same thing as Clear(), with the difference that the graph structure (i.e.
// how nodes are connected to each other) is maintained.
// This allows you to efficiently use the graph as if it were "pre-computed" (see the ForwardAll()
// method for this usage).
func (g *Graph) ClearForReuse() {
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
func (g *Graph) releaseMemory() {
	for _, node := range g.nodes {
		if node, ok := node.(*Operator); ok {
			g.releaseValue(node)
			g.releaseGrad(node)
		}
	}
}

// releaseValue set the node value to nil release the memory.
func (g *Graph) releaseValue(node *Operator) {
	if node.value == nil {
		return
	}
	mat.ReleaseMatrix(node.value)
	node.value = nil
}

// releaseGrad set the node gradient to nil and release the memory.
func (g *Graph) releaseGrad(node *Operator) {
	node.ZeroGrad()
}

// ZeroGrad sets the gradients of all nodes to zero.
func (g *Graph) ZeroGrad() {
	for _, node := range g.nodes {
		node.ZeroGrad()
	}
}

// NewVariable creates and returns a new node.
func (g *Graph) NewVariable(value mat.Matrix, requiresGrad bool) Node {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &Variable{
		graph:        g,
		timeStep:     g.curTimeStep,
		id:           g.newID(),
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requiresGrad,
	}
	// the new ID is sequential so it corresponds to the index in g.nodes
	g.nodes = append(g.nodes, newNode)
	return newNode
}

// NewVariableWithName creates and returns a new node.
func (g *Graph) NewVariableWithName(value mat.Matrix, requiresGrad bool, name string) Node {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &Variable{
		graph:        g,
		timeStep:     g.curTimeStep,
		id:           g.newID(),
		name:         name,
		value:        value,
		grad:         nil,
		hasGrad:      false,
		requiresGrad: requiresGrad,
	}
	// the new ID is sequential so it corresponds to the index in g.nodes
	g.nodes = append(g.nodes, newNode)
	return newNode
}

// NewScalar creates a variable node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func (g *Graph) NewScalar(value mat.Float) Node {
	return g.NewVariable(mat.NewScalar(value), false)
}

// NewScalarWithName creates a variable node that doesn't require gradients.
// TODO: Why shouldn't gradient be required by default?
func (g *Graph) NewScalarWithName(value mat.Float, name string) Node {
	return g.NewVariableWithName(mat.NewScalar(value), false, name)
}

// Constant returns a scalar Node that that doesn't require gradients.
// For the same value, a previously created Node is returned without creating a new one.
// Useful for example in the case of epsilon and number like 0.0 or 1.0.
func (g *Graph) Constant(value mat.Float) Node {
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
func (g *Graph) NewOperator(f fn.Function, operands ...Node) Node {
	for _, o := range operands {
		if o.Graph() != g {
			panic("ag: operations cannot be executed among nodes of different graphs. " +
				"You may consider wrapping the nodes you need with NewWrap().")
		}
	}
	var value mat.Matrix = nil
	if g.incrementalForward {
		// the calculation is out of the lock so it can run concurrently with other operators
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

	newNode := operatorPool.Get().(*Operator)

	g.mu.Lock()
	defer g.mu.Unlock()

	*newNode = Operator{
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

	// the new ID is sequential so it corresponds to the index in g.nodes
	g.nodes = append(g.nodes, newNode)
	return newNode
}

// NewWrap creates a new wrapper Node for the given value, attaching it to
// the graph.
func (g *Graph) NewWrap(value GradValue) Node {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &Wrapper{
		GradValue: value,
		timeStep:  g.curTimeStep,
		graph:     g,
		id:        g.newID(),
		wrapGrad:  true,
	}
	// the new ID is sequential so it corresponds to the index in g.nodes
	g.nodes = append(g.nodes, newNode)
	return newNode
}

// NewWrapNoGrad is similar to NewWrap, but it disables automatic
// differentiation on the new node.
func (g *Graph) NewWrapNoGrad(value GradValue) Node {
	g.mu.Lock()
	defer g.mu.Unlock()
	newNode := &Wrapper{
		GradValue: value,
		graph:     g,
		timeStep:  g.curTimeStep,
		id:        g.newID(),
		wrapGrad:  false,
	}
	// the new ID is sequential so it corresponds to the index in g.nodes
	g.nodes = append(g.nodes, newNode)
	return newNode
}

// ForwardOption allows to adapt the Forward() to your specific needs.
type ForwardOption func(*forwardHandler)

// Range allows you to limit the forward computation within a time-step range.
// By default, the forward computes from the first node at time-step 0 to the last node at the current time-step.
func Range(fromTimeStep, toTimeStep int) ForwardOption {
	if fromTimeStep < 0 {
		log.Fatalf("ag: expected fromTimeStep equal to or greater than zero. Found %d.", fromTimeStep)
	}
	if toTimeStep > -1 && toTimeStep < fromTimeStep {
		log.Fatalf("ag: expected toTimeStep equal to or greater than `%d` (fromTimeStep). Found `%d`.",
			fromTimeStep, toTimeStep)
	}
	return func(f *forwardHandler) {
		f.fromTimeStep = fromTimeStep
		f.toTimeStep = toTimeStep
	}
}

// Forward computes the results of the entire Graph.
// Usually you don't need to execute Forward() manually in the define-by-run configuration (default).
// If you do, all values will be recalculated. You can also choose through the Range option to recalculate only a portion of nodes.
// Instead, it is required to obtain the value of the nodes in case the Graph has been created with IncrementalForward(false).
func (g *Graph) Forward(opts ...ForwardOption) {
	handler := &forwardHandler{
		g:            g,
		fromTimeStep: 0,
		toTimeStep:   -1, // unlimited
	}
	for _, opt := range opts {
		opt(handler)
	}

	// Free the values that are about to be recalculated so that memory is not wasted
	for _, node := range g.nodes {
		if op, ok := node.(*Operator); ok {
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
type BackwardOption func(*backwardHandler)

// Truncate is an option that sets the number of back steps for the
// Truncated Back-Propagation.
func Truncate(backSteps int) BackwardOption {
	return func(f *backwardHandler) {
		f.stopAtTimeStep = f.node.TimeStep() - backSteps
	}
}

// OutputGrad is an option that sets the output gradients which are the starting
// point for the back-propagation (Backward).
func OutputGrad(grad mat.Matrix) BackwardOption {
	return func(f *backwardHandler) {
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
func (g *Graph) Backward(node Node, opts ...BackwardOption) {
	if node.Graph() != g {
		panic("ag: backward cannot be executed among nodes of different graphs")
	}

	handler := &backwardHandler{
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
func (g *Graph) BackwardAll() {
	handler := &backwardHandler{
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
func (g *Graph) GetCopiedValue(node Node) mat.Matrix {
	if node.Value() == nil {
		return nil
	}
	return node.Value().Clone()
}

// GetCopiedGrad returns a copy of the gradients of a Node. If the gradients are nil, GetCopiedGrad returns nil as well.
// The returned value is a copy, so it is safe to use even after the graph has been cleared calling Graph.Clear().
// It is important to remember that the Grad() property of a Node is a weak access, as the matrix derived from
// graph's operations can be freed.
func (g *Graph) GetCopiedGrad(node Node) mat.Matrix {
	if node.Grad() == nil {
		return nil
	}
	return node.Grad().Clone()
}

// ReplaceValue replaces the current value of a variable Node with the given value.
// It panics if node is not a variable.
func (g *Graph) ReplaceValue(node Node, value mat.Matrix) {
	if node, ok := node.(*Variable); !ok {
		panic("ag: invalid node. Only variables are allowed to change their value.")
	} else {
		node.value = value
	}
}

// IncTimeStep increments the value of the graph's TimeStep by one.
func (g *Graph) IncTimeStep() {
	g.curTimeStep++
}

// TimeStep is an integer value associated with the graph, which can be useful
// to perform truncated back propagation. This value is 0 for a new Graph, and
// can be incremented calling IncTimeStep.
func (g *Graph) TimeStep() int {
	return int(g.curTimeStep)
}

// Nodes returns the nodes of the graph.
func (g *Graph) Nodes() []Node {
	return g.nodes
}

// ConcurrentComputations returns the maximum number of concurrent computations handled by the Graph
// for heavy tasks such as forward and backward steps.
func (g *Graph) ConcurrentComputations() int {
	return g.processingQueue.Size()
}

// newID generates and returns a new incremental sequential ID.
func (g *Graph) newID() int {
	g.maxID++
	return g.maxID
}

func (g *Graph) groupNodesByHeight() [][]Node {
	if g.cache.maxID == g.maxID {
		return g.cache.nodesByHeight
	}
	groups := g.cache.nodesByHeight
	height := make([]int, len(g.nodes))
	copy(height[:len(g.cache.height)], g.cache.height)

	startIndex := g.cache.maxID + 1
	for _, node := range g.nodes[startIndex:] {
		h := 0
		if node, ok := node.(*Operator); ok {
			for _, operand := range node.operands {
				if operand, ok := operand.(*Operator); ok {
					if height[operand.id] >= h {
						h = height[operand.id] + 1
					}
				}
			}
		}
		height[node.ID()] = h
		if h == len(groups) {
			groups = append(groups, make([]Node, 0, 1))
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
func (g *Graph) MarshalBinary() ([]byte, error) {
	return []byte{}, nil
}

// UnmarshalBinary satisfies encoding.BinaryUnmarshaler interface.
func (g *Graph) UnmarshalBinary(data []byte) error {
	return nil
}
