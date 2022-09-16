# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2022-09-16

### Added
- Method `embeddings.Model.EmbeddingFast`.

## [1.0.0] - 2022-09-14

First stable release!

### Fixed
- Fix bug preventing the embeddings model from being traversed on `nn.Apply`.
- Fix incorrect use of self-attention cache when used for cross-attention.

### Changed
- Optimize implementation of some Dense matrix functions, especially on
  amd64 with AVX.

## [1.0.0-alpha] - 2022-06-14

With this release we introduce breaking changes that bring significant
improvements to the project's structure, API and performance.

It would be difficult and confusing to list every single API change. Instead, 
the following sections will broadly describe the most relevant changes,
arranged by topic.

### Project structure

Until this release, the project was essentially a monorepo in disguise: the
core packages for handling matrices and computational graphs were accompanied
by many models implementations (from the very simple up to the most
sophisticated ones) and commands (models management utilities and servers).

We now prefer to keep in this very repository only the core components of spaGO,
only enriched with an (opinionated) set of popular models and functionalities. 
Bigger sub-packages and related commands are moved to separate repositories.
The moved content includes, most notably, code related to Transformers
and Flair.
Please refer to the section **Projects using Spago** from the README for an
updated list of references to separate projects (note: some of them are still 
work in progress).
If you have the feeling that something big is missing in spaGO, chances are
it was moved to one of these separate projects: just have a look there first.  

The arrangement of packages has been simplified: there's no need anymore to
distinguish between `cmd` and `pkg`; all the main subpackages are located in
the project's root path. Similarly, many packages, previously nested under
`pkg/ml`, can now be found at root level too.

### Go version and dependencies

The minimum required Go version is `1.18`, primarily needed for the
introduction of type parameters (generics).

Thanks to the creation of separate projects, discussed above, and further
refactoring, the main set of required dependencies is limited to the ones
for testing.

Only the subpackage `embeddings/store/diskstore` requires something more, so
we defined it as "opt-in" submodule, with its own dependencies.

### float32 vs. float64

Instead of separate packages `mat32` and `mat64`, there is now a single unified
package `mat`. Many parts of the implementation make use of type parameters
(generics), however the package's public API makes a rather narrow use of them.

In particular, we abstained from adding type parameters to widely-used types,
such as the `Matrix` interface. Where suitable, we are simply favoring `float64`
values, the de-facto preferred floating point type in Go (just think about Go 
`math` package). For other situations, we introduced a new subpackage
`mat/float`. It provides simple types, holding either `float32` or `float64`
values, as scalars or slices, and makes it easy to convert values between
different precisions, all without making explicit use of generics.
This design prevents the excessive spreading of type arguments to tons of other
types that need to manipulate matrices, bot from other spaGO packages and
from your own code.

### Matrices

- The type `mat.Matrix` is the primary interface for matrices and vectors 
  throughout the project.
- The type `mat.Dense` is the concrete implementation for a dense matrix.
  Unlike the interface, it has a type argument to distinguish between `float32`
  and `float64`.
- We removed implementation and support for sparse matrices, since their
  efficacy and utility were marginal. A better implementation might come back
  in the future.
- A new dense matrix can be created "from scratch" by calling one of the several
  functions `mat.New***` (`NewDense`, `NewVecDense`, ...). Here you must choose
  which data type to use, specifying it as type parameter (unless implicit).
- Once you have an existing matrix, you can create new instances preserving
  the same data type of the initial one: simply use one of the `New***` methods
  on the matrix instance itself, rather than their top-level function
  counterparts.
- Any other operation performed on a matrix that creates a new instance will
  operate with the same type of the receiver, and returns an instance of that
  type too.
- Operations with matrices of different underlying data types are allowed, just
  beware the memory and computation overheads introduced by the necessary
  conversions.

### Auto-grad package

- The package `ag` now implicitly works in "define-by-run" mode only.
  It's way more performant compared to the previous releases, and there would 
  be no significant advantage in re-using a pre-defined graph ("define-and-run").
- There is no `Graph` anymore! At least, not as a first citizen: an implicit
  "virtual" graph is progressively formed each time an operation over some
  nodes is applied. The virtual graph can be observed by simply walking the
  tree of operations. Most methods of the former Graph are now simple
  functions in the `ag` package.
- We still provide a way to explicitly "free" some resources after use,
  both for helping the garbage collector and for returning some objects
  to their `sync.Pool`. The function `ag.ReleaseGraph` operates on the
  virtual graph described above, usually starting from the given output nodes.
- Forward operations are executed concurrently. As soon as an Operator is
  created (usually by calling one of the functions in `ag`, such as `Add`,
  `Prod`, etc.), the related Function's `Forward` procedure is performed
  on a new goroutine. Nevertheless, it's always safe to ask for the Operator's
  `Value` without worries: if it's called too soon, the function will lock
  until the result is computed, and only then return the value.
- To maximize performance, we removed the possibility to set a custom limit
  for concurrent computations. Thanks to the new design, we now let the Go
  runtime itself manage this problem for us, so that you can still limit
  and finetune concurrency with the `GOMAXPROCS` variable.
- The implementation of backpropagation is also redesigned and improved.
  Instead of invoking the backward procedure on an explicit Graph, you can call
  `ag.Backward` or `ag.BackwardMany`, specifying the output node (or nodes)
  of your computation (such as loss values, in traditional scenarios).
  The backward functions traverse the virtual graph and propagate the gradients,
  leveraging concurrency and making use of goroutines and locks in a way that's
  very similar to the forward procedure. The backward functions will lock and
  wait until the whole gradients propagation is complete before returning.
  The locking mechanism implemented in the nodes' `Grad` methods, will still
  prevent troubles in case your own code reads the gradients concurrently
  (that would be very uncommon).
- We also modified the implementation of time-steps handling and truncated 
  backpropagation. Since we don't have the support of a concrete Graph
  structure anymore, we introduced a new dedicated type `ag.TimeStepHandler`,
  and related functions, such as `NodeTimeStep`. For performing a truncated
  backpropagation, we provide the function `ag.BackwardT` and
  `ag.BackwardManyT`: they work similarly to the normal backpropagation
  functions described above, only additionally requiring a time-step
  handler and the desired amount of back steps.
- We simplified and polished the API for creating new node-variables. Instead
  of having multiple functions for simple variables, scalars, constants,
  with/without name or grads, and various combination of those, you can now
  create any new variable with `ag.Var`, which accepts a Matrix value and
  creates a new node-variable with gradients accumulation disabled by default.
  To enable gradients propagation, or setting an explicit name (useful for
  model params or constants), you can use the Variable's chainable methods
  `WithGrad` and `WithName`. As a shortcut to create a scalar-matrix variable
  you can use `ag.Scalar`.
- The package `ag/encoding` provides generic structures and functions to obtain
  a sort of view of a virtual graph, with the goal of facilitating the
  encoding/marshaling of a graph in various formats.
  The package `ag/encoding/dot` is a rewriting of the former `pkg/ml/graphviz`,
  that uses the `ag/encoding` structures to represent a virtual graph in
  Graphviz DOT format.

### Models

- As before, package `nn` provides types and functions for defining and
  handling models. Its subpackages are implementations of most common models.
  The set of built-in models has been remarkably revisited, moving some of them
  to separate projects, as previously explained.
- The `Model` interface has been extremely simplified: it only requires the
  special empty struct `Module` to be embedded in a model type. This is
  necessary only to distinguish an actual model from any other struct, which
  is especially useful for parameters traversal, or other similar operations.
- Since the Graph has been removed from `ag`, the models clearly don't need
  to hold a reference to it anymore. Similarly, there is no need for any other
  model-specific field, like the ones available from the former `BaseModel`.
  This implies the elimination of some seldomly used properties.
  Notable examples are the "processing mode" (from the old Graph) and the time
  step (from the old BaseModel).
  In situations where a removed value or feature is still needed, we suggest to 
  either reintroduce the missing elements on the models that needs them, or 
  to extract them to separate types and functions. An example of
  extracted behavior is the handling of time steps, already mentioned in the
  previous section.
- There is no distinction anymore between "pure" models and processors,
  making "reification" no longer necessary: once a model is created (or loaded),
  it can be immediately used, even for multiple concurrent inferences.
- A side effect of removing processor instances is that it's not possible
  to hold any sort of state related to a specific inference inside the
  structure of a model (or, at least, it's discouraged in most situations). 
  Keeping track of a state is quite common for models that work with a running
  "memory" or cache. The recommended approach is to represent the state
  as a separate type, so that the "old" state can be passed as argument
  to the model's forward function (along with any other input), and the "new"
  or updated state can be returned from the same function (along with any other
  output).
  Some good examples can be observed in the implementation of recurrent
  networks (RNNs), located at `nn/recurrent/...`: each model has a single-step
  forward function (usually called `Next`) that accepts a previous state
  and returns a new one.
- We removed the `Stack` Model, in favor of a new simple function `nn.Forward`,
  that operates on a slice of `StandardModel` interfaces, connecting outputs to
  inputs sequentially for each module.
- We introduced the new type `nn.Buffer`: it's a Node implementation that does
  not require gradients, but can be serialized just like any other
  parameter. This is useful, for example, to store constants, to track the mean
  and std in batch norm layers, etc.
  As a shortcut to create a Buffer with a scalar-matrix value you can use
  `nn.Const`.
- We refactored the arguments of the parameters-traversal functions
  `ForEachParam` and `ForEachParamStrict`.
  Furthermore, the new interface `ParamsTraverser` allows to traverse a model's
  parameters that are not automatically discovered by the traversal functions
  via reflection. If a model implements this interface, the function
  `TraverseParams` will take precedence over the regular parameters visit.
- We introduced the function `Apply`, which visits all sub-models of any Model.
  Typical usages of this function include parameters initialization.

### Embeddings

- The embeddings model has been refactored and made more flexible by
  splitting the new implementation into three main concerns: stores,
  the actual model, and the model's parameters.
- Raw embeddings data can be read from, and perhaps written to,
  virtually any suitable medium, be it in-memory, on-disk, local or remote
  services or databases, etc. The `Store` interface, defined in
  package `embeddings/store`, only requires an implementation to implement
  a bunch of read/write functions for key/value pairs. Both keys and values
  are just slice of bytes.
  For example, in a typical scenario involving word embeddings, a key might 
  be a `string` word converted to `[]byte`, and the value the byte-marshaled 
  representation of a vector (or a more complex struct also holding
  other properties).
- It's not uncommon for a complex model, or application, to make use of
  more than one store. For a more convenient handling, multiple independent
  Stores can be organized together in a `Repository`, another interface
  defined in `embeddings/store`. A Repository is simply a provider for Stores,
  where each Store is identified by a `string` name.
  For example, if we are going to use a relational database for storing
  embeddings data, the Repository might establish the connection to the
  database, whereas each Store might identify a separate table by name,
  used for reading/writing data.
- We provide two built-in implementations of Repository/Store pairs.
  The package `embeddings/store/diskstore` is a Go submodule that stores data
  on disk, using BadgerDB; this is comparable to the implementation
  from previous releases.
  The package `embeddings/store/memstore` is a simple volatile in-memory
  implementation; among other usages, it might be especially convenient for
  testing.
- The package `embeddings` implements the main embeddings `Model`.
  One Model can read and write data to a single Store, obtained from a
  Repository by the configured name.
  The model delegates to the embeddings Store the responsibility to actually
  store the data; for this reason, the Store value on a Model is prevented
  from being serialized (this is done with the utility type
  `embeddings/store.PreventStoreMarshaling`).
- To facilitate different use cases, the Model allows a limited set of
  possible key types, using the constraint `Key` as type argument.
- The type `Embedding` represents a single embedding value that can be handled
  by a Model. It satisfies the interface `nn.Param`, allowing seamless
  integration with operations involving any other model. Behind the hood,
  the implementation takes care of reading/writing data against a
  Store, efficiently handling marshaling/unmarshaling and preventing
  race conditions. The `Value` and the `Payload` (if any) are read/written
  against the Store; the `Grad` is only kept in memory. All properties
  of different `Embedding` instances for the same key are kept
  synchronized upon changes.
- A Model keeps track of all Embedding parameters with associated gradients.
  The method `TraverseParams` allows these parameters to be discovered and
  seen as if they were any other regular type of parameter. This is
  especially important for operations such as embeddings optimization.
- It is a common practice to share the same embeddings among multiple models.
  In this case it is important that the serialized (and deserialized)
  instance is very same one. Therefore, we introduced the `Shared` structure
  that prevents binary marshaling.

### Optimizers 

- Gradient descent optimization algorithms are available under the package
  `gd`, with minor API changes.
- We removed other methods, such as differential evolution, planning to
  re-implement them on separate forthcoming projects.

### Utilities

- We removed the formed package `pkg/utils`. Some of its content was related
  to functionalities now moved to separate projects. Any remaining useful code
  has been refactored and moved to more appropriate places. 

## [0.7.0] - 2021-05-24

### Added
- New package `ml/ag/encoding/dot`, for simple serialization of a Graph to 
  DOT (Graphviz) format.
- New package `ml/nn/sgu`, implementing a Spatial Gating Unit (SGU) model.
- New package `ml/nn/conv1x1`, implementing a simple 1-dimensional
  1-sized-kernel convolution model.
- New package `ml/nn/gmlp`, implementing a [gMLP](https://arxiv.org/pdf/2105.08050.pdf)
  model.

### Changed
- `ml/nn/activation/Model.Forward` now simply returns the input as it is if
  the activation function is the identity.

## [0.6.0] - 2021-05-13

### Added
- `ml/losses.WeightedCrossEntropy()`
- `ml/losses.FocalLoss()`
- `ml/losses.WeightedFocalLoss()`
- `nlp/sequencelabeler.LoadModel()` (it replaces `Load()` and `LoadEmbeddings()`)
- `nlp/charlm.LoadModel()`
- `nlp/transformers/bert.Model.PredictMLM()`
- `nlp/transformers/bart/tasks` package
- `nlp/transformers/bert.Model.Vectorize()`
- `ml/ag.Graph.Nodes()` and `ml/ag.Nodes()`
- `ml/nn.Model.Close()`
- `ml/nn.ReifyForTraining()` and `ml/nn.ReifyForInference()`
- `ml/ag.Graph.Backward()` now panics if it is executed with nodes belonging to
  different graphs.
- The new `ml/graphviz` package allows exporting a Graph to [Graphviz](https://graphviz.org/)
  [DOT](https://graphviz.org/pdf/dotguide.pdf) format. To make it possible,
  we introduced a new go-mod dependency [gographviz](https://github.com/awalterschulze/gographviz).
- A custom name can be optionally set to a Graph's Variables. This can be
  useful for debugging purposes and visual graph representation.
  You can now use `Graph.NewVariableWithName()` and `Graph.NewScalarWithName()`
  to create named Variables, and get the name of a Variable with
  `Variable.Name()`.

### Changed
- All `UnaryElementwise` functions provided by the package `ag/fn` have been
  promoted to separate dedicated structs. This improves debuggability and you
  can get appropriate function names when using reflection. Here is the full
  list of the modified functions: `Tan`, `Tanh`, `Sigmoid`, `HardSigmoid`,
  `HardTanh`, `ReLU`, `Softsign`, `Cos`, `Sin`, `Exp`, `Log`, `Neg`,
  `Reciprocal`, `Abs`, `Mish`, `GELU`, `Sqrt`, `Swish`.
  For the same reason, a dedicated `Square` function is introduced, replacing
  `Prod` with both operands set to the same value.
- `ml/ag` types `Operator`, `Variable`, `Wrapper` are now public.
- `ml/nn.Reify()` now expects a Graph and a Processing Mode arguments
  instead of a `Context` object (removed).
- `ml/nn.BaseModel` has been modified, replacing the field `Ctx Context` with
  a direct reference to the model's Graph and the Processing Mode (fields `G`
  and `ProcessingMode`).
- Refactoring server implementation of `nlp/sequencelabeler`,
  `nlp/transformers/bert`, and `nlp/transformers/bart`.
- Upgrade various dependencies.
- Regenerate protocol buffers files (with `protoc-gen-go` v1.26.0 and
  `protoc` v3.16.0).

### Removed
- `nlp/sequencelabeler.Load()` and `LoadEmbeddings()` (now replaced by
  `nlp/sequencelabeler.LoadModel()`)
- `ml/nn.Context` (see related changes on `Reify()` and `BaseModel`)

## [0.5.2] - 2021-03-16

### Added

- Handle multiple BERT pooling strategies (e.g. CLS_TOKEN, REDUCE_MEAN) in `nlp.transformers.bert.server_encode.go`.

## [0.5.1] - 2021-03-07

### Added

- Add `nlp.charlm.flair_converter.go` to
  import [Flair character language models](https://github.com/flairNLP/flair/issues/614).

### Changed

- Improve `nlp.transformer.generation` algorithms:
  - optimize `Generator.getTopKScoredTokens()`.
  - optimize `Generator.updateTokensScores()`.
- Simplify `mat32.Dense.Mul` when doing Matrix-Vector multiplication.
- Refactor `math32` functions using [chewxy/math32](https://github.com/chewxy/math32) functions.
- Improve `ag.Graph` efficiency:
  - Use pre-computed cache doing `ag.Graph.groupNodesByHeight()`.
  - Use `sync.pool` to reduce allocations of graph's operators.

### Fixed

- Fix past key-values usage on self-attention and cross-attention

## [0.5.0] - 2021-02-15

### Added

- Implement a beam-search algorithm for conditional generation:
  - `nlp.transformer.generation` package.
- Add implementation of the Sentence-Piece tokenizer:
  - `nlp.tokenizers.sentencepiece` package.
- BART improvements:
  - gRPC and HTTP API to perform Text Generation.
  - Add support for "Marian" architecture (used for translation tasks).
  - Add sinusoidal positional encoder (used by Marian).
  - Add "head" for conditional generation:
    - `nlp.transformers.bart.head.conditionalgeneration` package.
- Add `nn.Closer` interface (e.g. `embeddings.Model` needs to close the underlying key-value store).
- Add Swish act. function without trainable parameters.
- Add SiLU act. function (it is just an alias for Swish).
- New `pe.SinusoidalPositionalEncoder` (this implementation replaces unused `pe.PositionalEncoder`
  and `pe.AxialPositionalEncoder`)

### Changed

- Update urfave/cli to v2
- Update dgraph-io/badger to v3.
- Make the BART positional encoder an interface to support various encoding (i.e. trainable vs static).
- Rename to `fn.NewSwish` into `fn.NewSwishB` as this was the Swish variant with trainable parameters (*B*).
- Relax `ag.GetOpName` to match operator names in lower-case.
- Allow arbitrary activation function on BART encoder/decoder layers.
- Use precomputed "keys" and "values" in self-attention, multi-head attention and BART decoder.

### Removed

- In relation to the aforementioned positional encoding changes:
  - `pe.PositionalEncoder` and related functions
  - `pe.AxialPositionalEncoder` and related functions

### Fixed

- Fix causal-mask used by `nn.ScaledDotProductAttention`

## [0.4.1] - 2021-01-22

### Added

- New function `ReleaseMatrix` to packages `mat32` and `mat64`.
- New methods to `Matrix` interface, from `mat32` and `mat64`: `Minimum`,
  `Maximum`, `MulT`, `Inverse`, `DoNonZero`. However, the implementation on sparse matrices is not implemented yet (it
  always panics).

### Changed

- Prefer handling `Matrix` interface values over specific `Dense` or `Sparse`
  matrices, also avoiding unnecessary type casts. Relevant changes to the public API are listed below.
  - `mat(32|64).Stack` function's arguments and returned value are now `Matrix` 
    interfaces, instead of explicit `Dense` matrices.
  - `Dense.Minimum` and `Dense.Maximum`, from packages `mat32` and `mat64`,
    return a `Matrix` interface, instead of a specific `Dense` type.
  - The return values of `fofe.EncodeDense`, `fofe.Encode`, and `fofe.BiEncode`
    are slices of `Matrix` values, instead of `Dense` or `Sparse`.
  - The `z` argument of the function `fofe.Decode` is of type `Matrix`,
    instead of `Dense`.
  - `ml.optimizers.de` (Differential Evolution optimizer) API was changed
    handling `Matrix` values, instead of specific `Dense` matrices. Changes
    include: `Member.TargetVector`, `Member.DonorVector`, `ScoredVector.Vector`, 
    the `vector` argument of `NewMember` function, the `solution` argument
    of `score` and `validate` functions passed to `NewOptimizer`.
  - `PositionalEncoder.Cache` and `AxialPositionalEncoder.Cache` are slices
    of `Matrix`, instead of slices of `Dense`.
  - `AxialPositionalEncoder.EncodingAt` returns a `Matrix` value, instead of `Dense`.
  - `nn.DumpParamsVector` returns a `Matrix` value, instead of `Dense`.
  - The `vector` argument of the function `nn.LoadParamsVector` is a `Matrix`, 
    instead of `Dense`.
  - The `value` argument of the method `embeddings.Model.SetEmbedding` is of
    type `Matrix`, instead of `Dense`.
  - The type of the struct field `evolvingembeddings.WordVectorPair.Vector` is
   `Matrix`, instead of `Dense`.

## [0.4.0] - 2021-01-17

### Added

- Various new test cases (improving the coverage).
- `nlp.embeddings.syncmap` package.
- `ml.nn.recurrent.srnn.BiModel` which implements a bidirectional variant of the Shuffling Recurrent Neural Networks (
  SRNN).
- Configurable timeout and request limit to all HTTP and gRPC servers (see also commands help).

### Changed

- All CLI commands implementation has been refactored, so that the
  `docker-entrypoint` can reuse all other `cli.App` objects, instead of
  just running separate executables. By extension, now the Dockerfile builds
  a single executable file, and the final image is way smaller.
- All dependencies have been upgraded to the latest version.
- Simplify custom error definitions using `fmt.Errorf` instead of functions
  from `github.com/pkg/errors`.
- Custom binary data serialization of matrices and models is now achieved
  with Go's `encoding.gob`. Many specific functions and methods are now
  replaced by fewer and simpler encoding/decoding methods compatible with
  `gob`. A list of important related changes follows.
  - `utils.kvdb.KeyValueDB` is no longer an interface, but a struct which
    directly implements the former "badger backend".
  - `utils.SerializeToFile` and `utils.DeserializeFromFile` now handle
    generic `interface{}` objects, instead of values implementing
    `Serializer` and `Deserializer`.
  - `mat32` and `mat64` custom serialization functions (e.g.
    `MarshalBinarySlice`, `MarshalBinaryTo`, ...) are replaced by
    implementations of `BinaryMarshaler` and `BinaryUnmarshaler` interfaces
    on `Dense` and `Sparse` matrix types.
  - `PositionalEncoder.Cache` and `AxialPositionalEncoder.Cache` fields (from
    `ml.encoding.pe` package) are now public.
  - All types implementing `nn.Model` interface are registered for gob
    serialization (in init functions).
  - `embeddings.Model.UsedEmbeddings` type is now `nlp.embeddings.syncmap.Map`.
  - As a consequence, you will have to re-serialize all your models.
- Flair converter now sets the vocabulary directly in the model, instead
  of creating a separate file.
- `sequencelabeler.Model.LoadParams` has been renamed to `Load`.

### Removed
- In relation to the aforementioned gob serialization changes:
  - `nn.ParamSerializer` and related functions
  - `nn.ParamsSerializer` and related functions
  - `utils.Serializer` and `utils.Deserializer` interfaces
  - `utils.ReadFull` function
- `sequencelabeler.Model.LoadVocabulary`

### Fixed
- `docker-entrypoint` sub-command `hugging-face-importer` has been renamed to
  `huggingface-importer`, just like the main command itself.
- `docker-entrypoint` sub-command can be correctly specified without leading
  `./` or `/` when run from a Docker container.
- BREAKING: mat32.Matrix serialization has been fixed, now serializing single
  values to chunks of 4 bytes (instead of 8, like float64). Serialized 32-bit
  models will now be half the size! Unfortunately you will have to re-serialize
  your models (sorry!).

## [0.3.0] - 2021-01-10
### Added
- Static analysis job (`golint` and `gocyclo`) to `Go` GitHub workflow.
- You can set a limit for concurrent heavyweight Graph computations (e.g.
  forward and backward steps) - see `ml.ag.ConcurrentComputations()`
  (`GraphOption`) and `ml.ag.Graph.ConcurrentComputations()`.
  If no option is specified, by default the limit is set to `runtime.NumCPU()`.
- You can set a limit for concurrent heavyweight computations of
  `ml.optimizers.gd.GradientDescent` (e.g. params update step).
- New package `utils.processingqueue`.
- `mat32` package, which operates on `float32` data type.
- It's possible to switch between `float32` and `float64` as default
  floating-point data type, using the script `change-float-type.sh`
- `Go` GitHub workflow has been adapted to run tests using both `float32`
  and `float64` as main floating-point data type.
- This CHANGELOG file.
- Pull and convert Hugging Face models automatically if not found locally 
  when starting BERT or BART server.
- Move content from GitHub Wiki to README in related package folders.

### Changed
- `ml.ag.ConcurrentComputations` (`GraphOption`) expects the maximum number
  of concurrent computations handled by heavyweight Graph operations (e.g.
  forward and backward steps).
- `ml.nn.linear.Model` and `ml.nn.convolution.Model` read the concurrent
  computations limit set on the model's Graph, thus
  `SetConcurrentComputations()` methods have been removed.
- `mat` has been renamed to `mat64` and some functions have been renamed.
- The whole project now works with `float32` floating-point data type by
  default, by using the package `mat32`.
- When imported, the new package `mat32` is always aliased as `mat`. Then,
  explicit usages of `float64` type have been replaced with `mat.Float`.
  Moreover, bitsize-specific functions have been made more generic (i.e.
  operating with `mat.Float` type) or split into separate implementation,
  in `mat32` and `mat64`. In this way, switching the whole project between
  `float32` and `float64` is just a matter of changing all imports, from
  `mat32` to `mat64`, or vice-versa (see also the new file
  `change-float-type.sh`).
- Update internal links to pre-trained NER models to float32 versions.
- `nlp.sequencelabeler.Convert()` now loads and converts original Flair models,
   instead of pre-processed dumps.
- Change command line arguments to make them more consistent; please refer to
  the help messages of each command.
- Update Dockerfile using a new base building image and adding bart server.

### Fixed
- Added dedicated package names to different protocol buffers definition files
  to avoid name conflicts.

## [0.2.0] - 2020-12-31
### Added
- Support for BART model (tested on Natural Language Inference task).
- BART API to perform Zero-Shot Text Classification.

### Changed
- Significant reduction of boilerplate code through the unification of
  `nn.Model` and `nn.Processor` interfaces:
  - there is now a single `nn.Model` interface that can be reified to become a
    neural processor - see `nn.Reify()`;
  - there was no compelling reason to have a `Forward` method in the `nn.Model`
    interface, so it has been removed, gracefully increasing flexibility in the
    implementation of a model.

## [0.1.0] - 2020-12-09
First tagged release!

[Unreleased]: https://github.com/nlpodyssey/spago/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/nlpodyssey/spago/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/nlpodyssey/spago/compare/v1.0.0-alpha...v1.0.0
[1.0.0-alpha]: https://github.com/nlpodyssey/spago/compare/v0.7.0...v1.0.0-alpha
[0.7.0]: https://github.com/nlpodyssey/spago/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/nlpodyssey/spago/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/nlpodyssey/spago/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/nlpodyssey/spago/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/nlpodyssey/spago/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/nlpodyssey/spago/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/nlpodyssey/spago/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/nlpodyssey/spago/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nlpodyssey/spago/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nlpodyssey/spago/releases/tag/v0.1.0
