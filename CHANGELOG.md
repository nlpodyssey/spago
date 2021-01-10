# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
