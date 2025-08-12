Directory structure:
└── genjax-community-genjax/
├── README.md
├── poetry.toml
├── pyproject.toml
├── docs/
│ ├── CNAME
│ ├── codeofconduct.md
│ ├── contributing.md
│ ├── developing.md
│ ├── index.md
│ ├── license.md
│ ├── assets/
│ │ └── font/
│ │ └── BerkeleyMonoVariable-Regular.woff2
│ ├── cookbook/
│ │ ├── active/
│ │ │ ├── choice_maps.ipynb
│ │ │ ├── debugging.ipynb
│ │ │ ├── generative_function_interface.ipynb
│ │ │ ├── intro.ipynb
│ │ │ └── jax_basics.ipynb
│ │ └── inactive/
│ │ ├── generative_fun.ipynb
│ │ ├── differentiation/
│ │ │ ├── adev_demo.py
│ │ │ └── adev_example.py
│ │ ├── expressivity/
│ │ │ ├── conditionals.ipynb
│ │ │ ├── custom_distribution.ipynb
│ │ │ ├── iterating_computation.ipynb
│ │ │ ├── masking.ipynb
│ │ │ ├── mixture.ipynb
│ │ │ ├── ravi_stack.ipynb
│ │ │ ├── stochastic_probabilities.ipynb
│ │ │ └── stochastic_probabilities_math.ipynb
│ │ ├── inference/
│ │ │ ├── custom_proposal.ipynb
│ │ │ ├── importance_sampling.ipynb
│ │ │ └── mcmc.ipynb
│ │ ├── library_author/
│ │ │ └── dimap_combinator.ipynb
│ │ └── update/
│ │ ├── 1_importance.ipynb
│ │ ├── 2_update.ipynb
│ │ ├── 3_speed_gains.ipynb
│ │ ├── 4_index_request.ipynb
│ │ └── 7_application_dirichlet_mixture_model.ipynb
│ ├── css/
│ │ ├── custom.css
│ │ └── mkdocstrings.css
│ ├── js/
│ │ └── mathjax.js
│ └── library/
│ ├── combinators.md
│ ├── core.md
│ ├── generative_functions.md
│ └── inference.md
└── tests/
├── adev/
│ └── test_adev.py
├── core/
│ ├── test_diff.py
│ ├── test_pytree.py
│ ├── test_staging.py
│ ├── generative/
│ │ ├── test_core.py
│ │ └── test_functional_types.py
│ └── interpreters/
│ └── test_incremental.py
├── generative_functions/
│ ├── test_dimap_combinator.py
│ ├── test_distributions.py
│ ├── test_mask_combinator.py
│ ├── test_mix_combinator.py
│ ├── test_or_else.py
│ ├── test_repeat_combinator.py
│ ├── test_scan_combinator.py
│ ├── test_switch_combinator.py
│ └── test_vmap_combinator.py
└── inference/
├── test_requests.py
├── test_smc.py
└── test_vi.py

================================================
FILE: README.md
================================================
<br>

<p align="center">
<img width="350px" src="./docs/assets/img/logo.png"/>
</p>
<p align="center">
  <strong>
  Probabilistic programming with programmable inference for parallel accelerators.
  </strong>
</p>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/genjax)](https://pypi.org/project/GenJAX/)
[![codecov](https://codecov.io/gh/genjax-dev/genjax-chi/graph/badge.svg?token=OlfTXjcrEW)](https://codecov.io/gh/genjax-dev/genjax-chi)
[![][jax_badge]](https://github.com/google/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Public API: beartyped](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg?style=flat-square)](https://beartype.readthedocs.io)

|                                                                                                   **Documentation**                                                                                                   |                   **Build status**                    |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------: |
| [![](https://img.shields.io/badge/docs-stable-blue.svg?style=flat-square)](https://genjax.gen.dev) [![](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat-square&logo=jupyter&logoColor=white)][cookbook] | [![][main_build_action_badge]][main_build_status_url] |

</div>

> This is the community edition of [GenJAX](https://github.com/probcomp/genjax), a probabilistic programming language in development at MIT's Probabilistic Computing Project. We recommend this version for stability, community contributions, expanded features and more active community-driven maintenance. The research version is more likely to be unstable, and evolve sporadically.

## 🔎 What is GenJAX?

Gen is a multi-paradigm (generative, differentiable, incremental) language for probabilistic programming focused on [**generative functions**: computational objects which represent probability measures over structured sample spaces](https://genjax.gen.dev/cookbook/active/intro.html#generative-functions).

GenJAX is an implementation of Gen on top of [JAX](https://github.com/google/jax) - exposing the ability to programmatically construct and manipulate generative functions, as well as [JIT compile + auto-batch inference computations using generative functions onto GPU devices](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html).

<div align="center">
<a href="https://genjax.gen.dev/cookbook/active/jax_basics.html">Jump into the notebooks!</a>
<br>
<br>
</div>

> [!TIP]
> GenJAX is part of a larger ecosystem of probabilistic programming tools based upon Gen. [Explore more...](https://www.gen.dev/)

## Quickstart

To install GenJAX, run

```bash
pip install genjax
```

Then install [JAX](https://github.com/google/jax) using [this
guide](https://jax.readthedocs.io/en/latest/installation.html) to choose the command for the
architecture you're targeting. To run GenJAX without GPU support:

```sh
pip install jax[cpu]~=0.4.24
```

On a Linux machine with a GPU, run the following command:

```sh
pip install jax[cuda12]~=0.4.24
```

### Quick example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KWMa5No95tMDYEdmA4N0iqVFD-UsCSgp?usp=sharing)

The following code snippet defines a generative function called `beta_bernoulli` that

-   takes a shape parameter `beta`
-   uses this to create and draw a value `p` from a [Beta
    distribution](https://en.wikipedia.org/wiki/Beta_distribution)
-   Flips a coin that returns 1 with probability `p`, 0 with probability `1-p` and
    returns that value

Then, we create an inference problem (by specifying a posterior target), and utilize sampling
importance resampling to give produce single sample estimator of `p`.

We can JIT compile that entire process, run it in parallel, etc - which we utilize to produce an estimate for `p`
over 50 independent trials of SIR (with K = 50 particles).

```python
import jax
import jax.numpy as jnp
import genjax
from genjax import beta, flip, gen, Target, ChoiceMap
from genjax.inference.smc import ImportanceK

# Create a generative model.
@gen
def beta_bernoulli(α, β):
    p = beta(α, β) @ "p"
    v = flip(p) @ "v"
    return v

@jax.jit
def run_inference(obs: bool):
    # Create an inference query - a posterior target - by specifying
    # the model, arguments to the model, and constraints.
    posterior_target = Target(beta_bernoulli, # the model
                              (2.0, 2.0), # arguments to the model
                              ChoiceMap.d({"v": obs}), # constraints
                            )

    # Use a library algorithm, or design your own - more on that in the docs!
    alg = ImportanceK(posterior_target, k_particles=50)

    # Everything is JAX compatible by default.
    # JIT, vmap, to your heart's content.
    key = jax.random.key(314159)
    sub_keys = jax.random.split(key, 50)
    _, p_chm = jax.vmap(alg.random_weighted, in_axes=(0, None))(
        sub_keys, posterior_target
    )

    # An estimate of `p` over 50 independent trials of SIR (with K = 50 particles).
    return jnp.mean(p_chm["p"])

(run_inference(True), run_inference(False))
```

```python
(Array(0.6039314, dtype=float32), Array(0.3679334, dtype=float32))
```

## References

Many bits of knowledge have gone into this project -- [you can find many of these bits at the MIT Probabilistic Computing Project page](http://probcomp.csail.mit.edu/) under publications. Here's an abbreviated list of high value references:

-   [Marco Cusumano-Towner's thesis on Gen][marco_thesis]
-   [The main Gen.jl repository][gen_jl]
-   (Trace types) [(Lew et al) trace types][trace_types]
-   (RAVI) [(Lew et al) Recursive auxiliary-variable inference][ravi]
-   (GenSP) [Alex Lew's Gen.jl implementation of GenSP][gen_sp]
-   (ADEV) [(Lew & Huot, et al) Automatic differentiation of expected values of probabilistic programs][adev]

### JAX influences

This project has several JAX-based influences. Here's an abbreviated list:

-   [This notebook on static dispatch (Dan Piponi)][effect_handling_interp]
-   [Equinox (Patrick Kidger's work on neural networks via callable Pytrees)][equinox]
-   [Oryx (interpreters and interpreter design)][oryx]

### Acknowledgements

The maintainers of this library would like to acknowledge the JAX and Oryx maintainers for useful discussions and reference code for interpreter-based transformation patterns.

## Disclaimer

This is a research project. Expect bugs and sharp edges. Please help by trying out GenJAX, [reporting bugs](https://github.com/ChiSym/genjax/issues), and letting us know what you think!

## Get Involved + Get Support

Pull requests and bug reports are always welcome! Check out our [Contributor's
Guide](CONTRIBUTING.md) for information on how to get started contributing to GenJAX.

The TL;DR; is:

-   send us a pull request,
-   iterate on the feedback + discussion, and
-   get a +1 from a maintainer

in order to get your PR accepted.

Issues should be reported on the [GitHub issue tracker](https://github.com/ChiSym/genjax/issues).

If you want to discuss an idea for a new feature or ask us a question, discussion occurs primarily in the body of [Github Issues](https://github.com/ChiSym/genjax/issues)

<div align="center">
Created and maintained by the <a href="http://probcomp.csail.mit.edu/">MIT Probabilistic Computing Project</a>. All code is licensed under the <a href="LICENSE">Apache 2.0 License</a>.
</div>

[actions]: https://github.com/genjax-dev/genjax-chi/actions
[adev]: https://arxiv.org/abs/2212.06386
[cookbook]: https://genjax.gen.dev/cookbook/
[coverage_badge]: https://github.com/genjax-dev/genjax-chi/coverage.svg
[discord-url]: https://discord.gg/UTJj3zmJYb
[discord]: https://img.shields.io/discord/1331245195618029631?style=flat-square&colorA=000000&colorB=000000&label=&logo=discord
[effect_handling_interp]: https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=ukjVJ2Ls_6Q3
[equinox]: https://github.com/patrick-kidger/equinox
[gen_jl]: https://github.com/probcomp/Gen.jl
[gen_sp]: https://github.com/probcomp/GenSP.jl
[jax_badge]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC
[main_build_action_badge]: https://github.com/genjax-dev/genjax-chi/actions/workflows/ci.yml/badge.svg?style=flat-square&branch=main
[main_build_status_url]: https://github.com/genjax-dev/genjax-chi/actions/workflows/ci.yml?query=branch%3Amain
[marco_thesis]: https://www.mct.dev/assets/mct-thesis.pdf
[oryx]: https://github.com/jax-ml/oryx
[ravi]: https://arxiv.org/abs/2203.02836
[trace_types]: https://dl.acm.org/doi/10.1145/3371087

================================================
FILE: poetry.toml
================================================
[virtualenvs]
in-project = true

[installer]
no-binary = ["python-crfsuite"]

================================================
FILE: pyproject.toml
================================================
[tool.poetry]
name = "genjax"

# Leave this at 0.0.0; this key can't be missing, but it's subbed out

# dynamically by `poetry.dynamic-versioning`.

version = "0.0.0"

description = "Probabilistic programming with Gen, built on top of JAX."
authors = [
"McCoy R. Becker <mccoyb@mit.edu>",
"MIT Probabilistic Computing Project <probcomp-assist@csail.mit.edu>",
]
maintainers = [
"McCoy R. Becker <mccoyb@mit.edu>",
"Colin Smith <colin.smith@gmail.com>",
"Sam Ritchie <sam@mentat.org>",
]
license = "Apache 2.0"
readme = "README.md"
homepage = "https://github.com/genjax-dev/genjax-chi"
repository = "https://github.com/genjax-dev/genjax-chi"
documentation = "https://genjax.gen.dev"
keywords = [
"artificial-intelligence",
"probabilistic-programming",
"bayesian-inference",
"differentiable-programming",
]
classifiers = [
"Development Status :: 4 - Beta",
"Programming Language :: Python :: 3.10",
"Programming Language :: Python :: 3.11",
"Programming Language :: Python :: 3.12",
]

[tool.poetry.urls]
Changelog = "https://github.com/genjax-dev/genjax-chi/releases"

[tool.poetry.dependencies]
python = ">=3.11,<4.0"
jax = "0.5.2"
tensorflow-probability = "^0.23.0"
jaxtyping = "^0.2.24"
beartype = "^0.20.0"
deprecated = "^1.2.14"
penzai = "^0.2.2"
treescope = "^0.1.5"

# Numpy <2.0.0 is due to tfp issues: https://github.com/tensorflow/probability/issues/1814

# remove this constraint when that issue is closed.

numpy = ">=1.22,<2.0.0"
genstudio = ">=2025.2.1"
safety = "^3.5.1"

[tool.poetry.group.dev]
optional = true

[tool.poetry.group.dev.dependencies]
coverage = "^7.0.0"
hypothesis = "^6.119.0"
matplotlib = "^3.6.2"
mypy = "^0.991"
pytest = "^7.2.0"
pytest-benchmark = "^4.0.0"
pytest-xdist = { version = "^3.2.0", extras = ["psutil"] }
ruff = "0.11.2"
safety = ">=2.3.5"
seaborn = "^0.12.1"
xdoctest = "^1.1.0"
jupyterlab = "^4.2.5"
nox = "^2024.3.2"
nox-poetry = "^1.0.3"
jupytext = "^1.16.2"
pre-commit = "^4.2.0"
pyright = "1.1.399"

[tool.poetry.group.docs]
optional = true

[tool.poetry.group.docs.dependencies]
mkdocs = "^1.5.3"
mkdocs-git-authors-plugin = "^0.7.2"
mkdocs-git-revision-date-localized-plugin = "^1.1.0"
mkdocs-git-committers-plugin-2 = "^1.1.1"
markdown-exec = { version = "^1.8.3", extras = ["ansi"] }
black = "^24.4.2"
mkdocs-jupyter = "^0.25.1"
mkdocstrings-python = "^1.16.5"
mkdocs-material = "^9.6.8"

[tool.poetry.extras]
genstudio = ["genstudio"]
all = ["genstudio"]

[tool.poetry.requires-plugins]
poetry-plugin-export = ">=1.8"

[tool.coverage.paths]
source = ["src", "*/site-packages"]
tests = ["tests", "*/tests"]

[tool.coverage.run]
omit = [".*", "*/site-packages/*"]

[tool.coverage.report]
show_missing = true
fail_under = 45

[tool.vulture]
paths = ["src"]
ignore_names = ["cls"]
exclude = ["*/.ipynb_checkpoints/*"]
min_confidence = 70
sort_by_size = true

[tool.pyright]
pythonVersion = "3.11"
venvPath = "."
venv = ".venv"
include = ["src", "tests"]
exclude = ["**/__pycache__"]
defineConstant = { DEBUG = true }
typeCheckingMode = "strict"
deprecateTypingAliases = true

# `strict` sets all of these to error; these remaining `none` entries are tests that we can't yet

# pass.

reportMissingTypeStubs = "none"
reportMissingParameterType = "none"
reportUnknownArgumentType = "none"
reportUnknownLambdaType = "none"
reportUnknownMemberType = "none"
reportUnknownParameterType = "none"
reportUnknownVariableType = "none"

[tool.ruff]
target-version = "py311"
exclude = [
".bzr",
".direnv",
".eggs",
".git",
".git-rewrite",
".hg",
".mypy_cache",
".nox",
".pants.d",
".pytype",
".ruff_cache",
".svn",
".tox",
".venv",
"__pypackages__",
"_build",
"buck-out",
"build",
"dist",
"node_modules",
"venv",
".venv"
]
extend-include = ["*.ipynb"]
line-length = 88
indent-width = 4

[tool.ruff.lint.isort]
known-first-party = ["genjax"]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint]
preview = true
extend-select = ["I", "RUF"]
select = ["E4", "E7", "E9", "F"]

# F403 disables errors from `*` imports, which we currently use heavily.

ignore = ["F403", "F405", "F811", "E402", "RUF009", "RUF003"]
fixable = ["ALL"]
unfixable = []
dummy-variable-rgx = "^(_+|(_+[a-zA-Z0-9_]\*[a-zA-Z0-9]+?))$"

[tool.ruff.format]
preview = true
skip-magic-trailing-comma = false
docstring-code-format = true
quote-style = "double"
indent-style = "space"
line-ending = "auto"

[tool.mypy]
strict = true
warn_unreachable = true
pretty = true
show_column_numbers = true
show_error_codes = true
show_error_context = true

[tool.poetry-dynamic-versioning]
enable = true
vcs = "git"
style = "pep440"

[build-system]
requires = ["poetry-core>=1.0.0", "poetry-dynamic-versioning>=1.0.0,<2.0.0"]
build-backend = "poetry_dynamic_versioning.backend"

================================================
FILE: docs/CNAME
================================================
genjax.gen.dev

================================================
FILE: docs/codeofconduct.md
================================================

---

search:
exclude: true

---

# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, caste, color, religion, or sexual
identity and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming,
diverse, inclusive, and healthy community.

## Our Standards

Examples of behavior that contributes to a positive environment for our
community include:

-   Demonstrating empathy and kindness toward other people
-   Being respectful of differing opinions, viewpoints, and experiences
-   Giving and gracefully accepting constructive feedback
-   Accepting responsibility and apologizing to those affected by our mistakes,
    and learning from the experience
-   Focusing on what is best not just for us as individuals, but for the overall
    community

Examples of unacceptable behavior include:

-   The use of sexualized language or imagery, and sexual attention or advances of
    any kind
-   Trolling, insulting or derogatory comments, and personal or political attacks
-   Public or private harassment
-   Publishing others' private information, such as a physical or email address,
    without their explicit permission
-   Other conduct which could reasonably be considered inappropriate in a
    professional setting

## Enforcement Responsibilities

Community leaders are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in
response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

Community leaders have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct, and will communicate reasons for moderation
decisions when appropriate.

## Scope

This Code of Conduct applies within all community spaces, and also applies when
an individual is officially representing the community in public spaces.
Examples of representing our community include using an official e-mail address,
posting via an official social media account, or acting as an appointed
representative at an online or offline event.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the community leaders responsible for enforcement at
[mccoyb@mit.edu](mailto:mccoyb@mit.edu).
All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the
reporter of any incident.

## Enforcement Guidelines

Community leaders will follow these Community Impact Guidelines in determining
the consequences for any action they deem in violation of this Code of Conduct:

### 1. Correction

**Community Impact**: Use of inappropriate language or other behavior deemed
unprofessional or unwelcome in the community.

**Consequence**: A private, written warning from community leaders, providing
clarity around the nature of the violation and an explanation of why the
behavior was inappropriate. A public apology may be requested.

### 2. Warning

**Community Impact**: A violation through a single incident or series of
actions.

**Consequence**: A warning with consequences for continued behavior. No
interaction with the people involved, including unsolicited interaction with
those enforcing the Code of Conduct, for a specified period of time. This
includes avoiding interactions in community spaces as well as external channels
like social media. Violating these terms may lead to a temporary or permanent
ban.

### 3. Temporary Ban

**Community Impact**: A serious violation of community standards, including
sustained inappropriate behavior.

**Consequence**: A temporary ban from any sort of interaction or public
communication with the community for a specified period of time. No public or
private interaction with the people involved, including unsolicited interaction
with those enforcing the Code of Conduct, is allowed during this period.
Violating these terms may lead to a permanent ban.

### 4. Permanent Ban

**Community Impact**: Demonstrating a pattern of violation of community
standards, including sustained inappropriate behavior, harassment of an
individual, or aggression toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within the
community.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.1, available at
[https://www.contributor-covenant.org/version/2/1/code_of_conduct.html][v2.1].

Community Impact Guidelines were inspired by
[Mozilla's code of conduct enforcement ladder][mozilla coc].

For answers to common questions about this code of conduct, see the FAQ at
[https://www.contributor-covenant.org/faq][faq]. Translations are available at
[https://www.contributor-covenant.org/translations][translations].

[homepage]: https://www.contributor-covenant.org
[v2.1]: https://www.contributor-covenant.org/version/2/1/code_of_conduct.html
[mozilla coc]: https://github.com/mozilla/diversity
[faq]: https://www.contributor-covenant.org/faq
[translations]: https://www.contributor-covenant.org/translations

================================================
FILE: docs/contributing.md
================================================

---

search:
exclude: true

---

# Contributor Guide

Thank you for your interest in improving this project.
This project is open-source under the [Apache 2.0 license] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

Here is a list of important resources for contributors:

-   [Source Code]
-   [Documentation]
-   [Issue Tracker]
-   [Code of Conduct]

[apache 2.0 license]: https://opensource.org/licenses/Apache-2.0
[source code]: https://github.com/genjax-dev/genjax-chi
[documentation]: https://genjax.gen.dev
[issue tracker]: https://github.com/genjax-dev/genjax-chi/issues

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

-   Which operating system and Python version are you using?
-   Which version of this project are you using?
-   What did you do?
-   What did you expect to see?
-   What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## How to set up your development environment

### Dev Container

The easiest way to get started is using the provided Dev Container configuration, which automatically sets up a complete development environment.

#### What gets installed?

-   Python 3.11
-   Poetry (dependency management)
-   All project dependencies (including dev dependencies)
-   Pre-commit hooks
-   VS Code extensions for Python development, formatting, and testing

#### Setup

-   **GitHub Codespaces**

    1. Click "Code" → "Codespaces" → "Codespace repository configuration" (next to "Create codespace on main")
    2. Click "New with options..."
    3. Choose `genjax` configuration (not `genjax-gpu` - Codespaces don't support GPU access)
    4. Wait for the container to build and dependencies to install (~5 minutes)
    5. Start coding - everything is ready to go

-   **Local with VS Code:**

    1. Install [Docker](https://docs.docker.com/get-docker/) and [VS Code Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
    2. Clone the repository
    3. Open in editor (VS Code/Cursor/Windsurf) and click "Reopen in Container" when prompted
    4. Choose either:
        - `base` - Standard development environment
        - `gpu` - Includes GPU access for CUDA workloads

    The setup process runs automatically and takes 2-3 minutes on first launch.

Note: Upon first startup, reload the window so all extensions can properly load now that setup is complete.

### Manual Setup

You need Python 3.7+ (we recommend 3.11+) and the following tools:

-   [Poetry]
-   [Nox]
-   [nox-poetry]

Install the package with development requirements:

```console
$ poetry install
```

You can now run an interactive Python session:

```console
$ poetry run python
```

[poetry]: https://python-poetry.org/
[nox]: https://nox.thea.codes/
[nox-poetry]: https://nox-poetry.readthedocs.io/

## How to test the project

Run the full test suite:

```console
$ nox
```

List the available Nox sessions:

```console
$ nox --list-sessions
```

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

```console
$ nox --session=tests
```

Unit tests are located in the _tests_ directory,
and are written using the [pytest] testing framework.

[pytest]: https://pytest.readthedocs.io/

## How to submit changes

Open a [pull request] to submit changes to this project.

Your pull request needs to meet the following guidelines for acceptance:

-   The Nox test suite must pass without errors and warnings.
-   Include unit tests. This project maintains 100% code coverage.
-   If your changes add functionality, update the documentation accordingly.

Feel free to submit early, though—we can always iterate on this.

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

[pull request]: https://github.com/genjax-dev/genjax-chi/pulls

================================================
FILE: docs/developing.md
================================================

# Developer's Guide

This guide describes how to complete various tasks you'll encounter when working
on the GenJAX codebase.

## Development environment

This project uses:

-   [poetry](https://python-poetry.org/) for dependency management
-   [nox](https://nox.thea.codes/en/stable/) to automate testing/linting/building.
-   [mkdocs](https://www.mkdocs.org/) to generate static documentation.

### Commit Hooks

We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your python is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks,
install `pre-commit` if you don't yet have it. I prefer using
[pipx](https://github.com/pipxproject/pipx) so that `pre-commit` stays globally
available.

```bash
pipx install pre-commit
```

Then install the hooks with this command:

```bash
pre-commit install
```

Now they'll run on every commit. If you want to run them manually, run the
following command:

```bash
pre-commit run --all-files
```

### (Option 1): Development environment setup with `poetry`

#### Step 1: Setting up the environment with `poetry`

[First, install `poetry` to your system.](https://python-poetry.org/docs/#installing-with-the-official-installer)

Assuming you have `poetry`, here's a simple script to setup a compatible
development environment - if you can run this script, you have a working
development environment which can be used to execute tests, build and serve the
documentation, etc.

```bash
conda create --name genjax-py311 python=3.11 --channel=conda-forge
conda activate genjax-py311
pip install nox
pip install nox-poetry
git clone https://github.com/genjax-dev/genjax-chi
cd genjax
poetry self add "poetry-dynamic-versioning[plugin]"
poetry install
poetry run jupyter-lab
```

You can test your environment with:

```bash
nox -r
```

#### Step 2: Choose a `jaxlib`

GenJAX does not manage the version of `jaxlib` that you use in your execution
environment. The exact version of `jaxlib` can change depending upon the target
deployment hardware (CUDA, CPU, Metal). It is your responsibility to install a
version of `jaxlib` which is compatible with the JAX bounds (`jax = "^0.4.24"`
currently) in GenJAX (as specified in `pyproject.toml`).

[For further information, see this discussion.](https://github.com/google/jax/discussions/16380)

[You can likely install CUDA compatible versions by following environment setup above with a `pip` installation of the CUDA-enabled JAX.](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier)

When running any of the `nox` commands, append `-- <jax_specifier>` to install
the proper `jaxlib` into the session. For example,

```sh
nox -s tests -- cpu
```

will run the tests with the CPU `jaxlib` installed, while

```sh
nox -s tests -- cuda12
```

will install the CUDA bindings. By default, the CPU bindings will be installed.

### (Option 2): Self-managed development environment with `requirements.txt`

#### Using `requirements.txt`

> **This is not the recommended way to develop on `genjax`**, but may be
> required if you want to avoid environment collisions with `genjax` installing
> specific versions of `jax` and `jaxlib`.

`genjax` includes a `requirements.txt` file which is exported from the
`pyproject.toml` dependency requirements -- but with `jax` and `jaxlib` removed.

If you wish to setup a usable environment this way, you must ensure that you
have `jax` and `jaxlib` installed in your environment, then:

```bash
pip install -r requirements.txt
```

This should install a working environment - subject to the conditions that your
version of `jax` and `jaxlib` resolve with the versions of packages in the
`requirements.txt`

### Documentation environment setup

GenJAX builds documentation using an insiders-only version of
[mkdocs-material](https://squidfunk.github.io/mkdocs-material/). GenJAX will
attempt to fetch this repository during the documentation build step.

Run the following command to fully build the documentation:

```bash
nox -r -s docs-build
```

This command will use `mkdocs` to build the static site.

To view the generated site, run:

```bash
nox -r -s docs-serve
```

or to run both commands in sequence:

```bash
nox -r -s docs-build-serve
```

## Releasing GenJAX

Published GenJAX artifacts live [on PyPI](https://pypi.org/project/genjax/) and
are published automatically by GitHub with each new
[release](https://github.com/genjax-dev/genjax-chi/releases).

### Release checklist

Before cutting a new release:

-   Update README.md to reference new GenJAX versions
-   Make sure that the referenced `jax` and `jaxlib` versions match the version
    declared in `pyproject.toml`

### Releasing via GitHub

-   Visit https://github.com/genjax-dev/genjax-chi/releases/new to create a new release.
-   From the "Choose a tag" dropdown, type the new version (using the format
    `v<MAJOR>.<MINOR>.<INCREMENTAL>`, like `v0.1.0`) and select "Create new tag
    on publish"
-   Fill out an appropriate title, and add release notes generated by looking at
    PRs merged since the last release
-   Click "Publish Release"

This will build and publish the new version to Artifact Registry.

### Manually publishing to PyPI

To publish a version manually, you'll need to be added to the GenJAX Maintainers
list on PyPI, or ask a [current maintainer from the project
page](https://pypi.org/project/genjax/) for help. Once that's settled:

-   generate an API token on your [pypi account
    page](https://pypi.org/manage/account/token/), scoped to all projects or
    scoped specifically to genjax
-   copy the token and install it on your machine by running the following
    command:

```sh
poetry config pypi-token.pypi <api-token>
```

-   create a new version tag on the `main` branch of the form
    `v<MAJOR>.<MINOR>.<INCREMENTAL>`, like `v0.1.0`, and push the tag to the
    remote repository:

```sh
git tag v0.1.0
git push --tags
```

-   use Poetry to build and publish the artifact to pypi:

```sh
poetry publish --build
```

================================================
FILE: docs/index.md
================================================

#

<p align="center">
<img width="500px" src="./assets/img/logo.png"/>
</p>
<p align="center">
  <strong>
    Probabilistic programming with (parallel & differentiable) programmable inference.
  </strong>
</p>

## 🔎 What is GenJAX?

Gen is a multi-paradigm (generative, differentiable, incremental) language for probabilistic programming focused on [**generative functions**: computational objects which represent probability measures over structured sample spaces](https://genjax.gen.dev/cookbook/active/intro.html#generative-functions).

GenJAX is an implementation of Gen on top of [JAX](https://github.com/google/jax) - exposing the ability to programmatically construct and manipulate generative functions, as well as [JIT compile + auto-batch inference computations using generative functions onto GPU devices](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html).

<div align="center">
<a href="https://genjax.gen.dev/cookbook/">Jump into the notebooks!</a>
<br>
<br>
</div>

> GenJAX is part of a larger ecosystem of probabilistic programming tools based upon Gen. [Explore more...](https://www.gen.dev/)

## Quickstart

To install GenJAX, run

```bash
pip install genjax
```

Then install [JAX](https://github.com/google/jax) using [this
guide](https://jax.readthedocs.io/en/latest/installation.html) to choose the
command for the architecture you're targeting. To run GenJAX without GPU
support:

```sh
pip install jax[cpu]~=0.4.24
```

On a Linux machine with a GPU, run the following command:

```sh
pip install jax[cuda12]~=0.4.24
```

### Quick example [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://console.cloud.google.com/vertex-ai/colab/notebooks?project=probcomp-caliban&activeNb=projects%2Fprobcomp-caliban%2Flocations%2Fus-west1%2Frepositories%2F09be0f8e-ccfd-4d34-a029-fed94d455c48)

The following code snippet defines a generative function called `beta_bernoulli` that

-   takes a shape parameter `beta`
-   uses this to create and draw a value `p` from a [Beta
    distribution](https://en.wikipedia.org/wiki/Beta_distribution)
-   Flips a coin that returns 1 with probability `p`, 0 with probability `1-p` and
    returns that value

Then, we create an inference problem (by specifying a posterior target), and utilize sampling
importance resampling to give produce single sample estimator of `p`.

We can JIT compile that entire process, run it in parallel, etc - which we utilize to produce an estimate for `p`
over 50 independent trials of SIR (with K = 50 particles).

```python
import jax
import jax.numpy as jnp
import genjax
from genjax import beta, flip, gen, Target, ChoiceMap
from genjax.inference.smc import ImportanceK

# Create a generative model.
@gen
def beta_bernoulli(α, β):
    p = beta(α, β) @ "p"
    v = flip(p) @ "v"
    return v

@jax.jit
def run_inference(obs: bool):
    # Create an inference query - a posterior target - by specifying
    # the model, arguments to the model, and constraints.
    posterior_target = Target(beta_bernoulli, # the model
                              (2.0, 2.0), # arguments to the model
                              ChoiceMap.d({"v": obs}), # constraints
                            )

    # Use a library algorithm, or design your own - more on that in the docs!
    alg = ImportanceK(posterior_target, k_particles=50)

    # Everything is JAX compatible by default.
    # JIT, vmap, to your heart's content.
    key = jax.random.key(314159)
    sub_keys = jax.random.split(key, 50)
    _, p_chm = jax.vmap(alg.random_weighted, in_axes=(0, None))(
        sub_keys, posterior_target
    )

    # An estimate of `p` over 50 independent trials of SIR (with K = 50 particles).
    return jnp.mean(p_chm["p"])

(run_inference(True), run_inference(False))
```

```python
(Array(0.6039314, dtype=float32), Array(0.3679334, dtype=float32))
```

## References

Many bits of knowledge have gone into this project -- [you can find many of these bits at the MIT Probabilistic Computing Project page](http://probcomp.csail.mit.edu/) under publications. Here's an abbreviated list of high value references:

-   [Marco Cusumano-Towner's thesis on Gen][marco_thesis]
-   [The main Gen.jl repository][gen_jl]
-   (Trace types) [(Lew et al) trace types][trace_types]
-   (RAVI) [(Lew et al) Recursive auxiliary-variable inference][ravi]
-   (GenSP) [Alex Lew's Gen.jl implementation of GenSP][gen_sp]
-   (ADEV) [(Lew & Huot, et al) Automatic differentiation of expected values of probabilistic programs][adev]

### JAX influences

This project has several JAX-based influences. Here's an abbreviated list:

-   [This notebook on static dispatch (Dan Piponi)][effect_handling_interp]
-   [Equinox (Patrick Kidger's work on neural networks via callable Pytrees)][equinox]
-   [Oryx (interpreters and interpreter design)][oryx]

### Acknowledgements

The maintainers of this library would like to acknowledge the JAX and Oryx maintainers for useful discussions and reference code for interpreter-based transformation patterns.

---

<div align="center">
Created and maintained by the <a href="http://probcomp.csail.mit.edu/">MIT Probabilistic Computing Project</a>. All code is licensed under the <a href="LICENSE">Apache 2.0 License</a>.
</div>

[actions]: https://github.com/genjax-dev/genjax-chi/actions
[adev]: https://arxiv.org/abs/2212.06386
[cookbook]: https://genjax.gen.dev/cookbook/
[coverage_badge]: https://github.com/genjax-dev/genjax-chi/coverage.svg
[effect_handling_interp]: https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=ukjVJ2Ls_6Q3
[equinox]: https://github.com/patrick-kidger/equinox
[gen_jl]: https://github.com/probcomp/Gen.jl
[gen_sp]: https://github.com/probcomp/GenSP.jl
[jax_badge]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq5MTYAhOxXM%2FmvEpsoWmtfP9RxnkAIAr%2F5pVxqPxH93msKodRSXIct2l0OU0%2FL4eY506L%2B3GyJ6UMEZfjjCDbysNcWWmFweJP0Jz%2FA0g2gk80pGkYAAAAAElFTkSuQmCC
[main_build_action_badge]: https://github.com/genjax-dev/genjax-chi/actions/workflows/ci.yml/badge.svg?style=flat-square&branch=main
[main_build_status_url]: https://github.com/genjax-dev/genjax-chi/actions/workflows/ci.yml?query=branch%3Amain
[marco_thesis]: https://www.mct.dev/assets/mct-thesis.pdf
[oryx]: https://github.com/jax-ml/oryx
[ravi]: https://arxiv.org/abs/2203.02836
[trace_types]: https://dl.acm.org/doi/10.1145/3371087

================================================
FILE: docs/license.md
================================================

---

search:
exclude: true

---

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1.  Definitions.

    "License" shall mean the terms and conditions for use, reproduction,
    and distribution as defined by Sections 1 through 9 of this document.

    "Licensor" shall mean the copyright owner or entity authorized by
    the copyright owner that is granting the License.

    "Legal Entity" shall mean the union of the acting entity and all
    other entities that control, are controlled by, or are under common
    control with that entity. For the purposes of this definition,
    "control" means (i) the power, direct or indirect, to cause the
    direction or management of such entity, whether by contract or
    otherwise, or (ii) ownership of fifty percent (50%) or more of the
    outstanding shares, or (iii) beneficial ownership of such entity.

    "You" (or "Your") shall mean an individual or Legal Entity
    exercising permissions granted by this License.

    "Source" form shall mean the preferred form for making modifications,
    including but not limited to software source code, documentation
    source, and configuration files.

    "Object" form shall mean any form resulting from mechanical
    transformation or translation of a Source form, including but
    not limited to compiled object code, generated documentation,
    and conversions to other media types.

    "Work" shall mean the work of authorship, whether in Source or
    Object form, made available under the License, as indicated by a
    copyright notice that is included in or attached to the work
    (an example is provided in the Appendix below).

    "Derivative Works" shall mean any work, whether in Source or Object
    form, that is based on (or derived from) the Work and for which the
    editorial revisions, annotations, elaborations, or other modifications
    represent, as a whole, an original work of authorship. For the purposes
    of this License, Derivative Works shall not include works that remain
    separable from, or merely link (or bind by name) to the interfaces of,
    the Work and Derivative Works thereof.

    "Contribution" shall mean any work of authorship, including
    the original version of the Work and any modifications or additions
    to that Work or Derivative Works thereof, that is intentionally
    submitted to Licensor for inclusion in the Work by the copyright owner
    or by an individual or Legal Entity authorized to submit on behalf of
    the copyright owner. For the purposes of this definition, "submitted"
    means any form of electronic, verbal, or written communication sent
    to the Licensor or its representatives, including but not limited to
    communication on electronic mailing lists, source code control systems,
    and issue tracking systems that are managed by, or on behalf of, the
    Licensor for the purpose of discussing and improving the Work, but
    excluding communication that is conspicuously marked or otherwise
    designated in writing by the copyright owner as "Not a Contribution."

    "Contributor" shall mean Licensor and any individual or Legal Entity
    on behalf of whom a Contribution has been received by Licensor and
    subsequently incorporated within the Work.

2.  Grant of Copyright License. Subject to the terms and conditions of
    this License, each Contributor hereby grants to You a perpetual,
    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
    copyright license to reproduce, prepare Derivative Works of,
    publicly display, publicly perform, sublicense, and distribute the
    Work and such Derivative Works in Source or Object form.

3.  Grant of Patent License. Subject to the terms and conditions of
    this License, each Contributor hereby grants to You a perpetual,
    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
    (except as stated in this section) patent license to make, have made,
    use, offer to sell, sell, import, and otherwise transfer the Work,
    where such license applies only to those patent claims licensable
    by such Contributor that are necessarily infringed by their
    Contribution(s) alone or by combination of their Contribution(s)
    with the Work to which such Contribution(s) was submitted. If You
    institute patent litigation against any entity (including a
    cross-claim or counterclaim in a lawsuit) alleging that the Work
    or a Contribution incorporated within the Work constitutes direct
    or contributory patent infringement, then any patent licenses
    granted to You under this License for that Work shall terminate
    as of the date such litigation is filed.

4.  Redistribution. You may reproduce and distribute copies of the
    Work or Derivative Works thereof in any medium, with or without
    modifications, and in Source or Object form, provided that You
    meet the following conditions:

    (a) You must give any other recipients of the Work or
    Derivative Works a copy of this License; and

    (b) You must cause any modified files to carry prominent notices
    stating that You changed the files; and

    (c) You must retain, in the Source form of any Derivative Works
    that You distribute, all copyright, patent, trademark, and
    attribution notices from the Source form of the Work,
    excluding those notices that do not pertain to any part of
    the Derivative Works; and

    (d) If the Work includes a "NOTICE" text file as part of its
    distribution, then any Derivative Works that You distribute must
    include a readable copy of the attribution notices contained
    within such NOTICE file, excluding those notices that do not
    pertain to any part of the Derivative Works, in at least one
    of the following places: within a NOTICE text file distributed
    as part of the Derivative Works; within the Source form or
    documentation, if provided along with the Derivative Works; or,
    within a display generated by the Derivative Works, if and
    wherever such third-party notices normally appear. The contents
    of the NOTICE file are for informational purposes only and
    do not modify the License. You may add Your own attribution
    notices within Derivative Works that You distribute, alongside
    or as an addendum to the NOTICE text from the Work, provided
    that such additional attribution notices cannot be construed
    as modifying the License.

    You may add Your own copyright statement to Your modifications and
    may provide additional or different license terms and conditions
    for use, reproduction, or distribution of Your modifications, or
    for any such Derivative Works as a whole, provided Your use,
    reproduction, and distribution of the Work otherwise complies with
    the conditions stated in this License.

5.  Submission of Contributions. Unless You explicitly state otherwise,
    any Contribution intentionally submitted for inclusion in the Work
    by You to the Licensor shall be under the terms and conditions of
    this License, without any additional terms or conditions.
    Notwithstanding the above, nothing herein shall supersede or modify
    the terms of any separate license agreement you may have executed
    with Licensor regarding such Contributions.

6.  Trademarks. This License does not grant permission to use the trade
    names, trademarks, service marks, or product names of the Licensor,
    except as required for reasonable and customary use in describing the
    origin of the Work and reproducing the content of the NOTICE file.

7.  Disclaimer of Warranty. Unless required by applicable law or
    agreed to in writing, Licensor provides the Work (and each
    Contributor provides its Contributions) on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
    implied, including, without limitation, any warranties or conditions
    of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
    PARTICULAR PURPOSE. You are solely responsible for determining the
    appropriateness of using or redistributing the Work and assume any
    risks associated with Your exercise of permissions under this License.

8.  Limitation of Liability. In no event and under no legal theory,
    whether in tort (including negligence), contract, or otherwise,
    unless required by applicable law (such as deliberate and grossly
    negligent acts) or agreed to in writing, shall any Contributor be
    liable to You for damages, including any direct, indirect, special,
    incidental, or consequential damages of any character arising as a
    result of this License or out of the use or inability to use the
    Work (including but not limited to damages for loss of goodwill,
    work stoppage, computer failure or malfunction, or any and all
    other commercial damages or losses), even if such Contributor
    has been advised of the possibility of such damages.

9.  Accepting Warranty or Additional Liability. While redistributing
    the Work or Derivative Works thereof, You may choose to offer,
    and charge a fee for, acceptance of support, warranty, indemnity,
    or other liability obligations and/or rights consistent with this
    License. However, in accepting such obligations, You may act only
    on Your own behalf and on Your sole responsibility, not on behalf
    of any other Contributor, and only if You agree to indemnify,
    defend, and hold each Contributor harmless for any liability
    incurred by, or claims asserted against, such Contributor by reason
    of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

Copyright 2024 MIT Probabilistic Computing Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

================================================
FILE: docs/assets/font/BerkeleyMonoVariable-Regular.woff2
================================================
[Binary file]

================================================
FILE: docs/cookbook/active/choice_maps.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

# Choice maps [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/active/choice_maps.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

import jax
import jax.numpy as jnp
import jax.random as random

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import (
bernoulli,
beta,
gen,
mix,
normal,
or_else,
pretty,
repeat,
scan,
vmap,
)

pretty()
key = random.key(0)

"""
Choice maps are dictionary-like data structures that accumulate the random choices produced by generative functions which are `traced` by the system, i.e. that are indicated by `@ "p"` in generative functions.

They also serve as a set of constraints/observations when one tries to do inference: given the constraints, inference provides plausible value to complete a choice map to a full trace of a generative model (one value per traced random sample).
"""

@gen
def beta_bernoulli_process(u):
p = beta(1.0, u) @ "p"
v = bernoulli(p) @ "v"
return 2 \* v

"""
Simulating from a model produces a traces which contains a choice map.
"""

key, subkey = jax.random.split(key)
trace = jax.jit(beta_bernoulli_process.simulate)(subkey, (0.5,))

"""
From that trace, we can recover the choicemap with either of the two equivalent methods:
"""

trace.get_choices(), trace.get_choices()

"""
We can also print specific subparts of the choice map.
"""

trace.get_choices()["p"]

"""
Then, we can create a choice map of observations and perform diverse operations on it.
We can set the value of an address in the choice map.
For instance, we can add two choicemaps together, which behaves similarly to the union of two dictionaries.
"""

chm = C["p"].set(0.5) | C["v"].set(1)
chm

"""
A couple of extra ways to achieve the same result.
"""

chm_equiv_1 = (
C["p"].set(0.5).at["v"].set(1)
) # the at/set notation mimics JAX's array update pattern
chm_equiv_2 = C.d({"p": 0.5, "v": 1}) # creates a dictionary directly
assert chm == chm_equiv_1 == chm_equiv_2

"""
This also works for hierarchical addresses:
"""

chm = C["p", "v"].set(1)

# equivalent to

eq_chm = C.d({"p": C.d({"v": 1})})
assert chm == eq_chm
chm

"""
We can also directly set a value in the choice_map
"""

chm = C.v(5.0)
chm

"""
We can also create an empty choice_map
"""

chm = C.n()
chm

"""
Other examples of Choice map creation include iteratively adding choices to a choice map.
"""

chm = C.n()
for i in range(10):
chm = chm ^ C["p" + str(i)].set(i)

"""
An equivalent, more JAX-friendly way to do this
"""

chm = jax.vmap(lambda idx: C[idx].set(idx.astype(float)))(jnp.arange(10))

"""
And in fact, we can directly use the numpy notation to create a choice map.
"""

chm = C[:].set(jnp.arange(10.0))
chm

"""
For a nested vmap combinator, the creation of a choice map can be a bit more tricky.
"""

sample_image = genjax.vmap(in_axes=(0,))(
genjax.vmap(in_axes=(0,))(gen(lambda pixel: normal(pixel, 1.0) @ "new_pixel"))
)

image = jnp.zeros([4, 4], dtype=jnp.float32)
key, subkey = jax.random.split(key)
trace = sample_image.simulate(subkey, (image,))
trace.get_choices()

"""
Creating a few values for the choice map is simple.
"""

chm = C[1, 2, "new_pixel"].set(1.0) ^ C[0, 2, "new_pixel"].set(1.0)

key, subkey = jax.random.split(key)
tr, w = jax.jit(sample_image.importance)(subkey, chm, (image,))
w

"""
But because of the nested `vmap`, the address hierarchy can sometimes lead to unintuitive results, e.g. as there is no bound check on the address. We seemingly adding a new constraint but we obtain the same weight as before, meaning that the new choice was not used for inference.
"""

chm = chm ^ C[1, 5, "new_pixel"].set(1.0)
tr, w = jax.jit(sample_image.importance)(
subkey, chm, (image,)
) # reusing the key to make comparisons easier
w

"""
A different way to create a choicemap that is compatible with the nested vmap in this case.
"""

chm = C[:, :, "new_pixel"].set(jnp.ones((4, 4), dtype=jnp.float32))
key, subkey = jax.random.split(key)
tr, w = jax.jit(sample_image.importance)(subkey, chm, (image,))
w

"""
More generally, some combinators introduce an `Indexed` choicemap.
These are mainly `vmap, scan` as well as those derived from these 2, such as `iterate, repeat`.
An `Indexed` choicemap introduced an integer in the hierarchy of addresses, as the place where the combinator is introduced.
For instance:
"""

@genjax.gen
def submodel():
x = genjax.exponential.vmap()(1.0 + jnp.arange(50, dtype=jnp.float32)) @ "x"
return x

@genjax.gen
def model():
xs = submodel.repeat(n=5)() @ "xs"
return xs

key, subkey = jax.random.split(key)
tr = model.simulate(subkey, ())
chm = tr.get_choices()
chm

"""
In this case, we can create a hierarchical choicemap as follows:
"""

chm = C["xs", :, "x", :].set(jnp.ones((5, 50)))
key, subkey = jax.random.split(key)
model.importance(subkey, chm, ())

"""
We can also construct an indexed choicemap with more than one variable in it using the following syntax:
"""

\_phi, \_q, \_beta, \_r = (0.9, 1.0, 0.5, 1.0)

@genjax.gen
def step(state):
x*prev, z_prev = state
x = genjax.normal(\_phi * x_prev, \_q) @ "x"
z = \_beta \* z_prev + x

-   = genjax.normal(z, \_r) @ "y"
    return (x, z)

max_T = 20
model = step.iterate_final(n=max_T)

x*range = 1.0 * jnp.where(
(jnp.arange(20) >= 10) & (jnp.arange(20) < 15), jnp.arange(20) + 1, jnp.arange(20)
)
y*range = 1.0 * jnp.where(
(jnp.arange(20) >= 15) & (jnp.arange(20) < 20), jnp.arange(20) + 1, jnp.arange(20)
)
xy = C["x"].set(x_range).at["y"].set(y_range)
chm4 = C[jnp.arange(20)].set(xy)
chm4
key, subkey = jax.random.split(key)
model.importance(subkey, chm4, ((0.5, 0.5),))

"""
Accessing the right elements in the trace can become non-trivial when one creates hierarchical generative functions.
Here are minimal examples and solutions for selection.
"""

# For `or_else` combinator

@gen
def model(p):
branch_1 = gen(lambda p: bernoulli(p) @ "v1")
branch_2 = gen(lambda p: bernoulli(-p) @ "v2")
v = or_else(branch_1, branch_2)(p > 0, (p,), (p,)) @ "s"
return v

key, subkey = jax.random.split(key)
trace = jax.jit(model.simulate)(subkey, (0.5,))
trace.get_choices()["s", "v1"]

# For `vmap` combinator

sample_image = vmap(in_axes=(0,))(
vmap(in_axes=(0,))(gen(lambda pixel: normal(pixel, 1.0) @ "new_pixel"))
)

image = jnp.zeros([2, 3], dtype=jnp.float32)
key, subkey = jax.random.split(key)
trace = sample_image.simulate(subkey, (image,))
trace.get_choices()[:, :, "new_pixel"]

# For `scan_combinator`

@scan(n=10)
@gen
def hmm(x, c):
z = normal(x, 1.0) @ "z"
y = normal(z, 1.0) @ "y"
return y, None

key, subkey = jax.random.split(key)
trace = hmm.simulate(subkey, (0.0, None))
trace.get_choices()[:, "z"], trace.get_choices()[3, "y"]

# For `repeat_combinator`

@repeat(n=10)
@gen
def model(y):
x = normal(y, 0.01) @ "x"
y = normal(x, 0.01) @ "y"
return y

key, subkey = jax.random.split(key)
trace = model.simulate(subkey, (0.3,))
trace.get_choices()[:, "x"]

# For `mixture_combinator`

@gen
def mixture_model(p):
z = normal(p, 1.0) @ "z"
logits = (0.3, 0.5, 0.2)
arg_1 = (p,)
arg_2 = (p,)
arg_3 = (p,)
a = (
mix(
gen(lambda p: normal(p, 1.0) @ "x1"),
gen(lambda p: normal(p, 2.0) @ "x2"),
gen(lambda p: normal(p, 3.0) @ "x3"),
)(logits, arg_1, arg_2, arg_3)
@ "a"
)
return a + z

key, subkey = jax.random.split(key)
trace = mixture_model.simulate(subkey, (0.4,))

# The combinator uses a fixed address "mixture_component" for the components of the mixture model.

trace.get_choices()["a", "mixture_component"]

"""
Similarly, if traces were created as a batch using `jax.vmap`, in general it will not create a valid batched trace, e.g. the score will not be defined as a single float. It can be very useful for inference though.
"""

@genjax.gen
def random*walk_step(prev, *):
x = genjax.normal(prev, 1.0) @ "x"
return x, None

random_walk = random_walk_step.scan(n=1000)

init = 0.5
keys = jax.random.split(key, 10)

trs = jax.vmap(random_walk.simulate, (0, None))(keys, (init, None))
try:
if isinstance(trs.get_score(), float):
trs.get_score()
else:
raise ValueError("Expected a float value for the score.")
except Exception as e:
print(e)

"""
However, with a little extra step we can recover information in individual traces.
"""

jax.vmap(lambda tr: tr.get_choices())(trs)

"""
Note that this limitation is dependent on the model, and the simpler thing may work anyway for some classes' models.
"""

jitted = jax.jit(jax.vmap(model.simulate, in_axes=(0, None)))
keys = random.split(key, 10)
traces = jitted(keys, (0.5,))

traces.get_choices()

================================================
FILE: docs/cookbook/active/debugging.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

# Debugging [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/active/debugging.ipynb)

How can I debug my code? I want to add break points or print statements in my Jax/GenJax code but it doesn't seem to work because of traced values and/or jit compilation.
"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

import jax

from genjax import bernoulli, beta, gen

key = jax.random.key(0)

"""
TLDR: inside of generative functions, use `jax.debug.print`
and `jax.debug.breakpoint()` instead of `print()` statements.
We also recommend looking at the official JAX debug doc which applies to GenJAX as well:
https://jax.readthedocs.io/en/latest/debugging/print_breakpoint.html
"""

"""
Example of printing
"""

@gen
def beta_bernoulli_process(u):
p = beta(0.0, u) @ "p"
v = bernoulli(p) @ "v"
print("Bad looking printing:", v) # will print a traced Value, not what you want
jax.debug.print("Better looking printing: {v}", v=v)
return v

non_jitted = beta_bernoulli_process.simulate
key, subkey = jax.random.split(key)
tr = non_jitted(subkey, (1.0,))
key, subkey = jax.random.split(key)
jitted = jax.jit(beta_bernoulli_process.simulate)
tr = jitted(subkey, (1.0,))

"""
Inside generative functions, `jax.debug.print` is available and compatible with all the JAX transformations and higher-order functions like `jax.jit`, `jax.grad`, `jax.vmap`, `jax.lax.scan`, etc.
"""

"""
Running the cell below will open a pdb-like interface in the terminal where you can inspect the values of the variables in the scope of the breakpoint.
You can continue the execution of the program by typing c and pressing Enter. You can also inspect the values of the variables in the scope of the breakpoint by typing the name of the variable and pressing Enter. You can exit the breakpoint by typing q and pressing Enter. You can see the commands available in the breakpoint by typing h and pressing Enter.
It also works with jitted functions, but may affect performance.
It is compatible with all the JAX transformations and higher-order functions too but you can expect some sharp edges.

```python
# Example of breakpoint
@gen
def beta_bernoulli_process(u):
    p = beta(0.0, u) @ "p"
    v = bernoulli(p) @ "v"
    jax.debug.breakpoint()
    return v


non_jitted = beta_bernoulli_process.simulate
key, subkey = jax.random.split(key)
tr = non_jitted(subkey, (1.0,))
```

"""

================================================
FILE: docs/cookbook/active/generative_function_interface.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

# The generative function interface [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/active/generative_function_interface.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

import jax
from jax import jit

from genjax import ChoiceMapBuilder as C
from genjax import (
Diff,
NoChange,
UnknownChange,
bernoulli,
beta,
gen,
pretty,
)
from genjax.\_src.generative_functions.static import MissingAddress

key = jax.random.key(0)
pretty()

# Define a generative function

@gen
def beta_bernoulli_process(u):
p = beta(1.0, u) @ "p"
v = bernoulli(p) @ "v"
return 2 \* v

"""

1. Generate a traced sample and constructs choicemaps
   """

"""
There's an entire cookbook entry on this in `choicemap_creation_selection`.
"""

key, subkey = jax.random.split(key)
trace = jax.jit(beta_bernoulli_process.simulate)(subkey, (0.5,))

""" 2) Compute log probabilities
"""

"""
2.1 Print the log probability of the trace
"""

trace.get_score()

"""
2.2 Print the log probability of an observation encoded as a ChoiceMap under the model

It returns both the log probability and the return value
"""

chm = C["p"].set(0.5) ^ C["v"].set(1)
args = (0.5,)
beta_bernoulli_process.assess(chm, args)

"""
Note that the ChoiceMap should be complete, i.e. all random choices should be observed
"""

chm_2 = C["v"].set(1)
try:
beta_bernoulli_process.assess(chm_2, args)
except MissingAddress as e:
print(e)

""" 3) Generate a sample conditioned on the observations
"""

"""
We can also use a partial ChoiceMap as a constraint/observation and generate a full trace with these constraints.
"""

key, subkey = jax.random.split(key)
partial_chm = C["v"].set(1) # Creates a ChoiceMap of observations
args = (0.5,)
trace, weight = beta_bernoulli_process.importance(
subkey, partial_chm, args
) # Runs importance sampling

"""
This returns a pair containing the new trace and the log probability of produced trace under the model
"""

trace.get_choices()

weight

""" 4) Update a trace.
"""

"""
We can also update a trace. This is for instance useful as a performance optimization in Metropolis-Hastings algorithms where often most of the trace doesn't change between time steps.

We first define a model for which changing the argument will force a change in the trace.
"""

@gen
def beta_bernoulli_process(u):
p = beta(1.0, u) @ "p"
v = bernoulli(p) @ "v"
return 2 \* v

"""
We then create an trace to be updated and constraints.
"""

key, subkey = jax.random.split(key)
jitted = jit(beta_bernoulli_process.simulate)
old_trace = jitted(subkey, (1.0,))
constraint = C["v"].set(1)

"""
Now the update uses a form of incremental computation.
It works by tracking the differences between the old new values for arguments.
Just like for differentiation, it can be achieved by providing for each argument a tuple containing the new value and its change compared to the old value.
"""

"""
If there's no change for an argument, the change is set to NoChange.
"""

arg_diff = (Diff(1.0, NoChange),)

"""
If there is any change, the change is set to UnknownChange.
"""

arg_diff = (Diff(5.0, UnknownChange),)

"""
We finally use the update method by passing it a key, the trace to be updated, and the update to be performed.
"""

jitted_update = jit(beta_bernoulli_process.update)

key, subkey = jax.random.split(key)
new_trace, weight_diff, ret_diff, discard_choice = jitted_update(
subkey, old_trace, constraint, arg_diff
)

"""
We can compare the old and new values for the samples and notice that they have not changed.
"""

old_trace.get_choices() == new_trace.get_choices()

"""
We can also see that the weight has changed. In fact we can check that the following relation holds `new_weight` = `old_weight` + `weight_diff`.
"""

weight_diff, old_trace.get_score() + weight_diff == new_trace.get_score()

""" 5. A few more convenient methods
"""

"""
5.1 `propose`

It uses the same inputs as `simulate` but returns the sample, the score and the return value
"""

key, subkey = jax.random.split(key)
sample, score, retval = jit(beta_bernoulli_process.propose)(subkey, (0.5,))
sample, score, retval

"""
5.2 `get_gen_fn`

It returns the generative function that produced the trace.
"""

trace.get_gen_fn()

"""
5.3 `get_args`

It returns the arguments passed to the generative function used to produce the trace
"""

trace.get_args()

"""
5.4 `get_subtrace`

It takes a `StaticAddress` as argument and returns the sub-trace of a trace rooted at these addresses
"""

subtrace = trace.get_subtrace("p")
subtrace, subtrace.get_choices()

================================================
FILE: docs/cookbook/active/intro.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

# Introduction [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/active/intro.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

"""
GenJAX is a swiss army knife for probabilistic machine learning: it's designed to support probabilistic modeling workflows, and to make the resulting code extremely fast and parallelizable via JAX.

In this introduction, we'll focus on one such workflow: writing a latent variable model (we often say: a generative model) which describes a probability distribution over latent variables and data, and then asking questions about the conditional distribution over the latent variables given data.

In the following, we'll often shorten GenJAX to Gen -- because [GenJAX implements Gen](https://www.gen.dev/).

"""

import genstudio.plot as Plot
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from jax import jit, vmap
from jax import random as jrand

import genjax
from genjax import gen, normal, pretty

sns.set_theme(style="white")
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["savefig.transparent"] = True
%config InlineBackend.figure_format = 'svg'

pretty() # pretty print the types

"""

## Generative functions

"""

@gen
def model():
x = normal(0.0, 1.0) @ "x"
normal(x, 1.0) @ "y"

model

"""
In Gen, probabilistic models are represented by a computational object called _a generative function_. Once we create one of these objects, we can use one of several interfaces to gain access to probabilistic effects.

Here's one interface: `simulate` -- this samples from the probability distribution which the program represents, and stores the result, along with other data about the invocation of the function, in a data structure called a `Trace`.
"""

key = jrand.key(0)
tr = model.simulate(key, ())
tr

"""
We can dig around in this object uses its interfaces:
"""

chm = tr.get_choices()
chm

"""
A `ChoiceMap` is a representation of _the sample_ from the probability distribution which the generative function represents. We can ask _what values were sampled_ at the addresses (the `"x"` and `"y"` syntax in our model):
"""

(chm["x"], chm["y"])

"""
Neat -- all of our interfaces are JAX compatible, so we could sample 1000 times just by using `jax.vmap`:
"""

sub_keys = jrand.split(jrand.key(0), 1000)
tr = jit(vmap(model.simulate, in_axes=(0, None)))(sub_keys, ())
tr

"""
Let's plot our samples to get a sense of the distribution we wrote down.
"""

chm = tr.get_choices()
Plot.dot({"x": chm["x"], "y": chm["y"]})

"""
Traces also keep track of other data, like _the score_ of the execution (which is a value which estimates the joint probability of the random choices under the distribution):
"""

tr.get_score()

"""

## Composition of generative functions

Generative functions are probabilistic building blocks. You can combine them into larger probability distributions:
"""

# A regression distribution.

@gen
def regression(x, coefficients, sigma):
basis_value = jnp.array([1.0, x, x**2])
polynomial_value = jnp.sum(basis_value \* coefficients)
y = genjax.normal(polynomial_value, sigma) @ "v"
return y

# Regression, with an outlier random variable.

@gen
def regression_with_outlier(x, coefficients):
is_outlier = genjax.flip(0.1) @ "is_outlier"
sigma = jnp.where(is_outlier, 30.0, 0.3)
is_outlier = jnp.array(is_outlier, dtype=int)
return regression(x, coefficients, sigma) @ "y"

# The full model, sample coefficients for a curve, and then use

# them in independent draws from the regression submodel.

@gen
def full_model(xs):
coefficients = (
genjax.mv_normal(
jnp.zeros(3, dtype=float),
2.0 \* jnp.identity(3),
)
@ "alpha"
)
ys = regression_with_outlier.vmap(in_axes=(0, None))(xs, coefficients) @ "ys"
return ys

"""
Now, let's examine a sample from this model:
"""

data = jnp.arange(0, 10, 0.5)
full_model.simulate(key, (data,)).get_choices()["ys", :, "y", "v"]

"""
We can plot a few such samples.
"""

key, \*sub_keys = jrand.split(key, 10)
traces = vmap(lambda k: full_model.simulate(k, (data,)))(jnp.array(sub_keys))
ys = traces.get_choices()["ys", :, "y", "v"]

(
Plot.dot(
Plot.dimensions(ys, ["sample", "ys"], leaves="y"),
{"x": Plot.repeat(data), "y": "y", "facetGrid": "sample"},
) + Plot.frame()
)

"""
These are samples from the distribution _over curves_ which our generative function represents.

## Inference in generative functions

So we've written a regression model, a distribution over curves. Our model includes an outlier component. If we observe some data for `"y"`, can we predict which points might be outliers?
"""

x = jnp.array([0.3, 0.7, 1.1, 1.4, 2.3, 2.5, 3.0, 4.0, 5.0])
y = 2.0 \* x + 1.5 + x\*\*2
y = y.at[2].set(50.0)
y

"""
We've explored how generative functions represent joint distributions over random variables, but what about distributions induced by inference problems?

We can create an inference problem by pairing a generative function with arguments, and _a constraint_.

First, let's learn how to create one type of constraint -- a _choice map_ sample, just like the choice maps we saw earlier.
"""

from genjax import ChoiceMapBuilder as C

chm = C["ys", :, "y", "v"].set(y)
chm["ys", :, "y", "v"]

"""
The choice map holds the _value constraint_ for the distributions we used in our generative function. Choice maps are a lot like arrays, with a bit of extra metadata.

Now, we can specify an inference target.
"""

from genjax import Target

target = Target(full_model, (x,), chm)
target

"""
A `Target` represents an unnormalized distribution -- in this case, the posterior of the distribution represented by our generative function with arguments `args = (x, )`.

Now, we can approximate the solution to the inference problem using an inference algorithm. GenJAX exposes a standard library of approximate inference algorithms: let's use $K$-particle importance sampling for this one.
"""

from genjax.inference.smc import ImportanceK

alg = ImportanceK(target, k_particles=100)
alg

sub_keys = jrand.split(key, 50)
posterior_samples = jit(vmap(alg(target)))(sub_keys)

"""
With samples from our approximate posterior in hand, we can check queries like "estimate the probability that a point is an outlier":
"""

posterior_samples["ys", :, "is_outlier"]

"""
Here, we see that our approximate posterior assigns high probability to the query "the 3rd data point is an outlier". Remember, we set this point to be far away from the other points.
"""

posterior_samples["ys", :, "is_outlier"].mean(axis=0)

"""
We can also plot the sampled curves against the data.
"""

def polynomial_at_x(x, coefficients):
basis_values = jnp.array([1.0, x, x**2])
polynomial_value = jnp.sum(coefficients \* basis_values)
return polynomial_value

jitted = jit(vmap(polynomial_at_x, in_axes=(None, 0)))

coefficients = posterior_samples["alpha"]
evaluation_points = jnp.arange(0, 5, 0.01)

points = [(x, y) for x in evaluation_points for y in jitted(x, coefficients).tolist()]
(
Plot.dot(points, fill="gold", opacity=0.25, r=0.5) + Plot.dot({"x": x, "y": y}) + Plot.frame()
)

"""

## Summary

We’ve covered a lot of ground in this notebook. Please reflect, re-read, and post issues!

-   We discussed generative functions - the main computational object of Gen, and how these objects represent probability distributions.
-   We showed how to create generative functions.
-   We showed how to use interfaces on generative functions to compute with common operations on distributions.
-   We created a generative function to model a data-generating process based on sampling and evaluating random polynomials on input data - representing regression task.
-   We showed how to create _inference problems_ from generative functions.
-   We created an inference problem from our regression model.
-   We showed how to create approximate inference solutions to inference problems, and sample from them.
-   We investigated the approximate posterior samples, and visually inspected that they match the inferences that we might draw - both for the polynomials we expected to produce the data, as well as what data points might be outliers.

This is just the beginning! There’s a lot more to learn, but this is plenty to chew (for now).

"""

================================================
FILE: docs/cookbook/active/jax_basics.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

# JAX Basics [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/active/jax_basics.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

import multiprocessing
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import beta, gen, pretty

key = jax.random.key(0)
pretty()

"""

1. JAX expects arrays/tuples everywhere
   """

@gen
def f(p):
v = genjax.bernoulli(probs=p) @ "v"
return v

# First way of failing

key, subkey = jax.random.split(key)
try:
f.simulate(key, 0.5)
except Exception as e:
print(e)

# Second way of failing

key, subkey = jax.random.split(key)
try:
f.simulate(subkey, [0.5])
except Exception as e:
print(e)

# Third way of failing

key, subkey = jax.random.split(key)
try:
f.simulate(subkey, (0.5))
except Exception as e:
print(e)

# Correct way

key, subkey = jax.random.split(key)
f.simulate(subkey, (0.5,)).get_retval()

""" 2. GenJAX relies on Tensor Flow Probability and it sometimes does unintuitive things.
"""

"""
The Bernoulli distribution uses logits instead of probabilities
"""

@gen
def g(p):
v = genjax.bernoulli(probs=p) @ "v"
return v

key, subkey = jax.random.split(key)
arg = (3.0,) # 3 is not a valid probability but a valid logit
keys = jax.random.split(subkey, 30)

# simulate 30 times

jax.vmap(g.simulate, in_axes=(0, None))(keys, arg).get_choices()

"""
Values which are stricter than $0$ are considered to be the value True.
This means that observing that the value of `"v"` is $4$ will be considered possible while intuitively `"v"` should only have support on $0$ and $1$.
"""

chm = C["v"].set(3)
g.assess(chm, (0.5,))[0] # This should be -inf.

"""
Alternatively, we can use the flip function which uses probabilities instead of logits.
"""

@gen
def h(p):
v = genjax.flip(p) @ "v"
return v

key, subkey = jax.random.split(key)
arg = (0.3,) # 0.3 is a valid probability
keys = jax.random.split(subkey, 30)

# simulate 30 times

jax.vmap(h.simulate, in_axes=(0, None))(keys, arg).get_choices()

"""
Categorical distributions also use logits instead of probabilities
"""

@gen
def i(p):
v = genjax.categorical(p) @ "v"
return v

key, subkey = jax.random.split(key)
arg = ([3.0, 1.0, 2.0],) # lists of 3 logits
keys = jax.random.split(subkey, 30)

# simulate 30 times

jax.vmap(i.simulate, in_axes=(0, None))(keys, arg).get_choices()

""" 3. JAX code can be compiled for better performance.
"""

"""
`jit` is the way to force JAX to compile the code.
It can be used as a decorator.
"""

@jit
def f*v1(p):
return jax.lax.cond(p.sum(), lambda p: p * p, lambda p: p \_ p, p)

"""
Or as a function
"""

f*v2 = jit(lambda p: jax.lax.cond(p.sum(), lambda p: p * p, lambda p: p \_ p, p))

"""
Testing the effect. Notice that the first and second have the same performance while the third is much slower (~50x on a mac m2 cpu)
"""

# Baseline

def f*v3(p):
jax.lax.cond(p.sum(), lambda p: p * p, lambda p: p \_ p, p)

arg = jax.numpy.eye(500)

# Warmup to force jit compilation

f_v1(arg)
f_v2(arg)

# Runtime comparison

%timeit f_v1(arg)
%timeit f_v2(arg)
%timeit f_v3(arg)

#

""" 4. Going from Python to JAX
"""

"""
4.1 For loops
"""

def python_loop(x):
for i in range(100):
x = 2 \* x
return x

def jax_loop(x):
jax.lax.fori_loop(0, 100, lambda i, x: 2 \* x, x)

"""
4.2 Conditional statements
"""

def python_cond(x):
if x.sum() > 0:
return x \* x
else:
return x

def jax_cond(x):
jax.lax.cond(x.sum(), lambda x: x \* x, lambda x: x, x)

"""
4.3 While loops
"""

def python_while(x):
while x.sum() > 0:
x = x \* x
return x

def jax_while(x):
jax.lax.while_loop(lambda x: x.sum() > 0, lambda x: x \* x, x)

""" 5. Is my thing compiling or is it blocked at traced time?
"""

"""
In Jax, the first time you run a function, it is traced, which produces a Jaxpr, a representation of the computation that Jax can optimize.

So in order to debug whether a function is running or not, if it passes the first check that Python let's you write it, you can check if it is running by checking if it is traced, before actually running it on data.

This is done by calling `make_jaxpr` on the function. If it returns a Jaxpr, then the function is traced and ready to be run on data.
"""

def im_fine(x):
return x \* x

jax.make_jaxpr(im_fine)(1.0)

def i_wont_be_so_fine(x):
return jax.lax.while_loop(lambda x: x > 0, lambda x: x \* x, x)

jax.make_jaxpr(i_wont_be_so_fine)(1.0)

"""
Try running the function for 8 seconds
"""

def run_process():
ctx = multiprocessing.get_context("spawn")
p = ctx.Process(target=i_wont_be_so_fine, args=(1.0,))
p.start()
time.sleep(5000)
if p.is_alive():
print("I'm still running")
p.terminate()
p.join()

result = subprocess.run(
["python", "genjax/docs/sharp-edges-notebooks/basics/script.py"],
capture_output=True,
text=True,
)

# Print the output

result.stdout

""" 6. Using random keys for generative functions
"""

"""
In GenJAX, we use explicit random keys to generate random numbers. This is done by splitting a key into multiple keys, and using them to generate random numbers.
"""

@gen
def beta_bernoulli_process(u):
p = beta(0.0, u) @ "p"
v = genjax.bernoulli(probs=p) @ "v" # sweet
return v

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, 20)
jitted = jit(beta_bernoulli_process.simulate)

jax.vmap(jitted, in_axes=(0, None))(keys, (0.5,)).get_choices()

""" 7. JAX uses 32-bit floats by default
"""

key, subkey = jax.random.split(key)
x = random.uniform(subkey, (1000,), dtype=jnp.float64)
print("surprise surprise: ", x.dtype)

"""
A common TypeError occurs when one tries using np instead of jnp, which is the JAX version of numpy, the former uses 64-bit floats by default, while the JAX version uses 32-bit floats by default.
"""

"""
This on its own gives a UserWarning
"""

jnp.array([1, 2, 3], dtype=np.float64)

"""
Using an array from `numpy` instead of `jax.numpy` will truncate the array to 32-bit floats and also give a UserWarning when used in JAX code
"""

innocent_looking_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)

@jax.jit
def innocent_looking_function(x):
return jax.lax.cond(x.sum(), lambda x: x \* x, lambda x: innocent_looking_array, x)

input = jnp.array([1.0, 2.0, 3.0])
innocent_looking_function(input)

try: # This looks fine so far but...
innocent_looking_array = np.array([1, 2, 3], dtype=np.float64)

    # This actually raises a TypeError, as one branch has type float32
    # while the other has type float64
    @jax.jit
    def innocent_looking_function(x):
        return jax.lax.cond(
            x.sum(), lambda x: x * x, lambda x: innocent_looking_array, x
        )

    input = jnp.array([1, 2, 3])
    innocent_looking_function(input)

except Exception as e:
print(e)

""" 8. Beware to OOM on the GPU which happens faster than you might think
"""

"""
Here's a simple HMM model that can be run on the GPU.
By simply changing $N$ from $300$ to $1000$, the code will typically run out of memory on the GPU as it will take ~300GB of memory
"""

N = 300
n_repeats = 100
variance = jnp.eye(N)
key, subkey = jax.random.split(key)
initial_state = jax.random.normal(subkey, (N,))

@genjax.gen
def hmm*step(x, *):
new_x = genjax.mv_normal(x, variance) @ "new_x"
return new_x, None

hmm = hmm_step.scan(n=100)

key, subkey = jax.random.split(key)
jitted = jit(hmm.repeat(n=n_repeats).simulate)
trace = jitted(subkey, (initial_state, None))
key, subkey = jax.random.split(key)
%timeit jitted(subkey, (initial_state, None))

"""
If you are running out of memory, you can try de-batching one of the computations, or using a smaller batch size. For instance, in this example, we can de-batch the `repeat` combinator, which will reduce the memory usage by a factor of $100$, at the cost of some performance.
"""

jitted = jit(hmm.simulate)

def hmm_debatched(key, initial_state):
keys = jax.random.split(key, n_repeats)
traces = {}
for i in range(n_repeats):
trace = jitted(keys[i], (initial_state, None))
traces[i] = trace
return traces

key, subkey = jax.random.split(key)

# About 4x slower on arm64 CPU and 40x on a Google Colab GPU

%timeit hmm_debatched(subkey, initial_state)

""" 9. Fast sampling can be inaccurate and yield Nan/wrong results.
"""

"""
As an example, truncating a normal distribution outside 5.5 standard deviations from its mean can yield NaNs. Many default TFP/JAX implementations that run on the GPU use fast implementations on 32bits. If one really wants that, one could use slower implementations that use 64bits and an exponential tilting Monte Carlo algorithm.
"""

genjax.truncated_normal.sample(
jax.random.key(2), 0.5382424, 0.05, 0.83921564 - 0.03, 0.83921564 + 0.03
)

minv = 0.83921564 - 0.03
maxv = 0.83921564 + 0.03
mean = 0.5382424
std = 0.05

def raw_jax_truncated(key, minv, maxv, mean, std):
low = (minv - mean) / std
high = (maxv - mean) / std
return std \* jax.random.truncated_normal(key, low, high, (), jnp.float32) + mean

raw_jax_truncated(jax.random.key(2), minv, maxv, mean, std)

# ==> Array(0.80921566, dtype=float32)

jax.jit(raw_jax_truncated)(jax.random.key(2), minv, maxv, mean, std)

# ==> Array(nan, dtype=float32)

================================================
FILE: docs/cookbook/inactive/generative_fun.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### What is a generative function and how to use it?

"""

import jax

from genjax import bernoulli, beta, gen, pretty

pretty()

"""
The following is a simple of a beta-bernoulli process. We use the `@gen` decorator to create generative functions.
"""

@gen
def beta_bernoulli_process(u):
p = beta(1.0, u) @ "p"
v = bernoulli(p) @ "v"
return v

"""
We can now call the generative function with a specified random key
"""

key = jax.random.key(0)

"""
Running the function will return a trace, which records the arguments, random choices made, and the return value
"""

key, subkey = jax.random.split(key)
tr = beta_bernoulli_process.simulate(subkey, (1.0,))

"""
We can print the trace to see what happened
"""

tr.args, tr.get_retval(), tr.get_choices()

"""
GenJAX functions can be accelerated with `jit` compilation.
"""

"""
The non-optimal way is within the `@gen` decorator.
"""

@gen
@jax.jit
def fast_beta_bernoulli_process(u):
p = beta(0.0, u) @ "p"
v = bernoulli(p) @ "v" # sweet
return v

"""
And the better way is to `jit` the final function we aim to run
"""

jitted = jax.jit(beta_bernoulli_process.simulate)

"""
We can then compare the speed of the three functions.
To fairly compare we need to run the functions once to compile them.
"""

key, subkey = jax.random.split(key)
fast_beta_bernoulli_process.simulate(subkey, (1.0,))
key, subkey = jax.random.split(key)
jitted(subkey, (1.0,))

key, subkey = jax.random.split(key)
%timeit beta_bernoulli_process.simulate(subkey, (1.0,))
key, subkey = jax.random.split(key)
%timeit fast_beta_bernoulli_process.simulate(subkey, (1.0,))
key, subkey = jax.random.split(key)
%timeit jitted(subkey, (1.0,))

================================================
FILE: docs/cookbook/inactive/differentiation/adev_demo.py
================================================

# jupyter:

# jupytext:

# formats: ipynb,py:percent

# text_representation:

# extension: .py

# format_name: percent

# format_version: '1.3'

# jupytext_version: 1.16.4

# kernelspec:

# display_name: .venv

# language: python

# name: python3

# ---

# %%

# pyright: reportUnusedExpression=false

# %% [markdown]

# ## Differentiating probabilistic programs

# ### Differentiating probabilistic programs

# %%

# Import and constants

import genstudio.plot as Plot
import jax
import jax.numpy as jnp
from genstudio.plot import js

from genjax.\_src.adev.core import Dual, expectation
from genjax.\_src.adev.primitives import flip_enum, normal_reparam

key = jax.random.key(314159)
EPOCHS = 400
default_sigma = 0.05

# %%

# Model

def noisy*jax_model(key, theta, sigma):
b = jax.random.bernoulli(key, theta)
return jax.lax.cond(
b,
lambda theta: jax.random.normal(key) * sigma \_ theta,
lambda theta: jax.random.normal(key) \* sigma + theta / 2,
theta,
)

def expected_val(theta):
return (theta - theta\*\*2) / 2

# %%

# Samples

thetas = jnp.arange(0.0, 1.0, 0.0005)

def make_samples(key, thetas, sigma):
return jax.vmap(noisy_jax_model, in_axes=(0, 0, None))(
jax.random.split(key, len(thetas)), thetas, sigma
)

key, samples_key = jax.random.split(key)
noisy_samples = make_samples(samples_key, thetas, default_sigma)

plot_options = Plot.new(
Plot.color_legend(),
{"x": {"label": "θ"}, "y": {"label": "y"}},
Plot.aspect_ratio(1),
Plot.grid(),
)

samples_color_map = Plot.color_map({"Samples": "rgba(0, 128, 128, 0.5)"})

def make_samples_plot(thetas, samples):
return (
Plot.dot({"x": thetas, "y": samples}, fill=Plot.constantly("Samples"), r=2) + samples_color_map + plot_options + Plot.clip()
)

samples_plot = make_samples_plot(thetas, noisy_samples)

samples_plot

# %%

# Adding exact expectation

thetas_sparse = jnp.linspace(0.0, 1.0, 20) # fewer points, for the plot
exact_vals = jax.vmap(expected_val)(thetas_sparse)

expected_value_plot = (
Plot.line(
{"x": thetas_sparse, "y": exact_vals},
strokeWidth=2,
stroke=Plot.constantly("Expected value"),
curve="natural",
) + Plot.color_map({"Expected value": "black"}) + plot_options,
)

samples_plot + expected_value_plot

# %%

# JAX computed exact gradients

grad_exact = jax.jit(jax.grad(expected_val))
theta_tangent_points = [0.1, 0.3, 0.45]

# Optimization on ideal curve

arg = 0.2
vals = []
arg*list = []
for * in range(EPOCHS):
grad_val = grad_exact(arg)
arg_list.append(arg)
vals.append(expected_val(arg))
arg = arg + 0.01 \* grad_val
if arg < 0:
arg = 0
break
elif arg > 1:
arg = 1

(
Plot.line({"x": list(range(EPOCHS)), "y": vals}) + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
)

# %%

color1 = "rgba(255,165,0,0.5)"
color2 = "#FB575D"

def tangent_line_plot(theta_tan):
slope = grad_exact(theta_tan)
y_intercept = expected_val(theta_tan) - slope \* theta_tan
label = f"Tangent at θ={theta_tan}"

    return Plot.line(
        [[0, y_intercept], [1, slope + y_intercept]],
        stroke=Plot.constantly(label),
    ) + Plot.color_map({
        label: js(
            f"""d3.interpolateHsl("{color1}", "{color2}")({theta_tan}/{theta_tangent_points[-1]})"""
        )
    })

(
plot_options + [tangent_line_plot(theta_tan) for theta_tan in theta_tangent_points] + expected_value_plot + Plot.domain([0, 1], [0, 0.4]) + Plot.title("Expectation curve and its Tangent Lines")
)

# %%

theta_tan = 0.3

slope = grad_exact(theta_tan)
y_intercept = expected_val(theta_tan) - slope \* theta_tan

exact_tangent_plot = Plot.line(
[[0, y_intercept], [1, slope + y_intercept]],
strokeWidth=2,
stroke=Plot.constantly("Exact tangent at θ=0.3"),
)

def slope_estimate_plot(slope_est):
y_intercept = expected_val(theta_tan) - slope_est \* theta_tan
return Plot.line(
[[0, y_intercept], [1, slope_est + y_intercept]],
strokeWidth=2,
stroke=Plot.constantly("Tangent estimate"),
)

slope_estimates = [slope + i / 20 for i in range(-4, 4)]

(
samples_plot + expected_value_plot + [slope_estimate_plot(slope_est) for slope_est in slope_estimates] + exact_tangent_plot + Plot.title("Expectation curve and Tangent Estimates at θ=0.3") + Plot.color_map({
"Tangent estimate": color1,
"Exact tangent at θ=0.3": color2,
}) + Plot.domain([0, 1], [0, 0.4])
)

# %%

jax_grad = jax.jit(jax.grad(noisy_jax_model, argnums=1))

arg = 0.2
vals = []
grads = []
for \_ in range(EPOCHS):
key, subkey = jax.random.split(key)
grad_val = jax_grad(subkey, arg, default_sigma)
arg = arg + 0.01 \* grad_val
vals.append(expected_val(arg))
grads.append(grad_val)

(
Plot.line(
{"x": list(range(EPOCHS)), "y": vals},
stroke=Plot.constantly("Attempting gradient ascent with JAX"),
) + Plot.title("Maximization of the expected value of a probabilistic function") + {"x": {"label": "Iteration"}, "y": {"label": "y"}} + Plot.domainX([0, EPOCHS]) + Plot.color_legend()
)

# %%

theta_tangents = jnp.linspace(0, 1, 20)

def plot_tangents(gradients, title):
tangents_plots = Plot.new(Plot.aspectRatio(0.5))

    for theta, slope in gradients:
        y_intercept = expected_val(theta) - slope * theta
        tangents_plots += Plot.line(
            [[0, y_intercept], [1, slope + y_intercept]],
            stroke=js(
                f"""d3.interpolateHsl("{color1}", "{color2}")({theta}/{theta_tangents[-1]})"""
            ),
            opacity=0.75,
        )
    return Plot.new(
        expected_value_plot,
        Plot.domain([0, 1], [0, 0.4]),
        tangents_plots,
        Plot.title(title),
        Plot.color_map({
            f"Tangent at θ={theta_tangents[0]}": color1,
            f"Tangent at θ={theta_tangents[-1]}": color2,
        }),
    )

gradients = []
for theta in theta_tangents:
key, subkey = jax.random.split(key)
gradients.append((theta, jax_grad(subkey, theta, default_sigma)))

plot_tangents(gradients, "Expectation curve and JAX-computed tangent estimates")

# %%

@expectation
def flip_approx_loss(theta, sigma):
b = flip_enum(theta)
return jax.lax.cond(
b,
lambda theta: normal_reparam(0.0, sigma) \* theta,
lambda theta: normal_reparam(theta / 2, sigma),
theta,
)

adev_grad = jax.jit(flip_approx_loss.jvp_estimate)

def compute*jax_vals(key, initial_theta, sigma):
current_theta = initial_theta
out = []
for * in range(EPOCHS):
key, subkey = jax.random.split(key)
gradient = jax_grad(subkey, current_theta, sigma)
out.append((current_theta, expected_val(current_theta), gradient))
current_theta = current_theta + 0.01 \* gradient
return out

def compute*adev_vals(key, initial_theta, sigma):
current_theta = initial_theta
out = []
for * in range(EPOCHS):
key, subkey = jax.random.split(key)
gradient = adev_grad(
subkey, (Dual(current_theta, 1.0), Dual(sigma, 0.0))
).tangent
out.append((current_theta, expected_val(current_theta), gradient))
current_theta = current_theta + 0.01 \* gradient
return out

# %%

def select_evenly_spaced(items, num_samples=5):
"""Select evenly spaced items from a list."""
if num_samples <= 1:
return [items[0]]

    result = [items[0]]
    step = (len(items) - 1) / (num_samples - 1)

    for i in range(1, num_samples - 1):
        index = int(i * step)
        result.append(items[index])

    result.append(items[-1])
    return result

button_classes = (
"px-3 py-1 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600"
)

def input_slider(label, value, min, max, step, on_change, default):
return [
"label.flex.flex-col.gap-2",
["div", label, ["span.font-bold.px-1", value]],
[
"input",
{
"type": "range",
"min": min,
"max": max,
"step": step,
"defaultValue": default,
"onChange": on_change,
"class": "outline-none focus:outline-none",
},
],
]

def input_checkbox(label, value, on_change):
return [
"label.flex.items-center.gap-2",
[
"input",
{
"type": "checkbox",
"checked": value,
"onChange": on_change,
"class": "h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500",
},
],
label,
["span.font-bold.px-1", value],
]

def render_plot(initial_val, initial_sigma):
SLIDER_STEP = 0.01
ANIMATION_STEP = 4
COMPARISON_HEIGHT = 200
currentKey = key

    def computeState(val, sigma):
        jax_key, adev_key, samples_key = jax.random.split(currentKey, num=3)
        return {
            "JAX_gradients": compute_jax_vals(jax_key, val, sigma),
            "ADEV_gradients": compute_adev_vals(adev_key, val, sigma),
            "samples": make_samples(samples_key, thetas, sigma),
            "val": val,
            "sigma": sigma,
            "frame": 0,
        }

    initialState = Plot.initialState(
        computeState(initial_val, initial_sigma) | {"show_expected_value": True},
        sync={"sigma", "val"},
    )

    def refresh(widget):
        nonlocal currentKey
        currentKey = jax.random.split(currentKey)[0]
        widget.state.update(computeState(widget.state.val, widget.state.sigma))

    onChange = Plot.onChange({
        "val": lambda widget, e: widget.state.update(
            computeState(float(e["value"]), widget.state.sigma)
        ),
        "sigma": lambda widget, e: widget.state.update(
            computeState(widget.state.val, float(e["value"]))
        ),
    })

    samples_plot = make_samples_plot(thetas, js("$state.samples"))

    def plot_tangents(gradients_id):
        tangents_plots = Plot.new(Plot.aspectRatio(0.5))
        color = "blue" if gradients_id == "ADEV" else "orange"

        orange_to_red_plot = Plot.dot(
            js(f"$state.{gradients_id}_gradients"),
            x="0",
            y="1",
            fill=js(
                f"""(_, i) => d3.interpolateHsl('transparent', '{color}')(i/{EPOCHS})"""
            ),
            filter=(js("(d, i) => i <= $state.frame")),
        )

        tangents_plots += orange_to_red_plot

        tangents_plots += Plot.line(
            js(f"""$state.{gradients_id}_gradients.flatMap(([theta, expected_val, slope], i) => {{
                        const y_intercept = expected_val - slope * theta
                        return [[0, y_intercept, i], [1, slope + y_intercept, i]]
                    }})
                    """),
            z="2",
            stroke=Plot.constantly(f"{gradients_id} Tangent"),
            opacity=js("(data) => data[2] === $state.frame ? 1 : 0.5"),
            strokeWidth=js("(data) => data[2] === $state.frame ? 3 : 1"),
            filter=js(f"""(data) => {{
                const index = data[2];
                if (index === $state.frame) return true;
                if (index < $state.frame) {{
                    const step = Math.floor({EPOCHS} / 10);
                    return (index % step === 0);
                }}
                return false;
            }}"""),
        )

        return Plot.new(
            js("$state.show_expected_value ? %1 : null", expected_value_plot),
            Plot.domain([0, 1], [0, 0.4]),
            tangents_plots,
            Plot.title(f"{gradients_id} Gradient Estimates"),
            Plot.color_map({"JAX Tangent": "orange", "ADEV Tangent": "blue"}),
        )

    comparison_plot = (
        Plot.line(
            js("$state.JAX_gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="2",
            stroke=Plot.constantly("Gradients from JAX"),
        )
        + Plot.line(
            js("$state.ADEV_gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="2",
            stroke=Plot.constantly("Gradients from ADEV"),
        )
        + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
        + Plot.domainX([0, EPOCHS])
        + Plot.title("Comparison of computed gradients JAX vs ADEV")
        + Plot.color_legend()
        + {"height": COMPARISON_HEIGHT}
    )

    optimization_plot = Plot.new(
        Plot.line(
            js("$state.JAX_gradients"),
            x=Plot.index(),
            y="1",
            stroke=Plot.constantly("Gradient ascent with JAX"),
            filter=js("(d, i) => i <= $state.frame"),
        )
        + Plot.line(
            js("$state.ADEV_gradients"),
            x=Plot.index(),
            y="1",
            stroke=Plot.constantly("Gradient ascent with ADEV"),
            filter=js("(d, i) => i <= $state.frame"),
        )
        + {
            "x": {"label": "Iteration"},
            "y": {"label": "Expected Value"},
        }
        + Plot.domainX([0, EPOCHS])
        + Plot.title("Maximization of the expected value of a probabilistic function")
        + Plot.color_legend()
        + {"height": COMPARISON_HEIGHT}
    )

    jax_tangents_plot = samples_plot + plot_tangents("JAX")
    adev_tangents_plot = samples_plot + plot_tangents("ADEV")

    frame_slider = Plot.Slider(
        key="frame",
        init=0,
        range=[0, EPOCHS],
        step=ANIMATION_STEP,
        fps=30,
        label="Iteration:",
    )

    controls = Plot.html([
        "div.flex.mb-3.gap-4.bg-gray-200.rounded-md.p-3",
        [
            "div.flex.flex-col.gap-1.w-32",
            input_slider(
                label="Initial Value:",
                value=js("$state.val"),
                min=0,
                max=1,
                step=SLIDER_STEP,
                on_change=js("(e) => $state.val = parseFloat(e.target.value)"),
                default=initial_val,
            ),
            input_slider(
                label="Sigma:",
                value=js("$state.sigma"),
                min=0,
                max=0.2,
                step=0.01,
                on_change=js("(e) => $state.sigma = parseFloat(e.target.value)"),
                default=initial_sigma,
            ),
        ],
        [
            "div.flex.flex-col.gap-2.flex-auto",
            input_checkbox(
                label="Show expected value",
                value=js("$state.show_expected_value"),
                on_change=js("(e) => $state.show_expected_value = e.target.checked"),
            ),
            Plot.katex(r"""

y(\theta) = \mathbb{E}_{x\sim P(\theta)}[x] = \int_{\mathbb{R}}\left[\theta^2\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x}{\sigma}\right)^2} + (1-\theta)\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x-0.5\theta}{\sigma}\right)^2}\right]dx =\frac{\theta-\theta^2}{2}
"""),
[
"button.w-32",
{
"onClick": lambda widget, e: refresh(widget),
"class": button_classes,
},
"Refresh",
],
],
])

    jax_code = """def noisy_jax_model(key, theta, sigma):
    b = jax.random.bernoulli(key, theta)
    return jax.lax.cond(
        b,
        lambda theta: jax.random.normal(key) * sigma * theta,
        lambda theta: jax.random.normal(key) * sigma + theta / 2,
        theta,
    )"""

    adev_code = """@expectation

def flip_approx_loss(theta, sigma):
b = flip_enum(theta)
return jax.lax.cond(
b,
lambda theta: normal_reparam(0.0, sigma) \* theta,
lambda theta: normal_reparam(theta / 2, sigma),
theta,
)"""

    GRID = "div.grid.grid-cols-2.gap-4"
    PRE = "pre.whitespace-pre-wrap.text-2xs.p-3.rounded-md.bg-gray-100.flex-1"

    return (
        initialState
        | onChange
        | controls
        | [
            GRID,
            jax_tangents_plot,
            adev_tangents_plot,
            [PRE, jax_code],
            [PRE, adev_code],
            comparison_plot,
            optimization_plot,
        ]
        | frame_slider
    )

render_plot(0.2, 0.05)

================================================
FILE: docs/cookbook/inactive/differentiation/adev_example.py
================================================

# jupyter:

# jupytext:

# formats: ipynb,py:percent

# text_representation:

# extension: .py

# format_name: percent

# format_version: '1.3'

# jupytext_version: 1.16.4

# kernelspec:

# display_name: .venv

# language: python

# name: python3

# ---

# %% [markdown]

# ## Differentiating probabilistic programs

# ### Differentiating probabilistic programs

# %%

# pyright: reportUnusedExpression=false

# %%

# Import and constants

import genstudio.plot as Plot
import jax
import jax.numpy as jnp
from genstudio.plot import js

from genjax.\_src.adev.core import Dual, expectation
from genjax.\_src.adev.primitives import flip_enum, normal_reparam

key = jax.random.key(314159)
EPOCHS = 400
default_sigma = 0.05

# %% [markdown]

# We are often interested in the average returned value of a probabilistic program. For instance, it could be that

# a run of the program represents a run of a simulation of some form, and we would like to maximize the average reward across many simulations (or equivalently minimize a loss).

# %%

# Model

def noisy*jax_model(key, theta, sigma):
b = jax.random.bernoulli(key, theta)
return jax.lax.cond(
b,
lambda theta: jax.random.normal(key) * sigma \_ theta,
lambda theta: jax.random.normal(key) \* sigma + theta / 2,
theta,
)

def expected_val(theta):
return (theta - theta\*\*2) / 2

# %% [markdown]

# We can see that the simulation can have two "modes" that split further appart over time.

# %%

# Samples

thetas = jnp.arange(0.0, 1.0, 0.0005)

def make_samples(key, thetas, sigma):
return jax.vmap(noisy_jax_model, in_axes=(0, 0, None))(
jax.random.split(key, len(thetas)), thetas, sigma
)

key, samples_key = jax.random.split(key)
noisy_samples = make_samples(samples_key, thetas, default_sigma)

plot_options = Plot.new(
Plot.color_legend(),
{"x": {"label": "θ"}, "y": {"label": "y"}},
Plot.aspect_ratio(1),
Plot.grid(),
Plot.clip(),
)

samples_color_map = Plot.color_map({"Samples": "rgba(0, 128, 128, 0.5)"})

def make_samples_plot(thetas, samples):
return (
Plot.dot({"x": thetas, "y": samples}, fill=Plot.constantly("Samples"), r=2) + samples_color_map + plot_options + Plot.clip()
)

samples_plot = make_samples_plot(thetas, noisy_samples)

samples_plot

# %% [markdown]

# We can also easily imagine a more noisy version of the same idea.

# %%

def more*noisy_jax_model(key, theta, sigma):
b = jax.random.bernoulli(key, theta)
return jax.lax.cond(
b,
lambda *: jax.random.normal(key) _ sigma _ theta\*_2 / 3,
lambda \_: (jax.random.normal(key) _ sigma + theta) / 2,
None,
)

more_thetas = jnp.arange(0.0, 1.0, 0.0005)
key, \*keys = jax.random.split(key, len(more_thetas) + 1)

noisy_sample_plot = (
Plot.dot(
{
"x": more_thetas,
"y": jax.vmap(more_noisy_jax_model, in_axes=(0, 0, None))(
jnp.array(keys), more_thetas, default_sigma
),
},
fill=Plot.constantly("Samples"),
r=2,
) + samples_color_map
)

noisy_sample_plot + plot_options

# %% [markdown]

# As we can see better on the noisy version, the samples divide into two groups. One tends to go up as theta increases while the other stays relatively stable around 0 with a higher variance. For simplicity of the analysis, in the rest of this notebook we will stick to the simpler first example.

# %% [markdown]

#

# In that simple case, we can compute the exact average value of the random process as a function of $\theta$. We have probability $\theta$ to return $0$ and probablity $1-\theta$ to return $\frac{\theta}{2}$. So overall the expected value is

# $$\theta*0 + (1-\theta)*\frac{\theta}{2} = \frac{\theta-\theta^2}{2}$$

# %% [markdown]

# We can code this and plot the result for comparison.

# %%

# Adding exact expectation

thetas_sparse = jnp.linspace(0.0, 1.0, 20) # fewer points, for the plot
exact_vals = jax.vmap(expected_val)(thetas_sparse)

expected_value_plot = (
Plot.line(
{"x": thetas_sparse, "y": exact_vals},
strokeWidth=2,
stroke=Plot.constantly("Expected value"),
curve="natural",
) + Plot.color_map({"Expected value": "black"}) + plot_options,
)

samples_plot + expected_value_plot

# %% [markdown]

# We can see that the curve in yellow is a perfectly reasonable differentiable function. We can use JAX to compute its derivative (more generally its gradient) at various points.

# %%

grad_exact = jax.jit(jax.grad(expected_val))
theta_tangent_points = [0.1, 0.3, 0.45]

color1 = "rgba(255,165,0,0.5)"
color2 = "#FB575D"

def tangent_line_plot(theta_tan):
slope = grad_exact(theta_tan)
y_intercept = expected_val(theta_tan) - slope \* theta_tan
label = f"Tangent at θ={theta_tan}"

    return Plot.line(
        [[0, y_intercept], [1, slope + y_intercept]],
        stroke=Plot.constantly(label),
    ) + Plot.color_map({
        label: Plot.js(
            f"""d3.interpolateHsl("{color1}", "{color2}")({theta_tan}/{theta_tangent_points[-1]})"""
        )
    })

(
plot_options + [tangent_line_plot(theta_tan) for theta_tan in theta_tangent_points] + expected_value_plot + Plot.domain([0, 1], [0, 0.4]) + Plot.title("Expectation curve and its Tangent Lines")
)

# %% [markdown]

# A popular technique from optimization is to use iterative methods such as (stochastic) gradient ascent.

# Starting from any location, say 0.2, we can use JAX to find the maximum of the function.

# %%

arg = 0.2
vals = []
arg*list = []
for * in range(EPOCHS):
grad_val = grad_exact(arg)
arg_list.append(arg)
vals.append(expected_val(arg))
arg = arg + 0.01 \* grad_val
if arg < 0:
arg = 0
break
elif arg > 1:
arg = 1

# %% [markdown]

# We can plot the evolution of the value of the function over the iterations of the algorithms.

# %%

(
Plot.line({"x": list(range(EPOCHS)), "y": vals}) + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
)

# %% [markdown]

# We can also directly visualize the points on the curve.

# %%

(
expected*value_plot + Plot.dot(
{"x": arg_list, "y": vals},
fill=Plot.js(
f"""(*, i) => d3.interpolateHsl('{color1}', '{color2}')(i/{len(arg_list)})"""
),
) + Plot.subtitle("Gradient descent from start to end") + Plot.aspect_ratio(0.25) + {"width": 600} + Plot.color_map({"start": color1, "end": color2})
)

# %% [markdown]

# We have this in this example that we can compute the average value exactly. But will not be the case in general. One popular technique to approximate an average value is to use Monte Carlo Integration: we sample a bunch from the program and take the average value.

#

# As we use more and more samples we will converge to the correct result by the Central limit theorem.

# %%

number_of_samples = sorted([1, 3, 5, 10, 20, 50, 100, 200, 500, 1000] \* 7)
means = []
for n in number_of_samples:
key, subkey = jax.random.split(key)
keys = jax.random.split(key, n)
samples = jax.vmap(noisy_jax_model, in_axes=(0, None, None))(
keys, 0.3, default_sigma
)
mean = jnp.mean(samples)
means.append(mean)

(
Plot.dot(
{"x": number*of_samples, "y": means},
fill=Plot.js(
f"""(*, i) => d3.interpolateHsl('{color1}', '{color2}')(i/{len(number_of_samples)})"""
),
) + Plot.ruleY(
[expected_val(0.3)],
opacity=0.2,
strokeWidth=2,
stroke=Plot.constantly("True value"),
) + Plot.color_map({"Mean estimate": color1, "True value": "green"}) + Plot.color_legend() + {"x": {"label": "Number of samples", "type": "log"}, "y": {"label": "y"}}
)

# %% [markdown]

# As we just discussed, most of the time we will not be able to compute the average value and then compute the gradient using JAX. One thing we may want to try is to use JAX on the probabilistic program to get a gradient estimate, and hope that by using more and more samples this will converge to the correct gradient that we could use in optimization. Let's try it in JAX.

# %%

theta_tan = 0.3

slope = grad_exact(theta_tan)
y_intercept = expected_val(theta_tan) - slope \* theta_tan

exact_tangent_plot = Plot.line(
[[0, y_intercept], [1, slope + y_intercept]],
strokeWidth=2,
stroke=Plot.constantly("Exact tangent at θ=0.3"),
)

def slope_estimate_plot(slope_est):
y_intercept = expected_val(theta_tan) - slope_est \* theta_tan
return Plot.line(
[[0, y_intercept], [1, slope_est + y_intercept]],
strokeWidth=2,
stroke=Plot.constantly("Tangent estimate"),
)

slope_estimates = [slope + i / 20 for i in range(-4, 4)]

(
samples_plot + expected_value_plot + [slope_estimate_plot(slope_est) for slope_est in slope_estimates] + exact_tangent_plot + Plot.title("Expectation curve and Tangent Estimates at θ=0.3") + Plot.color_map({
"Tangent estimate": color1,
"Exact tangent at θ=0.3": color2,
}) + Plot.domain([0, 1], [0, 0.4])
)

# %%

jax_grad = jax.jit(jax.grad(noisy_jax_model, argnums=1))

arg = 0.2
vals = []
for \_ in range(EPOCHS):
key, subkey = jax.random.split(key)
grad_val = jax_grad(subkey, arg, default_sigma)
arg = arg + 0.01 \* grad_val
vals.append(expected_val(arg))

# %% [markdown]

# JAX seems happy to compute something and we can use the iterative technique from before, but let's see if we managed to minimize the function.

# %%

(
Plot.line(
{"x": list(range(EPOCHS)), "y": vals},
stroke=Plot.constantly("Attempting gradient ascent with JAX"),
) + {"x": {"label": "Iteration"}, "y": {"label": "y"}} + Plot.domainX([0, EPOCHS]) + Plot.title("Maximization of the expected value of a probabilistic function") + Plot.color_legend()
)

# %% [markdown]

# Woops! We seemed to start ok but then for some reason the curve goes back down and we end up minimizing the function instead of maximizing it!

#

# The reason is that we failed to account from the change of contribution of the coin flip from `bernoulli` in the differentiation process, and we will come back to this in more details in follow up notebooks.

#

# For now, let's just get a sense of what the gradient estimates computed by JAX look like.

# %%

theta_tangents = jnp.linspace(0, 1, 20)

def plot_tangents(gradients, title):
tangents_plots = Plot.new(Plot.aspectRatio(0.5))

    for theta, slope in gradients:
        y_intercept = expected_val(theta) - slope * theta
        tangents_plots += Plot.line(
            [[0, y_intercept], [1, slope + y_intercept]],
            stroke=Plot.js(
                f"""d3.interpolateHsl("{color1}", "{color2}")({theta}/{theta_tangents[-1]})"""
            ),
            opacity=0.75,
        )
    return Plot.new(
        expected_value_plot,
        Plot.domain([0, 1], [0, 0.4]),
        tangents_plots,
        Plot.title(title),
        Plot.color_map({
            f"Tangent at θ={theta_tangents[0]}": color1,
            f"Tangent at θ={theta_tangents[-1]}": color2,
        }),
    )

gradients = []
for theta in theta_tangents:
key, subkey = jax.random.split(key)
gradients.append((theta, jax_grad(subkey, theta, default_sigma)))

plot_tangents(gradients, "Expectation curve and JAX-computed tangent estimates")

# %% [markdown]

# Ouch. They do not look even remotely close to valid gradient estimates.

# %% [markdown]

# ADEV is a new algorithm that computes correct gradient estimates of expectations of probabilistic programs. It accounts for the change to the expectation coming from a change to the randomness present in the expectation.

#

# GenJAX implements ADEV. Slightly rewriting the example from above using GenJAX, we can see how different the behaviour of the optimization process with the corrected gradient estimates is.

# %%

@expectation
def flip_approx_loss(theta, sigma):
b = flip_enum(theta)
return jax.lax.cond(
b,
lambda theta: normal_reparam(0.0, sigma) \* theta,
lambda theta: normal_reparam(theta / 2, sigma),
theta,
)

adev_grad = jax.jit(flip_approx_loss.jvp_estimate)

def compute*jax_vals(key, initial_theta, sigma):
current_theta = initial_theta
out = []
for * in range(EPOCHS):
key, subkey = jax.random.split(key)
gradient = jax_grad(subkey, current_theta, sigma)
out.append((current_theta, expected_val(current_theta), gradient))
current_theta = current_theta + 0.01 \* gradient
return out

def compute*adev_vals(key, initial_theta, sigma):
current_theta = initial_theta
out = []
for * in range(EPOCHS):
key, subkey = jax.random.split(key)
gradient = adev_grad(
subkey, (Dual(current_theta, 1.0), Dual(sigma, 0.0))
).tangent
out.append((current_theta, expected_val(current_theta), gradient))
current_theta = current_theta + 0.01 \* gradient
return out

# %%

def select_evenly_spaced(items, num_samples=5):
"""Select evenly spaced items from a list."""
if num_samples <= 1:
return [items[0]]

    result = [items[0]]
    step = (len(items) - 1) / (num_samples - 1)

    for i in range(1, num_samples - 1):
        index = int(i * step)
        result.append(items[index])

    result.append(items[-1])
    return result

INITIAL_VAL = 0.2
SLIDER_STEP = 0.01
ANIMATION_STEP = 4

button_classes = (
"px-3 py-1 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600"
)

def input_slider(label, value, min, max, step, on_change, default):
return [
"label.flex.flex-col.gap-2",
["div", label, ["span.font-bold.px-1", value]],
[
"input",
{
"type": "range",
"min": min,
"max": max,
"step": step,
"defaultValue": default,
"onChange": on_change,
"class": "outline-none focus:outline-none",
},
],
]

def input_checkbox(label, value, on_change):
return [
"label.flex.items-center.gap-2",
[
"input",
{
"type": "checkbox",
"checked": value,
"onChange": on_change,
"class": "h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500",
},
],
label,
["span.font-bold.px-1", value],
]

def render_plot(initial_val, initial_sigma):
SLIDER_STEP = 0.01
ANIMATION_STEP = 4
COMPARISON_HEIGHT = 200
currentKey = key

    def computeState(val, sigma):
        jax_key, adev_key, samples_key = jax.random.split(currentKey, num=3)
        return {
            "JAX_gradients": compute_jax_vals(jax_key, val, sigma),
            "ADEV_gradients": compute_adev_vals(adev_key, val, sigma),
            "samples": make_samples(samples_key, thetas, sigma),
            "val": val,
            "sigma": sigma,
            "frame": 0,
        }

    initialState = Plot.initialState(
        computeState(initial_val, initial_sigma) | {"show_expected_value": True},
        sync={"sigma", "val"},
    )

    def refresh(widget):
        nonlocal currentKey
        currentKey = jax.random.split(currentKey)[0]
        widget.state.update(computeState(widget.state.val, widget.state.sigma))

    onChange = Plot.onChange({
        "val": lambda widget, e: widget.state.update(
            computeState(float(e["value"]), widget.state.sigma)
        ),
        "sigma": lambda widget, e: widget.state.update(
            computeState(widget.state.val, float(e["value"]))
        ),
    })

    samples_plot = make_samples_plot(thetas, js("$state.samples"))

    def plot_tangents(gradients_id):
        tangents_plots = Plot.new(Plot.aspectRatio(0.5))
        color = "blue" if gradients_id == "ADEV" else "orange"

        orange_to_red_plot = Plot.dot(
            js(f"$state.{gradients_id}_gradients"),
            x="0",
            y="1",
            fill=js(
                f"""(_, i) => d3.interpolateHsl('transparent', '{color}')(i/{EPOCHS})"""
            ),
            filter=(js("(d, i) => i <= $state.frame")),
        )

        tangents_plots += orange_to_red_plot

        tangents_plots += Plot.line(
            js(f"""$state.{gradients_id}_gradients.flatMap(([theta, expected_val, slope], i) => {{
                        const y_intercept = expected_val - slope * theta
                        return [[0, y_intercept, i], [1, slope + y_intercept, i]]
                    }})
                    """),
            z="2",
            stroke=Plot.constantly(f"{gradients_id} Tangent"),
            opacity=js("(data) => data[2] === $state.frame ? 1 : 0.5"),
            strokeWidth=js("(data) => data[2] === $state.frame ? 3 : 1"),
            filter=js(f"""(data) => {{
                const index = data[2];
                if (index === $state.frame) return true;
                if (index < $state.frame) {{
                    const step = Math.floor({EPOCHS} / 10);
                    return (index % step === 0);
                }}
                return false;
            }}"""),
        )

        return Plot.new(
            js("$state.show_expected_value ? %1 : null", expected_value_plot),
            Plot.domain([0, 1], [0, 0.4]),
            tangents_plots,
            Plot.title(f"{gradients_id} Gradient Estimates"),
            Plot.color_map({"JAX Tangent": "orange", "ADEV Tangent": "blue"}),
        )

    comparison_plot = (
        Plot.line(
            js("$state.JAX_gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="2",
            stroke=Plot.constantly("Gradients from JAX"),
        )
        + Plot.line(
            js("$state.ADEV_gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="2",
            stroke=Plot.constantly("Gradients from ADEV"),
        )
        + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
        + Plot.domainX([0, EPOCHS])
        + Plot.title("Comparison of computed gradients JAX vs ADEV")
        + Plot.color_legend()
        + {"height": COMPARISON_HEIGHT}
    )

    optimization_plot = Plot.new(
        Plot.line(
            js("$state.JAX_gradients"),
            x=Plot.index(),
            y="1",
            stroke=Plot.constantly("Gradient ascent with JAX"),
            filter=js("(d, i) => i <= $state.frame"),
        )
        + Plot.line(
            js("$state.ADEV_gradients"),
            x=Plot.index(),
            y="1",
            stroke=Plot.constantly("Gradient ascent with ADEV"),
            filter=js("(d, i) => i <= $state.frame"),
        )
        + {
            "x": {"label": "Iteration"},
            "y": {"label": "Expected Value"},
        }
        + Plot.domainX([0, EPOCHS])
        + Plot.title("Maximization of the expected value of a probabilistic function")
        + Plot.color_legend()
        + {"height": COMPARISON_HEIGHT}
    )

    jax_tangents_plot = samples_plot + plot_tangents("JAX")
    adev_tangents_plot = samples_plot + plot_tangents("ADEV")

    frame_slider = Plot.Slider(
        key="frame",
        init=0,
        range=[0, EPOCHS],
        step=ANIMATION_STEP,
        fps=30,
        label="Iteration:",
    )

    controls = Plot.html([
        "div.flex.mb-3.gap-4.bg-gray-200.rounded-md.p-3",
        [
            "div.flex.flex-col.gap-1.w-32",
            input_slider(
                label="Initial Value:",
                value=js("$state.val"),
                min=0,
                max=1,
                step=SLIDER_STEP,
                on_change=js("(e) => $state.val = parseFloat(e.target.value)"),
                default=initial_val,
            ),
            input_slider(
                label="Sigma:",
                value=js("$state.sigma"),
                min=0,
                max=0.2,
                step=0.01,
                on_change=js("(e) => $state.sigma = parseFloat(e.target.value)"),
                default=initial_sigma,
            ),
        ],
        [
            "div.flex.flex-col.gap-2.flex-auto",
            input_checkbox(
                label="Show expected value",
                value=js("$state.show_expected_value"),
                on_change=js("(e) => $state.show_expected_value = e.target.checked"),
            ),
            Plot.katex(r"""

y(\theta) = \mathbb{E}_{x\sim P(\theta)}[x] = \int_{\mathbb{R}}\left[\theta^2\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x}{\sigma}\right)^2} + (1-\theta)\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x-0.5\theta}{\sigma}\right)^2}\right]dx =\frac{\theta-\theta^2}{2}
"""),
[
"button.w-32",
{
"onClick": lambda widget, e: refresh(widget),
"class": button_classes,
},
"Refresh",
],
],
])

    jax_code = """def noisy_jax_model(key, theta, sigma):
    b = jax.random.bernoulli(key, theta)
    return jax.lax.cond(
        b,
        lambda theta: jax.random.normal(key) * sigma * theta,
        lambda theta: jax.random.normal(key) * sigma + theta / 2,
        theta,
    )"""

    adev_code = """@expectation

def flip_approx_loss(theta, sigma):
b = flip_enum(theta)
return jax.lax.cond(
b,
lambda theta: normal_reparam(0.0, sigma) \* theta,
lambda theta: normal_reparam(theta / 2, sigma),
theta,
)"""

    GRID = "div.grid.grid-cols-2.gap-4"
    PRE = "pre.whitespace-pre-wrap.text-2xs.p-3.rounded-md.bg-gray-100.flex-1"

    return (
        initialState
        | onChange
        | controls
        | [
            GRID,
            jax_tangents_plot,
            adev_tangents_plot,
            [PRE, jax_code],
            [PRE, adev_code],
            comparison_plot,
            optimization_plot,
        ]
        | frame_slider
    )

render_plot(0.2, 0.05)

# %% [markdown]

# In the above example, by using `jvp_estimate` we used a forward-mode version of ADEV. GenJAX also supports a reverse-mode version which is also fully compatible with JAX and can be jitted.

# %%

@expectation
def flip*exact_loss(theta):
b = flip_enum(theta)
return jax.lax.cond(
b,
lambda *: 0.0,
lambda \_: -theta / 2.0,
theta,
)

rev_adev_grad = jax.jit(flip_exact_loss.grad_estimate)

arg = 0.2
rev*adev_vals = []
for * in range(EPOCHS):
key, subkey = jax.random.split(key)
(grad_val,) = rev_adev_grad(subkey, (arg,))
arg = arg - 0.01 \* grad_val
rev_adev_vals.append(expected_val(arg))

(
Plot.line(
{"x": list(range(EPOCHS)), "y": rev_adev_vals},
stroke=Plot.constantly("Reverse mode ADEV"),
) + {"x": {"label": "Iteration"}, "y": {"label": "y"}} + Plot.domainX([0, EPOCHS]) + Plot.title("Maximization of the expected value of a probabilistic function") + Plot.color_legend()
)

================================================
FILE: docs/cookbook/inactive/expressivity/conditionals.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### How do I use conditionals in (Gen)JAX? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/expressivity/conditionals.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

import jax
import jax.numpy as jnp

from genjax import bernoulli, gen, normal, or_else, pretty, switch

key = jax.random.key(0)
pretty()

"""
In pure Python, we can use usual conditionals
"""

def simple_cond_python(p):
if p > 0:
return 2 \* p
else:
return -p

simple_cond_python(0.3), simple_cond_python(-0.4)

"""
In pure JAX, we write conditionals with `jax.lax.cond` as follows
"""

def branch_1(p):
return 2 \* p

def branch_2(p):
return -p

def simple_cond_jax(p):
pred = p > 0
arg_of_cond = p
cond_res = jax.lax.cond(pred, branch_1, branch_2, arg_of_cond)
return cond_res

simple_cond_jax(0.3), simple_cond_jax(-0.4)

"""
Compiled JAX code is usually quite faster than Python code
"""

def python_loop(x):
for i in range(40000):
if x < 100.0:
x = 2 \* x
else:
x = x - 97.0
return x

@jax.jit
def jax*loop(x):
return jax.lax.fori_loop(
0,
40000,
lambda *, x: jax.lax.cond(x < 100.0, lambda x: 2 \* x, lambda x: x - 97.0, x),
x,
)

%timeit python_loop(1.0)

# Get the JIT time out of the way

jax_loop(1.0)
%timeit jax_loop(1.0)

"""
One restriction is that both branches should have the same pytree structure
"""

def failing_simple_cond_1(p):
pred = p > 0

    def branch_1(p):
        return (p, p)

    def branch_2(p):
        return -p

    arg_of_cond = p
    cond_res = jax.lax.cond(pred, branch_1, branch_2, arg_of_cond)
    return cond_res

try:
print(failing_simple_cond_1(0.3))
except TypeError as e:
print(e)

"""
The other one is that the type of the output of the branches should be the same
"""

def failing_simple_cond_2(p):
pred = p > 0

    def branch_1(p):
        return 2 * p

    def branch_2(p):
        return 7

    arg_of_cond = p
    cond_res = jax.lax.cond(pred, branch_1, branch_2, arg_of_cond)
    return cond_res

try:
print(failing_simple_cond_2(0.3))
except TypeError as e:
print(e)

"""
In GenJAX, the syntax is a bit different still.
Similarly to JAX having a custom primitive `jax.lax.cond` that creates a conditional by "composing" two functions seen as branches, GenJAX has a custom combinator that "composes" two generative functions, called `genjax.or_else`.
"""

"""
We can first define the two branches as generative functions
"""

@gen
def branch_1(p):
v = bernoulli(p) @ "v1"
return v

@gen
def branch_2(p):
v = bernoulli(-p) @ "v2"
return v

"""
Then we use the combinator to compose them
"""

@gen
def cond_model(p):
pred = p > 0
arg_1 = (p,)
arg_2 = (p,)
v = or_else(branch_1, branch_2)(pred, arg_1, arg_2) @ "cond"
return v

jitted = jax.jit(cond_model.simulate)
key, subkey = jax.random.split(key)
tr = jitted(subkey, (0.0,))
tr.get_choices()

"""
Alternatively, we can write `or_else` as follows:
"""

@gen
def cond_model_v2(p):
pred = p > 0
arg_1 = (p,)
arg_2 = (p,)
v = branch_1.or_else(branch_2)(pred, arg_1, arg_2) @ "cond"
return v

key, subkey = jax.random.split(key)
cond_model_v2.simulate(subkey, (0.0,))

"""
Note that it may be possible to write the following down, but this will not give you what you want in general!
"""

# TODO: find a way to make it fail to better show the point.

@gen
def simple_cond_genjax(p):
def branch_1(p):
return bernoulli(p) @ "v1"

    def branch_2(p):
        return bernoulli(-p) @ "v2"

    cond = jax.lax.cond(p > 0, branch_1, branch_2, p)
    return cond

key, subkey = jax.random.split(key)
tr1 = simple_cond_genjax.simulate(subkey, (0.3,))
key, subkey = jax.random.split(key)
tr2 = simple_cond_genjax.simulate(subkey, (-0.4,))
tr1.get_retval(), tr2.get_retval()

"""
Alternatively, if we have more than two branches, in JAX we can use the `jax.lax.switch` function.
"""

def simple_switch_jax(p):
index = jnp.floor(jnp.abs(p)).astype(jnp.int32) % 3
branches = [lambda p: 2 * p, lambda p: -p, lambda p: p]
switch_res = jax.lax.switch(index, branches, p)
return switch_res

simple_switch_jax(0.3), simple_switch_jax(1.1), simple_switch_jax(2.3)

"""
Likewise, in GenJAX we can use the `switch` combinator if we have more than two branches.
We can first define three branches as generative functions
"""

@gen
def branch_1(p):
v = normal(p, 1.0) @ "v1"
return v

@gen
def branch_2(p):
v = normal(-p, 1.0) @ "v2"
return v

@gen
def branch_3(p):
v = normal(p \* p, 1.0) @ "v3"
return v

"""
Then we use the combinator to compose them.
"""

@gen
def switch_model(p):
index = jnp.floor(jnp.abs(p)).astype(jnp.int32) % 3
v = switch(branch_1, branch_2, branch_3)(index, (p,), (p,), (p,)) @ "s"
return v

key, subkey = jax.random.split(key)
jitted = jax.jit(switch_model.simulate)
tr1 = jitted(subkey, (0.0,))
key, subkey = jax.random.split(key)
tr2 = jitted(subkey, (1.1,))
key, subkey = jax.random.split(key)
tr3 = jitted(subkey, (2.2,))
(
tr1.get_choices()["s", "v1"],
tr2.get_choices()["s", "v2"],
tr3.get_choices()["s", "v3"],
)

"""
We can rewrite the above a bit more elegantly using the \*args syntax
"""

@gen
def switch_model_v2(p):
index = jnp.floor(jnp.abs(p)).astype(jnp.int32) % 3
branches = [branch_1, branch_2, branch_3]
args = [(p,), (p,), (p,)]
v = switch(*branches)(index, *args) @ "switch"
return v

jitted = switch_model_v2.simulate
key, subkey = jax.random.split(key)
tr = jitted(subkey, (0.0,))
tr.get_choices()["switch", "v1"]

================================================
FILE: docs/cookbook/inactive/expressivity/custom_distribution.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### How do I create a custom distribution in GenJAX? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/expressivity/custom_distribution.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from genjax import ChoiceMapBuilder as C
from genjax import Distribution, ExactDensity, Pytree, Weight, gen, normal, pretty
from genjax.typing import PRNGKey

tfd = tfp.distributions
key = jax.random.key(0)
pretty()

"""
In GenJAX, there are two simple ways to extend the language by adding custom distributions which can be seamlessly used by the system.
"""

"""
The first way is to add a distribution for which we can compute its density exactly.
In this case the API follows what one expects: one method to sample and one method to compute logpdf.
"""

@Pytree.dataclass
class NormalInverseGamma(ExactDensity):
def sample(self, key: PRNGKey, μ, σ, α, β):
key, subkey = jax.random.split(key)
x = tfd.Normal(μ, σ).sample(seed=key)
y = tfd.InverseGamma(α, β).sample(seed=subkey)
return (x, y)

    def logpdf(self, v, μ, σ, α, β):
        x, y = v
        a = tfd.Normal(μ, σ).log_prob(x)
        b = tfd.InverseGamma(α, β).log_prob(y)
        return a + b

"""
Testing
"""

# Create a particular instance of the distribution

norm_inv_gamma = NormalInverseGamma()

@gen
def model():
(x, y) = norm_inv_gamma(0.0, 1.0, 1.0, 1.0) @ "xy"
z = normal(x, y) @ "z"
return z

# Sampling from the model

key, subkey = jax.random.split(key)
jax.jit(model.simulate)(key, ())

# Computing density of joint

jax.jit(model.assess)(C["xy"].set((2.0, 2.0)) | C["z"].set(2.0), ())

"""
The second way is to create a distribution via the `Distribution` class.
Here, the `logpdf` method is replace by the more general `estimate_logpdf` method. The distribution is asked to return an unbiased density estimate of its logpdf at the provided value.
The `sample` method is replaced by `random_weighted`. It returns a sample from the distribution as well as an unbiased estimate of the reciprocal density, i.e. an estimate of $\frac{1}{p(x)}$.
Here we'll create a simple mixture of Gaussians.
"""

@Pytree.dataclass
class GaussianMixture(Distribution): # It can have static args
bias: float = Pytree.static(default=0.0)

    # For distributions that can compute their densities exactly, `random_weighted` should return a sample x and the reciprocal density 1/p(x).
    def random_weighted(self, key: PRNGKey, probs, means, vars) -> tuple[Weight, any]:
        # making sure that the inputs are jnp arrays for jax compatibility
        probs = jnp.asarray(probs)
        means = jnp.asarray(means)
        vars = jnp.asarray(vars)

        # sampling from the categorical distribution and then sampling from the normal distribution
        cat = tfd.Categorical(probs=probs)
        cat_index = jnp.asarray(cat.sample(seed=key))
        normal = tfd.Normal(
            loc=means[cat_index] + jnp.asarray(self.bias), scale=vars[cat_index]
        )
        key, subkey = jax.random.split(key)
        normal_sample = normal.sample(seed=subkey)

        # calculating the reciprocal density
        zipped = jnp.stack([probs, means, vars], axis=1)
        weight_recip = -jnp.log(
            sum(
                jax.vmap(
                    lambda z: tfd.Normal(
                        loc=z[1] + jnp.asarray(self.bias), scale=z[2]
                    ).prob(normal_sample)
                    * tfd.Categorical(probs=probs).prob(z[0])
                )(zipped)
            )
        )

        return weight_recip, normal_sample

    # For distributions that can compute their densities exactly, `estimate_logpdf` should return the log density at x.
    def estimate_logpdf(self, key: jax.random.key, x, probs, means, vars) -> Weight:
        zipped = jnp.stack([probs, means, vars], axis=1)
        return jnp.log(
            sum(
                jax.vmap(
                    lambda z: tfd.Normal(
                        loc=z[1] + jnp.asarray(self.bias), scale=z[2]
                    ).prob(x)
                    * tfd.Categorical(probs=probs).prob(z[0])
                )(zipped)
            )
        )

"""
Testing:
"""

gauss_mix = GaussianMixture(0.0)

@gen
def model(probs):
mix1 = gauss_mix(probs, jnp.array([0.0, 1.0]), jnp.array([1.0, 1.0])) @ "mix1"
mix2 = gauss_mix(probs, jnp.array([0.0, 1.0]), jnp.array([1.0, 1.0])) @ "mix2"
return mix1, mix2

probs = jnp.array([0.5, 0.5])

# Sampling from the model

key, subkey = jax.random.split(key)
jax.jit(model.simulate)(subkey, (probs,))

# Computing density of joint

key, subkey = jax.random.split(key)
jax.jit(model.importance)(subkey, C["mix1"].set(3.0) | C["mix2"].set(4.0), (probs,))

================================================
FILE: docs/cookbook/inactive/expressivity/iterating_computation.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### I have a generative function with a single variable but 2000 observations, or I just want to use/apply it repeatedly. What do I do? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/expressivity/iterating_computation.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

import jax
import jax.numpy as jnp

import genjax
from genjax import bernoulli, gen, pretty

key = jax.random.key(0)
pretty()

"""
First start by creating a simple generative function
"""

@gen
def double_flip(p, q):
v1 = bernoulli(p) @ "v1"
v2 = bernoulli(q) @ "v2"
return v1 + v2

"""

Now we can create a vectorized version that takes a batch of p values and calls the function for each value in the batch. The `in_axes` tell the `vmap` combinator which arguments are mapped over, and which are not. The value `0` means we will map over this argument and `None` means we will not.
"""

batched_double_flip = double_flip.vmap(in_axes=(0, None))

"""
Now we can use the batched version to generate a batch of samples
"""

size_of_batch = 20

"""

To do so, we have to create batched keys and p values
"""

key, subkey = jax.random.split(key)
p = jax.random.uniform(subkey, (size_of_batch,))
q = 0.5

"""
We will run the generative function once for (p1, q), once for (p2, q), ...
"""

key, subkey = jax.random.split(key)
traces = batched_double_flip.simulate(subkey, (p, q))
traces.get_retval()

"""
We can also use call it on `(p1, q1)`, `(p2, q2)`, ...
"""

key, subkey = jax.random.split(key)
p = jax.random.uniform(subkey, (size_of_batch,))
key, subkey = jax.random.split(key)
q = jax.random.uniform(subkey, (size_of_batch,))
batched_double_flip_v2 = double_flip.vmap(in_axes=(0, 0))
key, subkey = jax.random.split(key)
traces = batched_double_flip_v2.simulate(subkey, (p, q))
traces.get_retval()

"""
Note: We cannot batch different variables with different shapes
"""

try:
key, subkey = jax.random.split(key)
p = jax.random.uniform(subkey, (size_of_batch,))
key, subkey = jax.random.split(key)
q = jax.random.uniform(subkey, (size_of_batch + 1,))
key, subkey = jax.random.split(key)
traces = batched_double_flip_v2.simulate(subkey, (p, q))
print(traces.get_retval())
except ValueError as e:
print(e)

"""
What about iterating `vmap`, e.g. if we want to apply a generative function acting on a pixel over a 2D space?
"""

image = jnp.zeros([300, 500], dtype=jnp.float32)

"""
We first create a function on one "pixel" value.
"""

@gen
def sample_pixel(pixel):
new_pixel = genjax.normal(pixel, 1.0) @ "new_pixel"
return new_pixel

key, subkey = jax.random.split(key)
tr = sample_pixel.simulate(subkey, (0.0,))
tr.get_choices()["new_pixel"]

"""
Now what if we want to apply a generative function over a 2D space?

We can use a nested `vmap` combinator:
"""

sample_image = sample_pixel.vmap(in_axes=(0,)).vmap(in_axes=(0,))
key, subkey = jax.random.split(key)
tr = sample_image.simulate(subkey, (image,))

"""
We can access the new_pixel value for each pixel in the image
"""

(
tr.get_choices(),
tr.get_choices()[0, 0, "new_pixel"],
tr.get_choices()[299, 499, "new_pixel"],
)

"""
We can wrap this model in a bigger model.
"""

image = jnp.zeros([2, 3], dtype=jnp.float32)

@gen
def model(p):
sampled_image = sample_image(image) @ "sampled_image"
return sampled_image[0] + p

key, subkey = jax.random.split(key)
tr = model.simulate(subkey, (0.0,))
tr

"""
We can use ellipsis to access the new_pixel value for each pixel in the image
"""

tr.get_choices()["sampled_image", :, :, "new_pixel"]

"""
Alternatively, we can flatten the 2 dimensions into one and use a single `vmap` combinator.
This can be more efficient in some cases and usually has a faster compile time.
"""

sample_image_flat = sample_pixel.vmap(in_axes=(0,))
key, subkey = jax.random.split(key)
tr = sample_image_flat.simulate(subkey, (image.flatten(),))

# resize the sample to the original shape

out_image = tr.get_choices()[:, "new_pixel"].reshape(image.shape)
out_image

"""
But wait, now I have a `jax.vmap` and a `genjax.vmap`, when do I use one vs another?

The rule of thumb is that `jax.vmap` should only be applied to deterministic code. In particular, `model.simulate` is deterministic per given random key which we control explicitly, so we can use `jax.vmap` along the desired axes on this one. However, functions that use `~` in a `@genjax.gen` function should not be vmapped using `jax.vmap` and one should one `model.vmap` (or equivalently `genjax.vmap`) instead.
"""

"""
Oh but my iteration is actually over time, not space, i.e. I may want to reuse the same model by composing it with itself, e.g. for a Hidden Markov Model (HMM).

For this, we can use the `scan` combinator.
"""

@gen
def hmm_kernel(x):
z = genjax.normal(x, 1.0) @ "z"
y = genjax.normal(z, 1.0) @ "y"
return y

@genjax.scan(n=10)
@gen
def hmm(x, \_):
x1 = hmm_kernel(x) @ "x1"
return x1, None

"""
Testing
"""

key, subkey = jax.random.split(key)
initial_x = 0.0
tr_1 = hmm.simulate(subkey, (initial_x, None))
print("Value of z at the beginning:")
tr_1.get_choices()[0, "x1", "z"]

print("Value of y at the end:")
tr_1.get_choices()[9, "x1", "y"]

tr_1.get_choices()[:, "x1", "z"]

"""
Alternatively, we can directly create the same HMM model
"""

@genjax.scan(n=10)
@gen
def hmm*v2(x, *):
z = genjax.normal(x, 1.0) @ "z"
y = genjax.normal(z, 1.0) @ "y"
return y, None

"""
Testing the second version.
"""

key, subkey = jax.random.split(key)
tr_2 = hmm_v2.simulate(subkey, (initial_x, None))
tr_2.get_choices()[0, "z"], tr_2.get_choices()[9, "y"], tr_2.get_choices()[:, "z"]

"""
Yet another alternative, we can call the generative function with a `repeat` combinator.
This will run the generative function multiple times on a single argument and return the results
"""

@genjax.gen
def model(y):
x = genjax.normal(y, 0.01) @ "x"
y = genjax.normal(x, 0.01) @ "y"
return y

arg = 3.0
key, subkey = jax.random.split(key)
tr = model.repeat(n=10).simulate(subkey, (arg,))

tr.get_choices()[:, "x"], tr.get_retval()

"""
It can for instance be combined with JAX's `vmap`.
"""

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, 3)
args = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
n = 3
tr = jax.jit(jax.vmap(model.repeat(n=n).simulate, in_axes=(0, None)))(keys, (args,))
tr.get_choices()

"""
Note that it's running a computation |keys| _ |args| _ |n| times, i.e. 45 times in this case
"""

tr.get_retval()

================================================
FILE: docs/cookbook/inactive/expressivity/masking.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### I want more dynamic features but JAX only accepts arrays with statically known sizes, what do I do? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/expressivity/masking.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

import jax
import jax.numpy as jnp
from PIL import Image

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import bernoulli, categorical, gen, normal, or_else, pretty

pretty()
key = jax.random.key(0)

"""
One classic trick is to encode all the options as an array and pick the desired value from the array with a dynamic one.

Here's a first example:
"""

@gen
def model(
i, means, vars
): # provide all the possible values and the dynamic index to pick from them
x = normal(means[i], vars[i]) @ "x"
return x

key, subkey = jax.random.split(key)
model.simulate(subkey, (7, jnp.arange(10, dtype=jnp.float32), jnp.ones(10)))

"""
Now, what if there's a value we may or may not want to get depending on a dynamic value?

In this case, we can use masking. Let's look at an example in JAX.
"""

non_masked = jnp.arange(9).reshape(3, 3)

non_masked

# mask the upper triangular part of the matrix

mask = jnp.mask_indices(3, jnp.triu)

non_masked[mask]

"""
We can use similar logic for generative functions in GenJAX.

Let's create an HMM using the scan combinator.
"""

state_size = 10
length = 10
variance = jnp.eye(state_size)
key, subkey = jax.random.split(key)
initial_state = jax.random.normal(subkey, (state_size,))

@genjax.gen
def hmm_step(x):
new_x = genjax.mv_normal(x, variance) @ "new_x"
return new_x

hmm = hmm_step.iterate_final(n=length)

"""
When we run it, we get a full trace.
"""

jitted = jax.jit(hmm.simulate)
key, subkey = jax.random.split(key)
trace = jitted(subkey, (initial_state,))
trace.get_choices()

"""
To get the partial results in the HMM instead, we can use the masked version of `iterate_final` as follows:
"""

stop_at_index = 5
pairs = jnp.arange(state_size) < stop_at_index
masked_hmm = hmm_step.masked_iterate_final()
key, subkey = jax.random.split(key)
choices = masked_hmm.simulate(subkey, (initial_state, pairs)).get_choices()
choices

"""
We see that we obtain a filtered choice map, with a selection representing the boolean mask array.
Within the filtered choice map, we have a static choice map where all the results are computed, without the mask applied to them.
This is generally what will happen behind the scene in GenJAX: results will tend to be computed and then ignored, which is often more efficient on the GPU rather than being too eager in trying to avoid to do computations in the first place.

Let's now use it in a bigger computation where the masking index is dynamic and comes from a sampled value.
"""

@gen
def larger_model(init, probs):
i = categorical(probs) @ "i"
mask = jnp.arange(10) < i
x = masked_hmm(init, mask) @ "x"
return x

key, subkey = jax.random.split(key)
init = jax.random.normal(subkey, (state_size,))
probs = jnp.arange(state_size) / sum(jnp.arange(state_size))
key, subkey = jax.random.split(key)
choices = larger_model.simulate(subkey, (init, probs)).get_choices()
choices

"""
We have already seen how to use conditionals in GenJAX models in the `conditionals` notebook. Behind the scene, it's using the same logic with masks.
"""

@gen
def cond_model(p):
pred = p > 0
arg_1 = (p,)
arg_2 = (p,)
v = (
or_else(
gen(lambda p: bernoulli(p) @ "v1"), gen(lambda p: bernoulli(-p) @ "v1")
)(pred, arg_1, arg_2)
@ "cond"
)
return v

key, subkey = jax.random.split(key)
choices = cond_model.simulate(subkey, (0.5,)).get_choices()
choices

"""
We see that both branches will get evaluated and a mask will be applied to each branch, whose value depends on the evaluation of the boolean predicate `pred`.
"""

"""
What's happening behind the scene for masked values in the trace? Simply put, even though the system computes values, they are ignored w.r.t. the math of inference.

We can check it on a simple example, with two versions of a model, where one has an extra masked variable `y`.
Let's first define the two versions of the model.
"""

@gen
def simple_model():
x = normal(0.0, 1.0) @ "x"
return x

@gen
def submodel():
y = normal(0.0, 1.0) @ "y"
return y

@gen
def model\*with_mask():
x = normal(0.0, 1.0) @ "x"

-   = submodel.mask()(False) @ "y"
    return x

@gen
def proposal(\_: genjax.Target):
x = normal(3.0, 1.0) @ "x"
return x

"""
Let's now test that on the same key, they return the exact same score:
"""

key, subkey = jax.random.split(key)

simple_target = genjax.Target(simple_model, (), C.n())
masked_target = genjax.Target(model_with_mask, (), C.n())
simple_alg = genjax.smc.Importance(simple_target, q=proposal.marginal())
masked_alg = genjax.smc.Importance(masked_target, q=proposal.marginal())

# TODO: something's fishy here with the math. Get the same whether I mask or not.

simple_alg.simulate(subkey, (simple_target,)).get_score() == masked_alg.simulate(
subkey, (masked_target,)
).get_score()

masked_alg.simulate(subkey, (masked_target,))

"""
Let's see a final example for an unknown number of objects that may evolve over time.
For this, we can use `vmap` over a masked object andd we get to choose which ones are masked or not.

Let's create a model consisting of a 2D image where each pixel is traced.
"""

@gen
def single_pixel():
pixel = normal(0.0, 1.0) @ "pixel"
return pixel

image_model = single_pixel.mask().vmap(in_axes=(0,)).vmap(in_axes=(0,))

"""
Let's create a circular mask around the image.
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def create_circle_mask(size=200, center=None, radius=80):
if center is None:
center = (size // 2, size // 2)

    y, x = jnp.ogrid[:size, :size]
    dist_from_center = jnp.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask

circle_mask = create_circle_mask()

plt.imshow(circle_mask, cmap="gray")
plt.show()

"""
Let's now sample from the masked image and play with the mask and inference.
"""

key, subkey = jax.random.split(key)

tr = image_model.simulate(subkey, (circle_mask,))
flag = tr.get_choices()[:, :, "pixel"].flag
im = flag \* tr.get_choices()[:, :, "pixel"].value

plt.imshow(im, cmap="gray", vmin=0, vmax=1)
plt.show()

"""
We can create a small animation by updating the mask over time using the GenJAX `update` function to ensure that the probabilistic parts are taken properly into account.
"""

number_iter = 10
fig, ax = plt.subplots()

# Load the image

image_path = "./ending_dynamic_computation.png" # Update with your image path
image = Image.open(image_path)

# Convert to grayscale if needed and resize to match new_im dimensions

image = image.convert("L") # Convert to grayscale
image = image.resize(im.shape[1::-1]) # Resize to match (height, width)

# Convert to NumPy array

image_array = jnp.array(image) / 255.0

images = []
for i in range(number*iter):
key, subkey = jax.random.split(key)
new_circle_mask = create_circle_mask(radius=10 \* i)
arg_diff = (genjax.Diff(new_circle_mask, genjax.UnknownChange),)
constraints = C.n()
update_problem = genjax.Update(constraints)
tr, *, _, _ = tr.edit(key, update_problem, arg_diff)
flag = tr.get_choices()[:, :, "pixel"].flag
new_im = flag \* (tr.get_choices()[:, :, "pixel"].value / 5.0 + image_array)
images.append([ax.imshow(new_im, cmap="gray", vmin=0, vmax=1, animated=True)])

ani = animation.ArtistAnimation(fig, images, interval=200, blit=True, repeat_delay=1000)

# Save the animation as a GIF

ani.save("masked_image_animation.gif", writer="pillow")

# Display the animation in the notebook

from IPython.display import HTML

HTML(ani.to_jshtml())

================================================
FILE: docs/cookbook/inactive/expressivity/mixture.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### How can I write a mixture of models in GenJAX? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/expressivity/mixture.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

from jax import random

from genjax import flip, gen, inverse_gamma, mix, normal

key = random.key(0)

"""
We simply use the `mix` combinator.
Note that the trace is the join of the traces of the different components.

We first define the three components of the mixture model as generative functions.
"""

@gen
def mixture_component_1(p):
x = normal(p, 1.0) @ "x"
return x

@gen
def mixture_component_2(p):
b = flip(p) @ "b"
return b

@gen
def mixture_component_3(p):
y = inverse_gamma(p, 0.1) @ "y"
return y

"""
The mix combinator take as input the logits of the mixture components, and args for each component of the mixture.
"""

@gen
def mixture_model(p):
z = normal(p, 1.0) @ "z"
logits = (0.3, 0.5, 0.2)
arg_1 = (p,)
arg_2 = (p,)
arg_3 = (p,)
a = (
mix(mixture_component_1, mixture_component_2, mixture_component_3)(
logits, arg_1, arg_2, arg_3
)
@ "a"
)
return a + z

key, subkey = random.split(key)
tr = mixture_model.simulate(subkey, (0.4,))
print("return value:", tr.get_retval())
print("value for z:", tr.get_choices()["z"])

"""
The combinator uses a fix address "mixture_component" for the components of the mixture model.
"""

print("value for the mixture_component:", tr.get_choices()["a", "mixture_component"])

================================================
FILE: docs/cookbook/inactive/expressivity/ravi_stack.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

## Nested approximate marginalisation & RAVI stacks [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/expressivity/ravi_stack.ipynb)

### How to be recursively wrong everywhere all the time yet correct at the end

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

import jax
import jax.numpy as jnp

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import SelectionBuilder as S
from genjax import Target, gen, pretty

pretty()

"""
Say you have a model of interest for which you want to do inference. It consists of a mixture of 3 Gaussians, two of which are close to each other while the other one is far. We will informally call cluster 1 the single Gaussian far from the others and cluster 2 the other two.
"""

@gen
def model():
idx = genjax.categorical(probs=[0.5, 0.25, 0.25]) @ "idx" # under the prior, 50% chance to be in cluster 1 and 50% chance to be in cluster 2.
means = jnp.array([0.0, 10.0, 11.0])
vars = jnp.array([1.0, 1.0, 1.0])
x = genjax.normal(means[idx], vars[idx]) @ "x"
y = genjax.normal(means[idx], vars[idx]) @ "y"
return x, y

obs1 = C["x"].set(1.0)
obs2 = C["x"].set(10.5)

"""
We will only care about the values of "x" and "y" in the output, so we will marginalize "idx" out.
"""

marginal_model = model.marginal(
selection=S["x"] | S["y"]
) # This means we are projection onto the variables x and y, marginalizing out the rest

"""
Testing the marginal model
"""

key = jax.random.key(0)
marginal_model.simulate(key, ())

tr, w = marginal_model.importance(key, obs1, ())
tr.get_choices()

"""
Now depending on what we observe, we will want to infer that the data was likely generated from one cluster (the single Gaussian far from the other ones) or the other (the two Gaussians close to each other).

Let's create a data-driven proposal that targets the model and will incorporate this logic.
In order to avoid being too eager in our custom logic, we may want to just use this as a probabilistic heuristics instead of a deterministic one. After all, it's possible that the value 10.5 for "x" was generated from the cluster with a single Gaussian.
"""

@gen
def proposal(target: Target):
x*obs = target.constraint["x"]
probs = jax.lax.cond(
x_obs < 5.0,
lambda *: jnp.array([0.9, 0.1]),
lambda \_: jnp.array([0.1, 0.9]),
operand=None,
) # if x_obs < 5, then our heuristics is to propose something closer to cluster 1 with probability 0.9, otherwise we propose in cluster 2 with probability 0.9.
cluster_idx = genjax.categorical(probs=probs) @ "cluster_idx"
means = jnp.array([0.0, 10.5]) # second cluster is more spread out so we use a larger variance
vars = jnp.array([1.0, 3.0])
y = genjax.normal(means[cluster_idx], vars[cluster_idx]) @ "y"
return y

"""
Testing the proposal.
"""

target = Target(marginal_model, (), obs1)
proposal.simulate(key, (target,))

"""
So now this may seem great, but we cannot yet use this proposal as an importance sampler for the model. The issue is that the traces produced by the proposal don't match the ones the model accepts: the model doesn't know what to do with "cluster_idx".
"""

k_particles = 500
alg = genjax.smc.ImportanceK(target, q=proposal.marginal(), k_particles=k_particles)

try:
alg.simulate(key, (target,))
except Exception as e: # TODO: this currently doesn't raise an exception but in the future it should
print(e)

"""
There again, we can use marginal to marginalise out the variable from the proposal.
"""

k_particles = 500
alg = genjax.smc.ImportanceK(
target, q=proposal.marginal(selection=S["y"]), k_particles=k_particles
)

alg.simulate(key, (target,))

================================================
FILE: docs/cookbook/inactive/expressivity/stochastic_probabilities.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### How to create and use distributions with inexact likelihood evaluations [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/expressivity/stochastic_probabilities.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

"""
This notebook builds on top of the `custom_distribution` one.
"""

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Pytree, Weight, pretty
from genjax.\_src.generative_functions.distributions.distribution import Distribution
from genjax.typing import Any

tfd = tfp.distributions
key = jax.random.key(0)
pretty()

"""
Recall how we defined a distribution for a Gaussian mixture, using the `Distribution` class.
"""

@Pytree.dataclass
class GaussianMixture(Distribution):
def random_weighted(
self, key: jax.random.key, probs, means, vars
) -> tuple[Weight, Any]:
probs = jnp.asarray(probs)
means = jnp.asarray(means)
vars = jnp.asarray(vars)
cat = tfd.Categorical(probs=probs)
cat_index = jnp.asarray(cat.sample(seed=key))
normal = tfd.Normal(loc=means[cat_index], scale=vars[cat_index])
key, subkey = jax.random.split(key)
normal_sample = normal.sample(seed=subkey)
zipped = jnp.stack([jnp.arange(0, len(probs)), means, vars], axis=1)
weight_recip = -jax.scipy.special.logsumexp(
jax.vmap(
lambda z: tfd.Normal(loc=z[1], scale=z[2]).log_prob(normal_sample) + tfd.Categorical(probs=probs).log_prob(z[0])
)(zipped)
)

        return weight_recip, normal_sample

    def estimate_logpdf(self, key: jax.random.key, x, probs, means, vars) -> Weight:
        zipped = jnp.stack([jnp.arange(0, len(probs)), means, vars], axis=1)
        return jax.scipy.special.logsumexp(
            jax.vmap(
                lambda z: tfd.Normal(loc=z[1], scale=z[2]).log_prob(x)
                + tfd.Categorical(probs=probs).log_prob(z[0])
            )(zipped)
        )

"""
In the class above, note in `estimate_logpdf` how we computed the density as a sum over all possible paths in the that could lead to a particular outcome `x`.

In fact, the same occurs in `random_weighted`: even though we know exactly the path we took to get to the sample `normal_sample`, when evaluating the reciprocal density, we also sum over all possible paths that could lead to that `value`.

Precisely, this required to sum over all the possible values of the categorical distribution `cat`. We technically sampled two random values `cat_index` and `normal_sample`, but we are only interested in the distribution on `normal_sample`: we marginalized out the intermediate random variable `cat_index`.

Mathematically, we have
`p(normal_sample) = sum_{cat_index} p(normal_sample, cat_index)`.
"""

"""
GenJAX supports a more general kind of distribution, that only need to be able to estimate their densities.
The correctness criterion for this to be valid are that the estimation should be unbiased, i.e. the correct value on average.

More precisely, `estimate_logpdf` should return an unbiased density estimate, while `random_weighted` should return an unbiased estimate for the reciprocal density. In general you can't get one from the other, as the following example shows.

Flip a coin and with $50%$ chance return $1$, otherwise $3$. This gives an unbiased estimator of $2$.
If we now return $\frac{1}{1}$ with 50%, and $\frac{1}{3}$ otherwise, the average value is $\frac{2}{3}$, which is not $\frac{1}{2}$.
"""

"""
Let's now define a Gaussian mixture distribution that only estimates its density.
"""

@Pytree.dataclass
class StochasticGaussianMixture(Distribution):
def random_weighted(
self, key: jax.random.key, probs, means, vars
) -> tuple[Weight, Any]:
probs = jnp.asarray(probs)
means = jnp.asarray(means)
vars = jnp.asarray(vars)
cat = tfd.Categorical(probs=probs)
cat_index = jnp.asarray(cat.sample(seed=key))
normal = tfd.Normal(loc=means[cat_index], scale=vars[cat_index])
key, subkey = jax.random.split(key)
normal_sample = normal.sample(seed=subkey) # We can estimate the reciprocal (marginal) density in constant time. Math magic explained at the end!
weight_recip = -tfd.Normal(
loc=means[cat_index], scale=vars[cat_index]
).log_prob(normal_sample)
return weight_recip, normal_sample

    # Given a sample `x`, we can also estimate the density in constant time
    # Math again explained at the end.
    # TODO: we could probably improve further with a better proposal
    def estimate_logpdf(self, key: jax.random.key, x, probs, means, vars) -> Weight:
        cat = tfd.Categorical(probs=probs)
        cat_index = jnp.asarray(cat.sample(seed=key))
        return tfd.Normal(loc=means[cat_index], scale=vars[cat_index]).log_prob(x)

"""
To test, we start by creating a generative function using our new distribution.
"""

sgm = StochasticGaussianMixture()

@genjax.gen
def model(cat_probs, means, vars):
x = sgm(cat_probs, means, vars) @ "x"
y_means = jnp.repeat(x, len(means))
y = sgm(cat_probs, y_means, vars) @ "y"
return (x, y)

"""
We can then simulate from the model, assess a trace, or use importance sampling with the default proposal, seemlessly.
"""

cat_probs = jnp.array([0.1, 0.4, 0.2, 0.3])
means = jnp.array([0.0, 1.0, 2.0, 3.0])
vars = jnp.array([1.0, 1.0, 1.0, 1.0])

key, subkey = jax.random.split(key)
tr = model.simulate(subkey, (cat_probs, means, vars))
tr

# TODO: assess currently raises a not implemented error, but we can use importance with a full trace instead

# model.assess(tr.get_choices(), (cat_probs, means, vars))

key, subkey = jax.random.split(key)
\_, w = model.importance(subkey, tr.get_choices(), (cat_probs, means, vars))
w

y = 2.0
key, subkey = jax.random.split(key)
model.importance(subkey, C["y"].set(y), (cat_probs, means, vars))

"""
Let's also check that `estimate_logpdf` from our distribution `sgm` indeed correctly estimates the density.
"""

gm = GaussianMixture()
x = 2.0
N = 42
n_estimates = 2000000
cat_probs = jnp.array(jnp.arange(1.0 / N, 1.0 + 1.0 / N, 1.0 / N))
cat_probs = cat_probs / jnp.sum(cat_probs)
means = jnp.arange(0.0, N \* 1.0, 1.0)
vars = jnp.ones(N) / N
key, subkey = jax.random.split(key)
log_density = gm.estimate_logpdf(subkey, x, cat_probs, means, vars) # exact value
log_density
jitted = jax.jit(jax.vmap(sgm.estimate_logpdf, in_axes=(0, None, None, None, None)))
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, n_estimates)
estimates = jitted(keys, x, cat_probs, means, vars)
log_mean_estimates = jax.scipy.special.logsumexp(estimates) - jnp.log(len(estimates))
log_density, log_mean_estimates

"""
One benefit of using density estimates instead of exact ones is that it can be much faster to compute.
Here's a way to test it, though it will not shine on this example as it is too simple.
We will explore examples in different notebooks where this shines more brightly.
"""

N = 30000
n_estimates = 10
cat_probs = jnp.array(jnp.arange(1.0 / N, 1.0 + 1.0 / N, 1.0 / N))
cat_probs = cat_probs / jnp.sum(cat_probs)
means = jnp.arange(0.0, N \* 1.0, 1.0)
vars = jnp.ones(N) / N

jitted_exact = jax.jit(gm.estimate_logpdf)
jitted_approx = jax.jit(
lambda key, x, cat_probs, means, vars: jax.scipy.special.logsumexp(
jax.vmap(sgm.estimate_logpdf, in_axes=(0, None, None, None, None))(
key, x, cat_probs, means, vars
)
) - jnp.log(n_estimates)
)

# warmup the jit

key, subkey = jax.random.split(key)
jitted_exact(subkey, x, cat_probs, means, vars)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, n_estimates)
jitted_approx(keys, x, cat_probs, means, vars)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, n_estimates)
%timeit jitted(keys, x, cat_probs, means, vars)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, n_estimates)
%timeit jitted_approx(keys, x, cat_probs, means, vars)

"""
Now, the reason we need both methods `random_weighted` and `estimate_logpdf` is that both methods will be used at different times, notably depending on whether we use the distribution in a proposal or in a model, as we show next.

Let's define a simple model and a proposal which both use our `sgm` distribution.
"""

@genjax.gen
def model(cat_probs, means, vars):
x = sgm(cat_probs, means, vars) @ "x"
y_means = jnp.repeat(x, len(means))
y = sgm(cat_probs, y_means, vars) @ "y"
return (x, y)

@genjax.gen
def proposal(obs, cat_probs, means, vars):
y = obs["y"] # simple logic to propose a new x: its mean was presumably closer to y
new_means = jax.vmap(lambda m: (m + y) / 2)(means)
x = sgm(cat_probs, new_means, vars) @ "x"
return (x, y)

"""
Let's define importance sampling once again. Note that it is exactly the same as the usual one!

This is because behind the scenes GenJAX implements `simulate` using `random_weighted` and `assess` using `estimate_logpdf`.
"""

def gensp*importance_sampling(target, obs, proposal):
def \_inner(key, target_args, proposal_args):
key, subkey = jax.random.split(key)
trace = proposal.simulate(key, \*proposal_args)
chm = obs ^ trace.get_choices()
proposal_logpdf = trace.get_score() # TODO: using importance instead of assess, as assess is not implemented
*, target_logpdf = target.importance(subkey, chm, \*target_args)
importance_weight = target_logpdf - proposal_logpdf
return (trace, importance_weight)

    return _inner

"""
Testing
"""

obs = C["y"].set(2.0)

key, subkey = jax.random.split(key)
gensp_importance_sampling(model, obs, proposal)(
subkey, ((cat_probs, means, vars),), ((obs, cat_probs, means, vars),)
)

"""
Finally, for those curious about the math magic that enabled to correctly (meaning unbiasedly) estimate the pdf and its reciprocal, there's a follow up cookbook on this!
"""

================================================
FILE: docs/cookbook/inactive/expressivity/stochastic_probabilities_math.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### Details on random_weighted and estimate_logpdf [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/expressivity/stochastic_probabilities_math.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

"""
Let's start with `estimate_logpdf`.
We have that the marginal distribution over the returned value `x` (the sample from the normal distribution) is given by
$$p(x) = \sum_i p(x\mid z=i) p(z=i)$$
where the sum is over the possible values of the categorical distribution, $p(x|z=i)$ is the density of the $i$-th normal at $x$, and $p(z=i)$ is the density of the categorical at the value $i$.

This sum can be rewritten as the expectation under the categorical distribution $p(z)$:
$$\sum_i p(x\mid z=i)p(z=i) = \mathbb{E}_{z\sim p(z)}[p(x\mid z)]$$  
This means we can get an unbiased estimate of the expectation by simply sampling a `z` and returning `p(x|z)`: the average value of this process is obviously its expectation (it's the definition on the expectation).
In other words, we proved that the estimation strategy used in `estimate_logpdf` indeed returns an unbiased estimate of the exact marginal.
"""

"""
Lastly, as we discussed above we cannot in general invert an unbiased estimate to get an unbiased estimate of the reciprocal, so one may be suspicious that the returned weight in `random_weighted` looks like the negation (in logspace) of the one returned in `estimate_logpdf`.
Here the argument is different, based on the following identity:
$$\frac{1}{p(x)} = \mathbb{E}_{z\sim p(z\mid x)}[\frac{1}{p(x\mid z)}]$$
The idea is that we can get an unbiased estimate if we can sample from the posterior $p(z|x)$. Given an $x$, this is an intractable sampling problem in general. However, in `random_weighted`, we sample a $z$ together with the $x$, and this $z$ is an exact posterior sample of $z$ that we get "for free".
Now to finish the explanation, the compact way to prove the identity is as follows.

$$
\begin{matrix}
\frac{1}{p(x)} &\\
= \frac{1}{p(x)} \mathbb{E}_{z \sim B}[p(z)] & \text{$p(z)$ density w.r.t. base measure $B$ and of total mass 1}\\
= \frac{1}{p(x)} \mathbb{E}_{z \sim p(z\mid x)}[\frac{p(z)}{p(z\mid x)}]   &\text{seeing $p(z|x)$ as an importance sampler for $B$}\\
= \mathbb{E}_{z \sim p(z\mid x)}[\frac{p(z)}{p(z\mid x)p(x)}]  & \text{$p(x)$ doesn't depend on $z$ moved within the expectation}\\
= \mathbb{E}_{z \sim p(z\mid x)}[\frac{p(z)}{p(z,x)}]   & \text{ definition of joint distribution}\\
= \mathbb{E}_{z \sim p(z\mid x)}[\frac{p(z)}{p(z)p(x|z)}] & \text{definition of conditional distribution}\\
=  \mathbb{E}_{z \sim p(z\mid x)}[\frac{1}{p(x|z)}]   & \text{simplification}
\end{matrix}
$$

"""

================================================
FILE: docs/cookbook/inactive/inference/custom_proposal.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### I'm doing importance sampling as advised but it's bad, what can I do? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/inference/custom_proposal.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

"""
One thing one can do is write a custom proposal for importance sampling.
The idea is to sample from this one instead of the default one used by genjax when using `model.importance`.
The default one is only informed by the structure of the model, and not by the posterior defined by both the model and the observations.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.special import logsumexp

from genjax import ChoiceMapBuilder as C
from genjax import Target, gen, normal, pretty, smc

key = jax.random.key(0)
pretty()

"""
Let's first define a simple model with a broad normal prior and some observations
"""

@gen
def model(): # Initially, the prior is a pretty broad normal distribution centred at 0
x = normal(0.0, 100.0) @ "x" # We add some observations, which will shift the posterior towards these values
_ = normal(x, 1.0) @ "obs1"
_ = normal(x, 1.0) @ "obs2"
\_ = normal(x, 1.0) @ "obs3"
return x

# We create some data, 3 observed values at 234

obs = C["obs1"].set(234.0) ^ C["obs2"].set(234.0) ^ C["obs3"].set(234.0)

"""
We then run importance sampling with a default proposal,
snd print the average weight of the samples, to give us a sense of how well the proposal is doing.
"""

key, \*sub_keys = jax.random.split(key, 1000 + 1)
sub_keys = jnp.array(sub_keys)
args = ()
jitted = jit(vmap(model.importance, in_axes=(0, None, None)))
trace, weight = jitted(sub_keys, obs, args)
print("The average weight is", logsumexp(weight) - jnp.log(len(weight)))
print("The maximum weight is", weight.max())

"""
We can see that both the average and even maximum weight are quite low, which means that the proposal is not doing a great job.

If there is no observations, ideally, the weight should center around 1 and be quite concentrated around that value.
A weight much higher than 1 means that the proposal is too narrow and is missing modes. Indeed, for that to happen, one has to sample a very unlikely value under the proposal which is very likely under the target.
A weight much lower than 1 means that the proposal is too broad and is wasting samples. This happens in this case as the default proposal uses the broad prior `normal(0.0, 100.0)` as a proposal, which is far from the observed values centred around $234.0$.

If there are observations, as is the case above, the weight should center around the marginal on the observations.
More precisely, if the model has density $p(x,y)$ where $y$ are the observations and the proposal has density $q(x)$, then a weight is given by $w = \frac{p(x,y)}{q(x)}$ whose average value over many runs (expectations under the proposal) is $p(y)$.
"""

"""
We now define a custom proposal, which will be a normal distribution centred around the observed values

"""

@gen
def proposal(obs):
avg_val = jnp.array(obs).mean()
std = jnp.array(obs).std()
x = (
normal(avg_val, 0.1 + std) @ "x"
) # To avoid a degenerate proposal, we add a small value to the standard deviation
return x

"""
To do things by hand first, let's reimplement the importance function.
It samples from the proposal and then computes the importance weight
"""

def importance*sample(target, obs, proposal):
def \_inner(key, target_args, proposal_args):
trace = proposal.simulate(key, \*proposal_args) # the full choice map under which we evaluate the model # has the sampled values from the proposal and the observed values
chm = obs ^ trace.get_choices()
proposal_logpdf = trace.get_score()
target_logpdf, * = target.assess(chm, \*target_args)
importance_weight = target_logpdf - proposal_logpdf
return (trace, importance_weight)

    return _inner

"""
We then run importance sampling with the custom proposal
"""

key, \*sub_keys = jax.random.split(key, 1000 + 1)
sub_keys = jnp.array(sub_keys)
args_for_model = ()
args_for_proposal = (jnp.array([obs["obs1"], obs["obs2"], obs["obs3"]]),)
jitted = jit(vmap(importance_sample(model, obs, proposal), in_axes=(0, None, None)))
trace, new_weight = jitted(sub_keys, (args_for_model,), (args_for_proposal,))

"""
We see that the new values, both average and maximum, are much higher than before, which means that the custom proposal is doing a much better job
"""

print("The new average weight is", logsumexp(new_weight) - jnp.log(len(new_weight)))
print("The new maximum weight is", new_weight.max())

"""
We can also do the same using the library functions.

To do this, let's first create a target posterior distribution. It consists of the model, arguments for it, and observations.
"""

target_posterior = Target(model, args_for_model, obs)

"""
Next, we redefine the proposal slightly to take the target as argument.
This way, it can extract the observations fro the target as we previously used.
But the target can for instance also depend on the arguments passed to the model.
"""

@gen
def proposal(target: Target):
model_obs = target.constraint
used_obs = jnp.array([model_obs["obs1"], model_obs["obs2"], model_obs["obs3"]])
avg_val = jnp.array(used_obs).mean()
std = jnp.array(used_obs).std()
x = normal(avg_val, 0.1 + std) @ "x"
return x

"""
Now, similarly to the importance_sampling notebook, we create an instance algorithm: it specifies a strategy to approximate our posterior of interest, `target_posterior`, using importance sampling with `k_particles`, and our custom proposal.

To specify that we use all the traced variables from `proposal` in importance sampling (we will revisit why that may not be the case in the ravi_stack notebook) are to be used, we will use `proposal.marginal()`. This indicates that no traced variable from `proposal` is marginalized out.
"""

k_particles = 1000
alg = smc.ImportanceK(target_posterior, q=proposal.marginal(), k_particles=k_particles)

"""
This will perform sampling importance resampling (SIR) with a $1000$ intermediate particles and one resampled and returned at the end which is returned. Testing
"""

jitted = jit(alg.simulate)
key, subkey = jax.random.split(key)
posterior_samples = jitted(subkey, (target_posterior,))
posterior_samples

================================================
FILE: docs/cookbook/inactive/inference/importance_sampling.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### I want to do my first inference task, how do I do it? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/inference/importance_sampling.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

"""
We will do it with importance sampling, which works as follows. We choose a distribution $q$ called a proposal that you we will sample from, and we need a distribution $p$ of interest, typically representing a posterior from a model having received observations.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import jit, vmap

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Target, bernoulli, beta, gen, pretty, smc

key = jax.random.key(0)
pretty()

"""
Let's first look at a simple python version of the algorithm to get the idea.
"""

def importance*sample(model, proposal):
def \_inner(key, model_args, proposal_args): # we sample from the easy distribution, the proposal `q`
trace = proposal.simulate(key, \*proposal_args)
chm = trace.get_choices() # we evaluate the score of the easy distribution q(x)
proposal_logpdf = trace.get_score() # we evaluate the score of the hard distribution p(x)
model_logpdf, * = model.assess(chm, \*model_args) # the importance weight is p(x)/q(x), which corrects for the bias from sampling from q instead of p
importance_weight = model_logpdf - proposal_logpdf
return (trace, importance_weight) # we return the trace and the importance weight

    return _inner

"""
We can test this on a very simple example.
"""

model = genjax.normal
proposal = genjax.normal

model_args = (0.0, 1.0)
proposal_args = (3.0, 4.0)

key, subkey = jax.random.split(key)
sample, importance_weight = jit(importance_sample(model, proposal))(
subkey, (model_args,), (proposal_args,)
)
print(importance_weight, sample.get_choices())

"""
We can also run it in parallel!
"""

jitted = jit(
vmap(
importance_sample(model, proposal),
in_axes=(0, None, None),
)
)
key, \*sub_keys = jax.random.split(key, 100 + 1)
sub_keys = jnp.array(sub_keys)
(sample, importance_weight) = jitted(sub_keys, (model_args,), (proposal_args,))
sample.get_choices(), importance_weight

"""
In GenJAX, every generative function comes equipped with a default proposal which we can use for importance sampling.

Let's define a generative function.
"""

@gen
def beta_bernoulli_process(u):
p = beta(1.0, u) @ "p"
v = bernoulli(p) @ "v"
return v

"""
By giving constraints to some of the random samples, which we call observations, we obtain a posterior inference problem where the goal is to infer the distribution of the random variables which are not observed.
"""

obs = C["v"].set(1)
args = (0.5,)

"""
The method `.importance` defines a default proposal based on the generative function which targets the posterior distribution we just defined.
It returns a pair containing a trace and the log incremental weight.
This weight corrects for the bias from sampling from the proposal instead of the intractable posterior distribution.
"""

key, subkey = jax.random.split(key)
trace, weight = beta_bernoulli_process.importance(subkey, obs, args)

trace, weight

N = 1000
K = 100

def SIR(N, K, model, chm):
@jit
def \_inner(key, args):
key, subkey = jax.random.split(key)
traces, weights = vmap(model.importance, in_axes=(0, None, None))(
jax.random.split(key, N), chm, args
)
idxs = vmap(jax.jit(genjax.categorical.simulate), in_axes=(0, None))(
jax.random.split(subkey, K), (weights,)
).get_retval()
samples = traces.get_choices()
resampled_samples = vmap(lambda idx: jtu.tree_map(lambda v: v[idx], samples))(
idxs
)
return resampled_samples

    return _inner

"""
Testing
"""

chm = C["v"].set(1)
args = (0.5,)
key, subkey = jax.random.split(key)
samples = jit(SIR(N, K, beta_bernoulli_process, chm))(subkey, args)
samples

"""
Another way to do the basically the same thing using library functions.

To do this, we first define a Target for importance sampling, i.e. the posterior inference problem we're targetting. It consists of a generative function, arguments to the generative function, and observations.
"""

target_posterior = Target(beta_bernoulli_process, (args,), chm)

"""
Next, we define an inference strategy algorithm (Algorithm class) to use to approximate the target distribution.

It's importance sampling with $N$ particles in our case.
"""

alg = smc.ImportanceK(target_posterior, k_particles=N)

"""
To get a different sense of what's going on, the hierarchy of classes is as follows:

`ImportanceK <: SMCAlgorithm <: Algorithm <: SampleDistribution <: Distribution <: GenerativeFunction <: Pytree`

In words, importance sampling (`ImportanceK`) is a particular instance of Sequential Monte Carlo ( `SMCAlgorithm`). The latter is one instance of approximate inference strategy (`Algorithm`).
An inference strategy in particular produces samples for a distribution (`SampleDistribution`), which is a distribution (`Distribution`) whose return value is the sample. A distribution here is the definition from GenSP (Lew et al 2023) which has two methods `random_weighted` and `estimate_logpdf`. See the appropriate cookbook for details on these.
Finally, a distribution is a particular case of generative function (`GenerativeFunction`), which are all pytrees (`Pytree`) to be JAX-compatible and in particular jittable.

"""

"""
To get K independent samples from the approximate posterior distribution, we can for instance use `vmap`.
"""

# It's a bit different from the previous example, because each of the final

# K samples is obtained by running a different set of N-particles.

# This can of course be optimized but we keep it simple here.

jitted = jit(vmap(alg.simulate, in_axes=(0, None)))

"""
Testing
"""

key, \*sub_keys = jax.random.split(key, K + 1)
sub_keys = jnp.array(sub_keys)
posterior_samples = jitted(sub_keys, (target_posterior,)).get_retval()

# This only does the importance sampling step, not the resampling step

# Therefore the shape is (K, N, 1)

posterior_samples["p"]

"""
We can check the mean value estimate for `"p"`.
"""

posterior_samples["p"].mean(axis=(0, 1))

"""
And we compare the relative difference with the one obtained using the previous method.
"""

100.0 \* jnp.abs(
samples["p"].mean() - posterior_samples["p"].mean(axis=(0, 1))
) / posterior_samples["p"].mean(axis=(0, 1)) # about 2% difference

================================================
FILE: docs/cookbook/inactive/inference/mcmc.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### What is MCMC? How do I use it? How do I write one? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/inference/mcmc.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from genjax import ChoiceMapBuilder as C
from genjax import gen, normal, pretty
from genjax.\_src.core.compiler.interpreters.incremental import Diff

key = jax.random.key(0)
pretty()

"""
We can first define a simple model using GenJAX.
"""

@gen
def model(x):
a = normal(0.0, 5.0) @ "a"
b = normal(0.0, 1.0) @ "b"
y = normal(a \* x + b, 1.0) @ "y"
return y

"""
Together with observations, this creates a posterior inference problem.
"""

obs = C["y"].set(4.0)

"""
The key ingredient in MCMC is a transition kernel.
We can write it in GenJAX as a function that takes a current trace and returns a new trace.

Let's write a simple Metropolis-Hastings (MH) kernel.
"""

def metropolis_hastings_move(mh_args, key): # For now, we give the kernel the full state of the model, the proposal, and the observations.
trace, model, proposal, proposal_args, observations = mh_args
model_args = trace.get_args()

    # The core computation is updating a trace, and for that we will call the model's update method.
    # The update method takes a random key, a trace, and a choice map object, and argument difference objects.
    argdiffs = Diff.no_change(model_args)
    proposal_args_forward = (trace, *proposal_args)

    # We sample the proposed changes to the trace.
    # This is encapsulated in a simple GenJAX generative function.
    key, subkey = jax.random.split(key)
    fwd_choices, fwd_weight, _ = proposal.propose(key, proposal_args_forward)

    new_trace, weight, _, discard = model.update(subkey, trace, fwd_choices, argdiffs)

    # Because we are using MH, we don't directly accept the new trace.
    # Instead, we compute a (log) acceptance ratio α and decide whether to accept the new trace, and otherwise keep the old one.
    proposal_args_backward = (new_trace, *proposal_args)
    bwd_weight, _ = proposal.assess(discard, proposal_args_backward)
    α = weight - fwd_weight + bwd_weight
    key, subkey = jax.random.split(key)
    ret_fun = jax.lax.cond(
        jnp.log(jax.random.uniform(subkey)) < α, lambda: new_trace, lambda: trace
    )
    return (ret_fun, model, proposal, proposal_args, observations), ret_fun

"""
We define a simple proposal distribution for the changes in the trace using a Gaussian drift around the current value of `"a"`.
"""

@gen
def prop(tr, \*\_):
orig_a = tr.get_choices()["a"]
a = normal(orig_a, 1.0) @ "a"
return a

"""
The overall MH algorithm is a loop that repeatedly applies the MH kernel,
which can conveniently be written using `jax.lax.scan`.
"""

def mh(trace, model, proposal, proposal_args, observations, key, num_updates):
mh_keys = jax.random.split(key, num_updates)
last_carry, mh_chain = jax.lax.scan(
metropolis_hastings_move,
(trace, model, proposal, proposal_args, observations),
mh_keys,
)
return last_carry[0], mh_chain

"""
Our custom MH algorithm is a simple wrapper around the MH kernel using our chosen proposal distribution.
"""

def custom_mh(trace, model, observations, key, num_updates):
return mh(trace, model, prop, (), observations, key, num_updates)

"""
We now want to create a function run_inference that takes the inference problem, i.e. the model and observations, a random key, and returns traces from the posterior.
"""

def run*inference(model, model_args, obs, key, num_samples):
key, subkey1, subkey2 = jax.random.split(key, 3) # We sample once from a default importance sampler to get an initial trace. # The particular initial distribution is not important, as the MH kernel will rejuvenate it.
tr, * = model.importance(subkey1, obs, model_args) # We then run our custom Metropolis-Hastings kernel to rejuvenate the trace.
rejuvenated_trace, mh_chain = custom_mh(tr, model, obs, subkey2, num_samples)
return rejuvenated_trace, mh_chain

"""
We add a little visualization function to validate the results.
"""

def validate_mh(mh_chain):
a = mh_chain.get_choices()["a"]
b = mh_chain.get_choices()["b"]
y = mh_chain.get_retval()
x = mh_chain.get_args()[0]
plt.plot(range(len(y)), a \* x + b)
plt.plot(range(len(y)), y, color="k")
plt.show()

"""
Testing the inference function.
"""

model*args = (5.0,)
num_samples = 40000
key, subkey = jax.random.split(key)
*, mh_chain = run_inference(model, model_args, obs, subkey, num_samples)
validate_mh(mh_chain)

================================================
FILE: docs/cookbook/inactive/library_author/dimap_combinator.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### What is this magic? [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChiSym/genjax/blob/main/docs/cookbook/inactive/library_author/dimap_combinator.ipynb)

"""

import sys

if "google.colab" in sys.modules:
%pip install --quiet "genjax[genstudio]"

"""
!! It is only meant to be use by library authors. It is used to implement other combinators such as `or_else`, and `repeat`.
"""

import jax
import jax.numpy as jnp

from genjax import gen, normal, pretty
from genjax.\_src.core.generative import GenerativeFunction
from genjax.\_src.core.typing import Callable, ScalarFlag

key = jax.random.key(0)
pretty()

"""
Here's an example of rewriting the `OrElseCombinator` combinator using `contramap` and `switch`.
"""

def NewOrElseCombinator(
if_gen_fn: GenerativeFunction,
else_gen_fn: GenerativeFunction,
) -> GenerativeFunction:
def argument_mapping(b: ScalarFlag, if_args: tuple, else_args: tuple):
idx = jnp.array(jnp.logical_not(b), dtype=int)
return (idx, if_args, else_args)

    # The `contramap` method is used to map the input arguments to the expected input of the generative function, and then call the switch combinator
    return if_gen_fn.switch(else_gen_fn).contramap(argument_mapping)

"""
To add a version accessible as decorator
"""

def new_or_else(
else_gen_fn: GenerativeFunction,
) -> Callable[[GenerativeFunction], GenerativeFunction]:
def decorator(if_gen_fn) -> GenerativeFunction:
return NewOrElseCombinator(if_gen_fn, else_gen_fn)

    return decorator

"""
To add a version accessible using postfix syntax, one would need to add the following method as part of the `GenerativeFunction` dataclass in `core.py`.
"""

def postfix_new_or_else(self, gen_fn: "GenerativeFunction", /) -> "GenerativeFunction":
return new_or_else(gen_fn)(self)

"""
Testing the rewritten version on an example
"""

@gen
def if_model(x):
return normal(x, 1.0) @ "if_value"

@gen
def else_model(x):
return normal(x, 5.0) @ "else_value"

@gen
def model(toss: bool):
return NewOrElseCombinator(if_model, else_model)(toss, (1.0,), (10.0,)) @ "tossed"

key, subkey = jax.random.split(key)
tr = jax.jit(model.simulate)(subkey, (True,))
tr.get_choices()

"""
Checking that the two versions are equivalent on an example
"""

@new_or_else(else_model)
@gen
def or_else_model(x):
return normal(x, 1.0) @ "if_value"

@gen
def model_v2(toss: bool):
return or_else_model(toss, (1.0,), (10.0,)) @ "tossed"

# reusing subkey to get the same result

tr2 = jax.jit(model_v2.simulate)(subkey, (True,))
tr.get_choices() == tr2.get_choices()

================================================
FILE: docs/cookbook/inactive/update/1_importance.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### Intro to the `update` logic

"""

import jax
import jax.numpy as jnp

from genjax import ChoiceMapBuilder as C
from genjax import gen, normal, pretty

pretty()
key = jax.random.key(0)

"""
One of the most important building block of the library is the `update` method. Before investigating its details, let's look at the more user-friendly version called `importance`.

`importance` is a method on generative functions. It takes a key, constraints in the form of a choicemap, and arguments for the generative function. Let's first see how we use it and then explain what happened.
"""

@gen
def model(x):
y = normal(x, 1.0) @ "y"
z = normal(y, 1.0) @ "z"
return y + z

constraints = C.n()
args = (1.0,)
key, subkey = jax.random.split(key)
tr, w = model.importance(subkey, constraints, args)

"""
We obtain a pair of a trace `tr` and a weight `w`. `tr` is produced by the model, and its choicemap satisfies the constraints given by `constraints`.

For the choices that are not constrained, they are sampled from the prior distribution given by the model.
"""

# we expect normal(0., 1.) for y and constant 4. for z

constraints = C["z"].set(4.0)
args = (0.0,)

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, 100000)
trs, ws = jax.vmap(lambda key: model.importance(key, constraints, args))(keys)
import matplotlib.pyplot as plt
import numpy as np

ys = trs.get_choices()["y"]
zs = trs.get_choices()["z"]
plt.hist(ys, bins=200, density=True, alpha=0.5, color="b", label="ys")
plt.scatter(zs, np.zeros_like(zs), color="r", label="zs")
plt.title("Gaussian Distribution of ys and Constant z")
plt.legend()
plt.show()

"""
The weights computed represent the ratio $\frac{P(y, 4. ; x)}{P(y ; x)}$ where $P(y, z ; x)$ is the joint density given by the model at the argument $x$, and $P(y ; x)$ is the density of the subpart of the model that does not contain the constrained variables. As "z" is constrained in our example, it only leaves "y".

We can easily check this:
"""

numerators, \_ = jax.vmap(lambda y: model.assess(C["y"].set(y) ^ C["z"].set(4.0), args))(
ys
)

denominators = trs.get_subtrace("y").get_score()

# yeah, numerical stability of floats implies it's not even exactly equal ...

jnp.allclose(ws, numerators - denominators, atol=1e-7)

"""
More generally the denominator is the joint on the sampled variables (the constraints are not sampled) and Gen has a way to automatically sampled from the generative function obtained by replacing the sampling operations of the constrained addresses by the values of the constraints. For instance in our example it would mean:
"""

@gen
def constrained_model(x):
y = normal(x, 1.0) @ "y"
z = 4.0
return y + z

"""
Thanks to the factorisation $P(y, z ; x) = P(y ; x)P(z | y ; x)$, the weight `ws` simplifies to $P(z | y ; x)$.
In fact we can easily check it
"""

ws == trs.get_subtrace("z").get_score()

"""
And this time the equality is exact as this is how `importance` computes it. The algebraic simplification $\frac{P(y ; x)}{P(y ; x)}=1$ is done automatically.
"""

"""
Let's review. `importance` completes a set of constraints given by a partial choicemap to a full choicemap under the model. It also efficiently computes a weight which simplifies to a distribution of the form $P(\text{sampled } | \text{ constraints} ; \text{arguments})$.

The complex recursive nature of this formula becomes a bit more apparent in the following example:
"""

@gen
def fancier_model(x):
y1 = normal(x, 1.0) @ "y1"
z1 = normal(y1, 1.0) @ "z1"
y2 = normal(z1, 1.0) @ "y2"
z2 = normal(z1 + y2, 1.0) @ "z2"
return y2 + z2

# if we constraint `z1` to be 4. and `z2` to be 2. we'd get a constrained model as follows:

@gen
def constrained_fancier_model(x):
y1 = normal(x, 1.0) @ "y1"
z1 = 4.0
y2 = normal(z1, 1.0) @ "y2" # note how the sampled `y2` depends on a constraint
z2 = 2.0
return y1 + z1 + y2 + z2

"""

### But what does this have to do this importance sampling?

What we effectively did was to sample a value `y` from the distribution `constrained_model`, which is called a proposal in importance sampling, often noted $q$. We then computed the weight $\frac{p(y)}{q(y)}$ under some model $p$.
Given that we constrained `z`, an equivalent view is that we observed `z` and we have a posterior inference problem: we want to approximately sample from the posterior $P(y | z)$ (all for a given argument `x`).

Note that $P(y | z) = \frac{P(y,z)}{P(z)}$ by Bayes rule.
So our fraction $\frac{P(y, z ; x)}{P(y ; x)}$ for the weight rewrites as $\frac{P(y | z)P(z)}{q(y)}= P(z)\frac{p(y)}{q(y)}$ (1).

Also remember that the weight $\frac{dp}{dq}$ for importance comes from the proper weight guarantee, i.e. it satisfies this equation: $$\forall f.\mathbb{E}_{y\sim p}[f(y)]= \mathbb{E}_{y\sim q}[\frac{dp}{dq}(y)f(y)] =  \frac{1}{p(z)} \mathbb{E}_{y\sim q}[w(y)f(y)] $$

where in the last step we used (1) and called `w` the weight computed by `importance`.

By taking $f:= \lambda y.1$, we derive that $p(z) = \mathbb{E}_{y\sim q}[w(y)]$. That is, by sampling from our proposal distribution, we can estimate the marginal $p(z)$. Theferore with the same samples we can estimate any quantity $\mathbb{E}_{y\sim p}[f(y)]$ using our estimate of $\mathbb{E}_{y\sim q}[w(y)f(y)]$ and our estimate of $p(z)$. That's the essence of self-normalizing importance sampling.

"""

"""

### The special case of the fully constrained choicemap

"""

"""
In the case where we give constraints that are a full choicemap for the model, `importance` returns the same value as `assess`.
"""

args = (1.0,)
key, subkey = jax.random.split(key)
tr = model.simulate(key, args)

constraints = tr.get*choices()
new_tr, w = model.importance(subkey, constraints, args)
score, * = model.assess(constraints, args)
w == score

================================================
FILE: docs/cookbook/inactive/update/2_update.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### Compositional Incremental Weight Computation

"""

import jax

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import gen, normal, pretty

pretty()
key = jax.random.key(0)

"""
Let's now see how `importance` and `update` are related.

The high level is that

-   `importance` starts with an empty trace, adds the constraints, and then samples the missing values to form a full choicemap under the moek
-   `update` starts with any trace, overwrites those given by the constraints, and samples the missing ones. The missing ones can come from the initial trace possibly having an incomplete choicemap for the model, but also if some constraints force the computation in the model to take a different path which has different sampled values.
    -   It also returns a weight ratio which generalizes the one from `importance`.
    -   It also takes and returns additional data to make it compositional, i.e. `update` is defined inductively on the structure of the `model`.
        """

@gen
def model(x):
y = normal(x, 1.0) @ "y"
z = normal(y, 1.0) @ "z"
return y + z

"""
Let's first check that `update` does not do anything if we provide no changes nor constraints.
"""

args = (1.0,)
key, subkey = jax.random.split(key)
tr = model.simulate(subkey, args)

constraints = C.n()
argdiffs = genjax.Diff.no*change(args)
key, subkey = jax.random.split(key)
new_trace, *, _, _ = model.update(subkey, tr, constraints, argdiffs)
new_trace == tr

"""
Let's now check that it returns a trace where the constraints overwrite the value from the initial trace.
"""

key, subkey = jax.random.split(key)
constraints = C["y"].set(3.0)
new*tr, *, _, _ = model.update(subkey, tr, constraints, argdiffs)
new_tr.get_choices()["y"] == 3.0 and new_tr.get_choices()["z"] == tr.get_choices()["z"]

"""
Next, let's look at the new input and new outputs compared to `importance`.
"""

args = (1.0,)
key, subkey = jax.random.split(key)
tr = model.simulate(subkey, args)

constraints = C["z"].set(3.0)
argdiffs = genjax.Diff.no_change(args)
new_trace, weight, ret_diff, discarded = model.update(subkey, tr, constraints, argdiffs)
argdiffs, ret_diff, discarded

"""
`discarded` represents a choicemap of the choices that were overwritten by the constraints.
"""

discarded["z"] == tr.get_choices()["z"]

"""
`argdiffs` and `ret_diff` use a special `Diff` type which is a simpler analogue of dual-numbers from automatic differentiation (AD). They represent a pair of a primal value and a tangent value.
In AD, the primal would be the point at which we're differentiating the function and the dual would be the derivative of the current variable w.r.t. an outside variable.

Here, the tangent type is much simpler and Boolean. It either consists of the `NoChange()` tag or the `UnknownChange` tag.
This is inspired by the literature on incremental computation, and is only there for efficiently computing the density ratio `weight` by doing algebraic simplifications at compile time as we have briefly seen for the case of `importance` in the previous cookbook.

The idea is that a change in the argument `x` of the generative function implies a change to the distribution on `y`. So given a value of `y`, when we want to compute its density we need to know the value of `x`. Maybe a change in `x` would force resampling a different variable `y`, which would then force a change on the distribution on `z`. That's the basic idea behind the `Diff` system and why it needs to be compositional. It's a form of dependency tracking to check which distributions might have changed given a change in arguments, and importantly know which ones didn't change for sure so we can apply some algebraic simplifications.
"""

"""

### Now what about the weight? what does it compute?

"""

"""
Let's denote a trace by a pair `(x,t)` of the argument `x` and the choicemap `t`.
Given a trace `(x,t)`, a new argument `x'`, and a map of constraints `u`, `update` returns a new trace `(x', t')` that is consistent with `u`. The values of choices in `t'` are copied from `t` and `u` (with `u` taking precedence) or sampled from the internal proposal $q$ (i.e. the equivalent to `constrained_model` that we have seen in the `importance` cookbook).

The weight $w$ satisfies $$w_{update} = \frac{p(t' ; x)}{q(t' ; x', t+u).p(t ; x)}$$
where $t+u$ denotes the choicemap where `u` overwrites the values in `t` on their common addresses.

Let's contrast it with the weight $w$ computed by importance which we can write as
$$w_{importance}\frac{p(t' ; x)}{q(t' ; x, u)}$$
which we can see as the special case of `update` with an empty starting trace `t`.
"""

"""

### What to do with the weight from `update`?

"""

"""
One simple thing is that given a trace with choicemap $y$ and a full choicemap $y'$ used as a constraint, `update` will not need to call the internal proposal `q` and the weight returned will be $\frac{p(y')}{p(y)}$. This is a useful quantity that appears in many SMC algorithms, and for instance in the ratio in the MH acceptance ratio $\alpha$.

Given a current value `y` for the choicemap and proposed value `u` for a change in some variables of the choicemap, if we call the model `p` and the proposal `q` (a kernel which may depend on `y` and proposes the value `u`), we write $y':= y+u$. Then, the MH acceptance ratio is defined as $$\alpha := \frac{q(y | y')p(y')}{p(y)q(y' | y)} = \frac{q(y | y')}{q(y' | y)}w_{update}$$

"""

"""

### A convenient usage of `update`

"""

"""
`update` has a derived convenient usage. If you have a trace `tr` and want to do some inference move, e.g. propose a new value for a specific address "x". We obtain a new trace with the new value for "x" using `update`:

```
new_tr, _ = model.update(subkey, tr, C["x"].set(new_val_for_x), genjax.Diff.no_change(args))
```

"""

================================================
FILE: docs/cookbook/inactive/update/3_speed_gains.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### Compute gains via incremental computation or how to not compute log pdfs

"""

import timeit

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import gen, normal, pretty
from genjax.\_src.core.pytree import Const

pretty()
key = jax.random.PRNGKey(0)

"""
In the previous cookbooks, we have seen that `importance` and `update` do algebraic simplifications in the weight ratios that they are computing.
Let's first see the difference in the case of `importance` by testing a naive version of sampling importance resampling (SIR) to one using `importance`.
"""

"""
Let's define a model to be used in the rest ot the cookbook.
"""

@gen
def model(size_model: Const[int]):
size_model = size_model.unwrap()
x = normal(0.0, 1.0) @ "x"
a = normal.vmap()(jnp.zeros(size_model), jnp.ones(size_model)) @ "a"
b = normal.vmap()(jnp.zeros(size_model), jnp.ones(size_model)) @ "b"
c = normal.vmap()(jnp.zeros(size_model), jnp.ones(size_model)) @ "c"
obs = normal(jnp.sum(a) + jnp.sum(b) + jnp.sum(c) + x, 5.0) @ "obs"
return obs

"""
To compare naive SIR to the one using `importance` and the default proposal, let's write define the default proposal manually:
"""

@gen
def default\*proposal(size_model: Const[int]):
size_model = size_model.unwrap()

-   = normal(0.0, 1.0) @ "x"
    _ = normal.vmap()(jnp.zeros(size_model), jnp.ones(size_model)) @ "a"
    _ = normal.vmap()(jnp.zeros(size\*model), jnp.ones(size_model)) @ "b"
-   = normal.vmap()(jnp.zeros(size_model), jnp.ones(size_model)) @ "c"
    return None

"""
Let's now write SIR with a parameter controlling whether to call the slow or fast version.
"""

obs = C["obs"].set(
1.0,
)

def sir(key, N: int, use_fast: bool, size_model):
if use_fast:
traces, weights = jax.vmap(model.importance, in_axes=(0, None, None))(
jax.random.split(key, N), obs, size_model
)
else:
traces = jax.vmap(default_proposal.simulate, in_axes=(0, None))(
jax.random.split(key, N), size_model
)

        chm_proposal = traces.get_choices()
        q_weights, _ = jax.vmap(
            lambda idx: default_proposal.assess(
                jax.tree_util.tree_map(lambda v: v[idx], chm_proposal), size_model
            )
        )(jnp.arange(N))

        chm_model = chm_proposal | C["obs"].set(jnp.ones(N) * obs["obs"])
        p_weights, _ = jax.vmap(
            lambda idx: model.assess(
                jax.tree_util.tree_map(lambda v: v[idx], chm_model), size_model
            )
        )(jnp.arange(N))

        weights = p_weights - q_weights

    idx = genjax.categorical.simulate(key, (weights,)).get_retval()
    samples = traces.get_choices()
    resampled = jax.tree_util.tree_map(lambda v: v[idx], samples)
    return resampled

"""
Let's now compare the speed of the 2 versions (beware there's some variance in the estimate, but adding more trials makes the runtime comparison take a while).
"""

obs = C["obs"].set(
1.0,
)
model_sizes = [10, 100, 1000]
N_sir = 100
num_trials = 30
slow_times = []
fast_times = []

for model_size in model_sizes:
total_time_slow = 0
total_time_fast = 0
model_size = Const(model_size)
obs = C["obs"].set(
1.0,
)
key, subkey = jax.random.split(key)

    # warm up run to trigger jit compilation
    jitted = jax.jit(sir, static_argnums=(1, 2))
    jitted(subkey, N_sir, False, (Const(model_sizes),))
    jitted(subkey, N_sir, True, (Const(model_sizes),))

    # measure time for each algorithm
    key, subkey = jax.random.split(key)
    total_time_slow = timeit.timeit(
        lambda: jitted(subkey, N_sir, False, (Const(model_sizes),)), number=num_trials
    )
    total_time_fast = timeit.timeit(
        lambda: jitted(subkey, N_sir, True, (Const(model_sizes),)), number=num_trials
    )

    average_time_slow = total_time_slow / num_trials
    average_time_fast = total_time_fast / num_trials
    slow_times.append(average_time_slow)
    fast_times.append(average_time_fast)

plt.plot(model_sizes, [time for time in slow_times], marker="o", label="Slow Algorithm")
plt.plot(model_sizes, [time for time in fast_times], marker="o", label="Fast Algorithm")
plt.xscale("log")
plt.xlabel("Argument (n)")
plt.ylabel("Average Time (seconds)")
plt.title("Average Execution Time of MH move for different model sizes")
plt.grid(True)
plt.legend()
plt.show()

"""
When doing inference with iterative algorithms like MCMC, we often need to make small adjustments to the choice map.
We have seen that `update` can be used to compute part of the MH acceptance ratio.
So now let's try to compare two versions of an MH move, one computing naively thee ratio and one using update.
"""

"""
Let's create a very basic kernel to rejuvenate the variable "x" in an MH algorithm.
"""

@gen
def rejuv_x(x):
x = normal(x, 1.0) @ "x"
return x

"""
Let's now write 2 versions of computing the MH acceptance ratio as well as the MH algorithm to rejuvenate the variable "x".
"""

def compute*ratio_slow(key, fwd_choice, fwd_weight, model_args, chm):
model_weight_old, * = model.assess(chm, model*args)
new_chm = fwd_choice | chm
model_weight_new, * = model.assess(new*chm, model_args)
old_x = C["x"].set(chm["x"])
proposal_args_backward = (fwd_choice["x"],)
bwd_weight, * = rejuv_x.assess(old_x, proposal_args_backward)
α = model_weight_new - model_weight_old - fwd_weight + bwd_weight
return α

def compute*ratio_fast(key, fwd_choice, fwd_weight, model_args, trace):
argdiffs = genjax.Diff.no_change(model_args)
*, weight, _, discard = model.update(key, trace, fwd_choice, argdiffs)
proposal_args_backward = (fwd_choice["x"],)
bwd_weight, _ = rejuv_x.assess(discard, proposal_args_backward)
α = weight - fwd_weight + bwd_weight
return α

def metropolis*hastings_move(key, trace, use_fast):
model_args = trace.get_args()
proposal_args_forward = (trace.get_choices()["x"],)
key, subkey = jax.random.split(key)
fwd_choice, fwd_weight, * = rejuv_x.propose(subkey, proposal_args_forward)
key, subkey = jax.random.split(key)

    if use_fast:
        α = compute_ratio_fast(subkey, fwd_choice, fwd_weight, model_args, trace)
    else:
        chm = trace.get_choices()
        α = compute_ratio_slow(subkey, fwd_choice, fwd_weight, model_args, chm)

    old_choice = C["x"].set(trace.get_choices()["x"])
    key, subkey = jax.random.split(key)
    ret_trace = jax.lax.cond(
        jnp.log(jax.random.uniform(subkey)) < α, lambda: fwd_choice, lambda: old_choice
    )
    return ret_trace

"""
Let's measure the performance of each variant.
"""

model_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000]
slow_times = []
fast_times = []

for model_size in model_sizes:
total_time_slow = 0
total_time_fast = 0
num_trials = 5000 if model_size <= 1000000 else 100
model_size = Const(model_size)
obs = C["obs"].set(
1.0,
)
key, subkey = jax.random.split(key)

    # create a trace from the model of the right size
    tr, _ = jax.jit(model.importance, static_argnums=(2))(subkey, obs, (model_size,))

    # warm up run to trigger jit compilation
    jitted = jax.jit(metropolis_hastings_move, static_argnums=(2))
    jitted(subkey, tr, False)
    jitted(subkey, tr, True)

    # measure time for each algorithm
    key, subkey = jax.random.split(key)
    total_time_slow = timeit.timeit(
        lambda: jitted(subkey, tr, False), number=num_trials
    )
    total_time_fast = timeit.timeit(lambda: jitted(subkey, tr, True), number=num_trials)
    average_time_slow = total_time_slow / num_trials
    average_time_fast = total_time_fast / num_trials
    slow_times.append(average_time_slow)
    fast_times.append(average_time_fast)

"""
Plotting the results.
"""

plt.figure(figsize=(20, 5))

# First half of the values

plt.subplot(1, 2, 1)
plt.plot(
model*sizes[: len(model_sizes) // 2],
[time * 1000 for time in slow*times[: len(slow_times) // 2]],
marker="o",
label="No incremental computation",
)
plt.plot(
model_sizes[: len(model_sizes) // 2],
[time * 1000 for time in fast_times[: len(fast_times) // 2]],
marker="o",
label="Default incremental computation",
)
plt.xscale("log")
plt.xlabel("Argument (n)")
plt.ylabel("Average Time (milliseconds)")
plt.title("Average Execution Time of MH move for different model sizes (First Half)")
plt.grid(True)
plt.legend()

# Second half of the values

plt.subplot(1, 2, 2)
plt.plot(
model*sizes[len(model_sizes) // 2 :],
[time * 1000 for time in slow*times[len(slow_times) // 2 :]],
marker="o",
label="No incremental computation",
)
plt.plot(
model_sizes[len(model_sizes) // 2 :],
[time * 1000 for time in fast_times[len(fast_times) // 2 :]],
marker="o",
label="Default incremental computation",
)
plt.xscale("log")
plt.xlabel("Argument (n)")
plt.ylabel("Average Time (milliseconds)")
plt.title("Average Execution Time of MH move for different model sizes (Second Half)")
plt.grid(True)
plt.legend()

plt.show()

================================================
FILE: docs/cookbook/inactive/update/4_index_request.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

### Speed Gains Part 2: Optimizing updates for vmap

"""

import timeit

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import (
IndexRequest,
StaticRequest,
Update,
gen,
normal,
pretty,
)
from genjax.\_src.core.pytree import Const

pretty()
key = jax.random.key(0)

"""
As we discussed in the previous cookbook entries, a main point of `update` is to be used for incremental computation: `update` performs algebraic simplifications of the logpdf-ratios computed in the weight that it returns. This is tracked through the `Diff` system.

A limitation of the current automation is that if an address "x" has a tensor value, and any index of "x" changes, the system will consider that "x" has changed without capturing a finer description of what exactly changed.

However, we can manually specify how something has changed in a more specific way.
"""

@gen
def model(size_model_const: Const[int]):
size_model = size_model_const.unwrap()
x = normal(0.0, 1.0) @ "x"
a = normal.vmap()(jnp.zeros(size_model), jnp.ones(size_model)) @ "a"
b = normal.vmap()(jnp.zeros(size_model), jnp.ones(size_model)) @ "b"
c = normal.vmap()(jnp.zeros(size_model), jnp.ones(size_model)) @ "c"
obs = normal(jnp.sum(a) + jnp.sum(b) + jnp.sum(c) + x, 5.0) @ "obs"
return obs

"""
Let's create a trace from our model.
"""

obs = C["obs"].set(
1.0,
)
size*model = 10000
args = (Const(size_model),)
key, subkey = jax.random.split(key)
tr, * = model.importance(subkey, obs, args)

"""
Let's first see an equivalent way to perform do what `update` does.
Just like `update` generalizes `importance`, there is yet another more general interface, `edit`, which generalizes `update`.

We will go into the details of `edit` in a follow up cookbook.
For now, let's see the equivalent of `update` using `edit`. For this, we introduce a `Request` to change the trace.
`edit` will then answer the `Request` and change the trace following the logic of the request.
To mimick `update`, we will perform an `Update` request.
"""

change_in_value_for_a = jnp.ones(size_model)

# usual update

constraints = C["a"].set(change*in_value_for_a)
argdiffs = genjax.Diff.no_change(args)
key, subkey = jax.random.split(key)
new_tr1, *, _, _ = tr.update(subkey, constraints, argdiffs)

# update using `Request`

val = C.v(change*in_value_for_a)
request = StaticRequest({"a": Update(val)})
key, subkey = jax.random.split(key)
new_tr2, *, _, _ = request.edit(subkey, tr, args)

# comparing the values of both choicemaps after the update

jax.tree_util.tree_all(
jax.tree.map(jnp.allclose, new_tr1.get_choices(), new_tr2.get_choices())
)

"""
Now let's see how we can efficiently change the value of "a" at a specific index.
For that, we create a more specific `Request` called an `IndexRequest`. This request expects another request for what to do at the given index.
"""

request = StaticRequest({"a": IndexRequest(jnp.array(3), Update(C.v(42.0)))})

key, subkey = jax.random.split(key)
new*tr, *, _, _ = request.edit(subkey, tr, args)

# Checking we only made one change by checking that only one value in the choicemap is 42

jnp.sum(new_tr.get_choices()["a"] == 42.0) == 1

"""
Now, let's compare the 3 options: naive density ratio computation vs `update` vs `IndexRequest`.
For this, we will do a comparison of doing an MH move on a specific variable in the model as we did in the previous cookbook, but this time for a specific index of the traced value "a".
We will also compare
"""

IDX_WHERE_CHANGE_A = 3

@gen
def rejuv_a(a):
a = normal(a, 1.0) @ "a"
return a

def compute*ratio_slow(key, fwd_choice, fwd_weight, model_args, chm):
model_weight_old, * = model.assess(chm, model*args)
new_a = chm["a"].at[IDX_WHERE_CHANGE_A].set(fwd_choice["a"])
new_chm = C["a"].set(new_a) | chm
model_weight_new, * = model.assess(new_chm, model_args)

    old_a = C["a"].set(chm["a", IDX_WHERE_CHANGE_A])
    proposal_args_backward = (fwd_choice["a"],)
    bwd_weight, _ = rejuv_a.assess(old_a, proposal_args_backward)
    α = model_weight_new - model_weight_old - fwd_weight + bwd_weight
    return α

def compute*ratio_fast(key, fwd_choice, fwd_weight, model_args, trace):
argdiffs = genjax.Diff.no_change(model_args)
constraint = C["a"].set(
trace.get_choices()["a"].at[IDX_WHERE_CHANGE_A].set(fwd_choice["a"])
)
*, weight, _, discard = model.update(key, trace, constraint, argdiffs)
proposal_args_backward = (fwd_choice["a"],)
bwd_weight, _ = rejuv_a.assess(
C["a"].set(discard["a", IDX_WHERE_CHANGE_A]), proposal_args_backward
)
α = weight - fwd_weight + bwd_weight
return α

def compute*ratio_very_fast(key, fwd_choice, fwd_weight, model_args, trace):
request = StaticRequest({
"a": IndexRequest(jnp.array(IDX_WHERE_CHANGE_A), Update(C.v(fwd_choice["a"])))
})
*, weight, _, _ = request.edit(key, trace, model*args)
proposal_args_backward = (fwd_choice["a"],)
bwd_weight, * = rejuv_a.assess(
C["a"].set(trace.get_choices()["a", IDX_WHERE_CHANGE_A]), proposal_args_backward
)
α = weight - fwd_weight + bwd_weight
return α

def metropolis*hastings_move(key, trace, which_move):
model_args = trace.get_args()
proposal_args_forward = (trace.get_choices()["a", IDX_WHERE_CHANGE_A],)
key, subkey = jax.random.split(key)
fwd_choice, fwd_weight, * = rejuv_a.propose(subkey, proposal_args_forward)
key, subkey = jax.random.split(key)

    if which_move == 0:
        chm = trace.get_choices()
        α = compute_ratio_slow(subkey, fwd_choice, fwd_weight, model_args, chm)
    elif which_move == 1:
        α = compute_ratio_fast(subkey, fwd_choice, fwd_weight, model_args, trace)
    else:
        α = compute_ratio_very_fast(subkey, fwd_choice, fwd_weight, model_args, trace)

    old_chm = C["a"].set(trace.get_choices()["a"])
    new_chm = C["a"].set(old_chm["a"].at[IDX_WHERE_CHANGE_A].set(fwd_choice["a"]))
    key, subkey = jax.random.split(key)
    ret_chm = jax.lax.cond(
        jnp.log(jax.random.uniform(subkey)) < α, lambda: new_chm, lambda: old_chm
    )
    return ret_chm

model_sizes = [1000, 10000, 100000, 1000000, 10000000, 100000000]
slow_times = []
fast_times = []
very_fast_times = []

for model_size in model_sizes:
total_time_slow = 0
total_time_fast = 0
total_time_very_fast = 0
num_trials = 10000 if model_size <= 1000000 else 200
model_size = Const(model_size)
obs = C["obs"].set(
1.0,
)
key, subkey = jax.random.split(key)

    # create a trace from the model of the right size
    tr, _ = jax.jit(model.importance, static_argnums=(2))(subkey, obs, (model_size,))

    # warm up run to trigger jit compilation
    jitted = jax.jit(metropolis_hastings_move, static_argnums=(2))
    jitted(subkey, tr, 0)
    jitted(subkey, tr, 1)
    jitted(subkey, tr, 2)

    # measure time for each algorithm
    total_time_slow = timeit.timeit(lambda: jitted(subkey, tr, 0), number=num_trials)
    total_time_fast = timeit.timeit(lambda: jitted(subkey, tr, 1), number=num_trials)
    total_time_very_fast = timeit.timeit(
        lambda: jitted(subkey, tr, 2), number=num_trials
    )
    average_time_slow = total_time_slow / num_trials
    average_time_fast = total_time_fast / num_trials
    average_time_very_fast = total_time_very_fast / num_trials
    slow_times.append(average_time_slow)
    fast_times.append(average_time_fast)
    very_fast_times.append(average_time_very_fast)

plt.figure(figsize=(20, 5))

# First half of the values

plt.subplot(1, 2, 1)
plt.plot(
model*sizes[: len(model_sizes) // 2],
[time * 1000 for time in slow*times[: len(slow_times) // 2]],
marker="o",
label="No incremental computation",
)
plt.plot(
model_sizes[: len(model_sizes) // 2],
[time * 1000 for time in fast_times[: len(fast_times) // 2]],
marker="o",
label="Default incremental computation",
)
plt.plot(
model_sizes[: len(model_sizes) // 2],
[time \* 1000 for time in very_fast_times[: len(very_fast_times) // 2]],
marker="o",
label="Optimized incremental computation",
)
plt.xscale("log")
plt.xlabel("Argument (n)")
plt.ylabel("Average Time (milliseconds)")
plt.title("Average Execution Time of MH move for different model sizes (First Half)")
plt.grid(True)
plt.legend()

# Second half of the values

plt.subplot(1, 2, 2)
plt.plot(
model*sizes[len(model_sizes) // 2 :],
[time * 1000 for time in slow*times[len(slow_times) // 2 :]],
marker="o",
label="No incremental computation",
)
plt.plot(
model_sizes[len(model_sizes) // 2 :],
[time * 1000 for time in fast_times[len(fast_times) // 2 :]],
marker="o",
label="Default incremental computation",
)
plt.plot(
model_sizes[len(model_sizes) // 2 :],
[time \* 1000 for time in very_fast_times[len(very_fast_times) // 2 :]],
marker="o",
label="Optimized incremental computation",
)
plt.xscale("log")
plt.xlabel("Argument (n)")
plt.ylabel("Average Time (milliseconds)")
plt.title("Average Execution Time of MH move for different model sizes (Second Half)")
plt.grid(True)
plt.legend()

plt.show()

================================================
FILE: docs/cookbook/inactive/update/7_application_dirichlet_mixture_model.ipynb
================================================

# Jupyter notebook converted to Python script.

"""

# Block-Gibbs on Dirichlet Mixture Model

"""

"""
We will now see some of the key ingredients in action in a simple but more realistic setting and write a Dirichlet mixture model in GenJAX.
"""

"""

## Clustering Points on the Real Line

The goal here is to cluster datapoints on the real line. To do so, we model a fixed number of clusters, each as a 1D-Gaussian with fixed variance, and we want to infer their means.
"""

"""

### Model Description

The "model of the world" postulates:

-   A fixed number of 1D Gaussians
-   Each Gaussian is assigned a weight, representing the proportion of points assigned to each cluster
-   Each datapoint is assigned to a cluster

### Generative Process

We turn this into a generative model as follows:

-   We have a fixed prior mean and variance for where the cluster centers might be
-   We sample a mean for each cluster
-   We sample an initial weight per cluster (sum of weights is 1)
-   For each datapoint:
    -   We sample a cluster assignment proportional to the cluster weights
    -   We sample the datapoint noisily around the mean of the cluster

### Implementation Details

We, the modelers, get to choose how this process is implemented.

-   We choose distributions for each sampling step in a way that makes **inference tractable**.
-   More precisely, we choose conjugate pairs so that we can do inference via Gibbs sampling.
    -   Gibbs sampling is an MCMC method that samples an initial trace, and then updates the traced choices we want to infer over time.
    -   To update a choice, Gibbs sampling samples from a conditional distribution, which is tractable with conjugate relationships.
        """

import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import numpy as np

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import categorical, dirichlet, gen, normal, pretty
from genjax.\_src.core.pytree import Const

pretty()
key = jax.random.key(0)

"""
We define the generative model we described above. It has several hyperparameters that are somewhat manually inferred. An extension to the model could instead do inference over these hyperparameters, and fix hyper-hyperparameters instead.
"""

# Hyper parameters

PRIOR_VARIANCE = 10.0
OBS_VARIANCE = 1.0
N_DATAPOINTS = 5000
N_CLUSTERS = 40
ALPHA = float(N_DATAPOINTS / (N_CLUSTERS \* 10))
PRIOR_MEAN = 50.0
N_ITER = 50

# Debugging mode

DEBUG = True

# Sub generative functions of the bigger model

@gen
def generate_cluster(mean, var):
cluster_mean = normal(mean, var) @ "mean"
return cluster_mean

@gen
def generate_cluster_weight(alphas):
probs = dirichlet(alphas) @ "probs"
return probs

@gen
def generate_datapoint(probs, clusters):
idx = categorical(jnp.log(probs)) @ "idx"
obs = normal(clusters[idx], OBS_VARIANCE) @ "obs"
return obs

@gen
def generate_datapoints(probs, clusters, n_datapoints):
idx = categorical(jnp.log(probs), sample_shape=n_datapoints) @ "idx"
obs = normal(clusters[idx], OBS_VARIANCE) @ "obs"
return obs

# Main model

@gen
def generate_data(n_clusters: Const[int], n_datapoints: Const[int], alpha: float):
clusters = (
generate_cluster.repeat(n=n_clusters.unwrap())(PRIOR_MEAN, PRIOR_VARIANCE)
@ "clusters"
)

    probs = generate_cluster_weight.inline(
        alpha / n_clusters.unwrap() * jnp.ones(n_clusters.unwrap())
    )

    datapoints = generate_datapoints(probs, clusters, n_datapoints) @ "datapoints"

    return datapoints

"""
We create some synthetic data to test inference.
"""

# Generate synthetic data with N_CLUSTERS clusters evenly spaced

points*per_cluster = int(N_DATAPOINTS / N_CLUSTERS)
cluster_indices = jnp.arange(N_CLUSTERS)
offsets = PRIOR_VARIANCE * (-4 + 8 \_ cluster_indices / N_CLUSTERS)

# Create keys for each cluster

keys = jax.random.split(jax.random.key(0), N_CLUSTERS)

# Generate uniform random points for each cluster

uniform_points = jax.vmap(lambda k: jax.random.uniform(k, shape=(points_per_cluster,)))(
keys
)

# Add offset and prior mean to each cluster's points

shifted_points = uniform_points + (PRIOR_MEAN + offsets[:, None])

datapoints = C["datapoints", "obs"].set(shifted_points.reshape(-1))

"""
We now write the main inference loop. As we said at the beginning, we do MCMC via Gibbs sampling. Inference therefore consist of a main loop and we evolve a trace over time. The final trace contains a sample from the approximate posterior.
"""

def infer(datapoints):
key = jax.random.key(32421)
args = (Const(N*CLUSTERS), Const(N_DATAPOINTS), ALPHA)
key, subkey = jax.random.split(key)
initial_weights = C["probs"].set(jnp.ones(N_CLUSTERS) / N_CLUSTERS)
constraints = datapoints | initial_weights
tr, * = generate_data.importance(subkey, constraints, args)

    if DEBUG:
        all_posterior_means = [tr.get_choices()["clusters", "mean"]]
        all_posterior_weights = [tr.get_choices()["probs"]]
        all_cluster_assignment = [tr.get_choices()["datapoints", "idx"]]

        jax.debug.print("Initial means: {v}", v=all_posterior_means[0])
        jax.debug.print("Initial weights: {v}", v=all_posterior_weights[0])

        for _ in range(N_ITER):
            # Gibbs update on `("clusters", i, "mean")` for each i, in parallel
            key, subkey = jax.random.split(key)
            tr = jax.jit(update_cluster_means)(subkey, tr)
            all_posterior_means.append(tr.get_choices()["clusters", "mean"])

            # # Gibbs update on `("datapoints", i, "idx")` for each `i`, in parallel
            key, subkey = jax.random.split(key)
            tr = jax.jit(update_datapoint_assignment)(subkey, tr)
            all_cluster_assignment.append(tr.get_choices()["datapoints", "idx"])

            # # Gibbs update on `probs`
            key, subkey = jax.random.split(key)
            tr = jax.jit(update_cluster_weights)(subkey, tr)
            all_posterior_weights.append(tr.get_choices()["probs"])

        return all_posterior_means, all_posterior_weights, all_cluster_assignment, tr

    else:
        # One Gibbs sweep consist of updating each latent variable
        def update(carry, _):
            key, tr = carry
            # Gibbs update on `("clusters", i, "mean")` for each i, in parallel
            key, subkey = jax.random.split(key)
            tr = update_cluster_means(subkey, tr)

            # Gibbs update on `("datapoints", i, "idx")` for each `i`, in parallel
            key, subkey = jax.random.split(key)
            tr = update_datapoint_assignment(subkey, tr)

            # Gibbs update on `probs`
            key, subkey = jax.random.split(key)
            tr = update_cluster_weights(subkey, tr)
            return (key, tr), None

        # Overall inference performs a fixed number of Gibbs sweeps
        (key, tr), _ = jax.jit(jax.lax.scan)(update, (key, tr), None, length=N_ITER)
        return tr

def update_cluster_means(key, trace): # We can update each cluster in parallel # For each cluster, we find the datapoints in that cluster and compute their mean
datapoint_indexes = trace.get_choices()["datapoints", "idx"]
datapoints = trace.get_choices()["datapoints", "obs"]
n_clusters = trace.get_args()[0].unwrap()
current_means = trace.get_choices()["clusters", "mean"]

    # Count number of points per cluster
    category_counts = jnp.bincount(
        trace.get_choices()["datapoints", "idx"],
        length=n_clusters,
        minlength=n_clusters,
    )

    # Will contain some NaN due to clusters having no datapoint
    cluster_means = (
        jax.vmap(
            lambda i: jnp.sum(jnp.where(datapoint_indexes == i, datapoints, 0)),
            in_axes=(0),
            out_axes=(0),
        )(jnp.arange(n_clusters))
        / category_counts
    )

    # Conjugate update for Normal-iid-Normal distribution
    # See https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/lectures/lecture5.pdf
    # Note that there's a typo in the math for the posterior mean.
    posterior_means = (
        PRIOR_VARIANCE
        / (PRIOR_VARIANCE + OBS_VARIANCE / category_counts)
        * cluster_means
        + (OBS_VARIANCE / category_counts)
        / (PRIOR_VARIANCE + OBS_VARIANCE / category_counts)
        * PRIOR_MEAN
    )

    posterior_variances = 1 / (1 / PRIOR_VARIANCE + category_counts / OBS_VARIANCE)

    # Gibbs resampling of cluster means
    key, subkey = jax.random.split(key)
    new_means = (
        generate_cluster.vmap()
        .simulate(key, (posterior_means, posterior_variances))
        .get_choices()["mean"]
    )

    # Remove the sampled Nan due to clusters having no datapoint and pick previous mean in that case, i.e. no Gibbs update for them
    chosen_means = jnp.where(category_counts == 0, current_means, new_means)

    if DEBUG:
        jax.debug.print("Category counts: {v}", v=category_counts)
        jax.debug.print("Current means: {v}", v=cluster_means)
        jax.debug.print("Posterior means: {v}", v=posterior_means)
        jax.debug.print(fmt="Posterior variance: {v}", v=posterior_variances)
        jax.debug.print("Resampled means: {v}", v=new_means)
        jax.debug.print("Chosen means: {v}", v=chosen_means)

    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        subkey, C["clusters", "mean"].set(chosen_means), argdiffs
    )
    return new_trace

def update_datapoint_assignment(key, trace): # We want to update the index for each datapoint, in parallel. # It means we want to resample the i, but instead of being from the prior # P(i | probs), we do it from the local posterior P(i | probs, xs). # We need to do it for all addresses ["datapoints", "idx", i], # and as these are independent (when conditioned on the rest) # we can resample them in parallel.

    # Conjugate update for a categorical is just exact posterior via enumeration
    # P(x | y ) = P(x, y) \ sum_x P(x, y).
    # P(x | y1, y2) = P(x | y1)
    # Sampling from
    # (P(x = 1 | y ), P(x = 2 | y), ...) is the same as
    # sampling from Categorical(P(x = 1, y), P(x = 2, y))
    # as the weights need not be normalized
    # In addition, if the model factorizes as P(x, y1, y2) = P(x, y1)P(y1 | y2),
    # we can further simplify P(y1 | y2) from the categorical as it does not depend on x. More generally We only need to look at the children and parents of x ("idx" in our situation, which are conveniently wrapped in the generate_datapoint generative function).

    def compute_local_density(x, i):
        datapoint_mean = trace.get_choices()["datapoints", "obs", x]
        chm = C["obs"].set(datapoint_mean).at["idx"].set(i)
        clusters = trace.get_choices()["clusters", "mean"]
        probs = trace.get_choices()["probs"]
        args = (probs, clusters)
        model_logpdf, _ = generate_datapoint.assess(chm, args)
        return model_logpdf

    n_clusters = trace.get_args()[0].unwrap()
    n_datapoints = trace.get_args()[1].unwrap()
    local_densities = jax.vmap(
        lambda x: jax.vmap(lambda i: compute_local_density(x, i))(
            jnp.arange(n_clusters)
        )
    )(jnp.arange(n_datapoints))

    # Conjugate update by sampling from posterior categorical
    # Note: I think we could've used something like
    # generate_datapoint.vmap().importance which would perhaps
    # work in a more general setting but would definitely be slower here.
    key, subkey = jax.random.split(key)
    new_datapoint_indexes = genjax.categorical.simulate(
        key, (local_densities,)
    ).get_choices()
    # Gibbs resampling of datapoint assignment to clusters
    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(
        subkey, C["datapoints", "idx"].set(new_datapoint_indexes), argdiffs
    )
    return new_trace

def update_cluster_weights(key, trace): # Count number of points per cluster
n_clusters = trace.get_args()[0].unwrap()
category_counts = jnp.bincount(
trace.get_choices()["datapoints", "idx"],
length=n_clusters,
minlength=n_clusters,
)

    # Conjugate update for Dirichlet distribution
    # See https://en.wikipedia.org/wiki/Dirichlet_distribution#Conjugate_to_categorical_or_multinomial
    new_alpha = ALPHA / n_clusters * jnp.ones(n_clusters) + category_counts

    # Gibbs resampling of cluster weights
    key, subkey = jax.random.split(key)
    new_probs = generate_cluster_weight.simulate(key, (new_alpha,)).get_retval()

    if DEBUG:
        jax.debug.print(fmt="Category counts: {v}", v=category_counts)
        jax.debug.print(fmt="New alpha: {v}", v=new_alpha)
        jax.debug.print(fmt="New probs: {v}", v=new_probs)
    argdiffs = genjax.Diff.no_change(trace.args)
    new_trace, _, _, _ = trace.update(subkey, C["probs"].set(new_probs), argdiffs)
    return new_trace

"""
We can now run inference, obtaining the final trace and some intermediate traces for visualizing inference.
"""

if DEBUG:
(
all_posterior_means,
all_posterior_weights,
all_cluster_assignment,
posterior_trace,
) = infer(datapoints)
else:
posterior_trace = infer(datapoints)

posterior_trace

"""
Plotting results
"""

# Prepare data for the animation

data_points = datapoints["datapoints", "obs"].tolist()
np.random.seed(42)
jitter = np.random.uniform(-0.05, 0.05, size=len(data_points)).tolist()
std_dev = np.sqrt(OBS_VARIANCE) \* 1.5
all_cluster_assignments_list = [a.tolist() for a in all_cluster_assignment]
all_posterior_means_list = [m.tolist() for m in all_posterior_means]
all_posterior_weights_list = [w.tolist() for w in all_posterior_weights]

# Define a consistent color palette to use throughout the visualization

color_palette = """
const plotColors = [
"#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b",
"#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac",
"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
"#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
];
"""

# Shared data initialization for all plot components

frame_data_js = (
"""
// Get current frame data
const frame = $state.frame;
const hoveredCluster = $state.hoveredCluster;
const means = """ + str(all_posterior_means_list) + """[frame];
const weights = """ + str(all_posterior_weights_list) + """[frame];
const assignments = """ + str(all_cluster_assignments_list) + """[frame];
const stdDev = """ + str(std_dev) + """;
"""
)

# Create a visualizer with animation

(
Plot.initialState({"frame": 0, "hoveredCluster": None})
| # Main visualization that updates based on the current frame
Plot.plot({
"marks": [ # 1. Data points with jitter - show all data points with optional highlighting
Plot.dot(
Plot.js(
"""function() {
""" + frame_data_js + """
const dataPoints = """ + str(data_points) + """;
const jitter = """ + str(jitter) + """;

                    """
                    + color_palette
                    + """

                    // Return all points with hover-aware opacity
                    return dataPoints.map((x, i) => {
                        const clusterIdx = assignments[i];
                        // If a cluster is hovered, reduce opacity of other clusters' points
                        const isHovered = hoveredCluster !== null && clusterIdx === hoveredCluster;
                        const opacity = hoveredCluster === null ? 0.5 : (isHovered ? 0.7 : 0.15);
                        return {
                            x: x,
                            y: jitter[i],
                            color: plotColors[clusterIdx % 20],
                            opacity: opacity
                        };
                    });
                }()"""
                ),
                {"x": "x", "y": "y", "fill": "color", "r": 3, "opacity": "opacity"},
            ),
            # 2. Combined error bars (both horizontal lines and vertical caps)
            Plot.line(
                Plot.js(
                    """function() {
                    """
                    + frame_data_js
                    + """
                    const capSize = 0.04;  // Size of the vertical cap lines

                    """
                    + color_palette
                    + """

                    // We'll collect all line segments in a flat array
                    const result = [];

                    for (let i = 0; i < means.length; i++) {
                        // Only include error bars for clusters with weight >= 0.01
                        if (weights[i] >= 0.01) {
                            // Determine if this cluster is being hovered
                            const isHovered = hoveredCluster === i;
                            const opacity = hoveredCluster === null ? 0.7 : (isHovered ? 1.0 : 0.3);
                            const strokeWidth = isHovered ? 4 : 3;
                            const color = plotColors[i % 20];

                            // Add horizontal line (error bar itself)
                            result.push({x: means[i] - stdDev, y: 0, cluster: i, color, opacity, width: strokeWidth});
                            result.push({x: means[i] + stdDev, y: 0, cluster: i, color, opacity, width: strokeWidth});

                            // Add left cap (vertical line)
                            result.push({x: means[i] - stdDev, y: -capSize, cluster: i, color, opacity, width: strokeWidth});
                            result.push({x: means[i] - stdDev, y: capSize, cluster: i, color, opacity, width: strokeWidth});

                            // Add right cap (vertical line)
                            result.push({x: means[i] + stdDev, y: -capSize, cluster: i, color, opacity, width: strokeWidth});
                            result.push({x: means[i] + stdDev, y: capSize, cluster: i, color, opacity, width: strokeWidth});
                        }
                    }
                    return result;
                }()"""
                ),
                {
                    "x": "x",
                    "y": "y",
                    "stroke": "color",
                    "strokeWidth": "width",
                    "opacity": "opacity",
                    "z": "cluster",
                },
            ),
            # 3. Cluster means as stars
            Plot.dot(
                Plot.js(
                    """function() {
                    """
                    + frame_data_js
                    + """
                    """
                    + color_palette
                    + """

                    // Create a simple array for each cluster mean
                    return means.map((mean, i) => {
                        // Only include means for clusters with sufficient weight
                        if (weights[i] >= 0.01) {
                            const isHovered = hoveredCluster === i;
                            return {
                                x: mean,
                                y: 0,
                                cluster: i,
                                color: plotColors[i % 20],
                                opacity: isHovered ? 1.0 : 0.8
                            };
                        }
                        return null;  // Skip low-weight clusters
                    }).filter(d => d !== null);  // Remove null values
                }()"""
                ),
                {
                    "x": "x",
                    "y": "y",
                    "fill": "color",
                    "r": 10,
                    "symbol": "star",
                    "stroke": "black",
                    "strokeWidth": 2,
                    "opacity": "opacity",
                },
            ),
        ],
        "grid": True,
        "marginTop": 40,
        "marginRight": 40,
        "marginBottom": 40,
        "marginLeft": 40,
        "style": {"height": "400px"},
        "title": Plot.js(
            "`Dirichlet Mixture Model - Iteration ${$state.frame} of "
            + str(len(all_posterior_means) - 1)
            + "`"
        ),
        "subtitle": "Cluster centers (★) with standard deviation (—) and data points (•)",
    })
    |
    # Animation controls and legend with hover effects
    Plot.html([
        "div",
        {"className": "p-4"},
        [
            "div",
            {"className": "mb-4"},
            Plot.Slider(
                "frame",
                init=0,
                range=[0, len(all_posterior_means) - 1],
                step=1,
                label="Iteration",
                width="100%",
                fps=8,
            ),
        ],
        [
            "div",
            {"className": "mt-4"},
            Plot.js(
                """function() {
                """
                + frame_data_js
                + """
                // Count assignments in current frame
                const counts = {};
                assignments.forEach(a => { counts[a] = (counts[a] || 0) + 1; });

                """
                + color_palette
                + """

                // Sort clusters by weight, filter by minimum weight, and limit to top 10
                const topClusters = Object.keys(weights)
                    .map(i => ({
                        id: parseInt(i),
                        weight: weights[i],
                        count: counts[parseInt(i)] || 0
                    }))
                    .filter(c => c.weight >= 0.01)
                    .sort((a, b) => b.weight - a.weight)
                    .slice(0, 10);

                // Create placeholder rows for consistent height
                const placeholders = Array(Math.max(0, 10 - topClusters.length))
                    .fill(0)
                    .map(() => ["tr", {"className": "h-8"}, ["td", {"colSpan": 3}, ""]]);

                return [
                    "div", {},
                    ["h3", {}, `Top Clusters by Weight (Iteration ${frame})`],
                    ["div", {"style": {"height": "280px", "overflow": "auto"}},
                        ["table", {"className": "w-full mt-2"},
                            ["thead", ["tr",
                                ["th", {"className": "text-left"}, "Cluster"],
                                ["th", {"className": "text-left"}, "Weight"],
                                ["th", {"className": "text-left"}, "Points"]
                            ]],
                            ["tbody",
                                ...topClusters.map(cluster =>
                                    ["tr", {
                                        "className": "h-8",
                                        "style": {
                                            "cursor": "pointer",
                                            "backgroundColor": $state.hoveredCluster === cluster.id ? "#f0f0f0" : "transparent"
                                        },
                                        "onMouseEnter": () => { $state.hoveredCluster = cluster.id; },
                                        "onMouseLeave": () => { $state.hoveredCluster = null; }
                                    },
                                    ["td", {"className": "py-1"},
                                        ["div", {"className": "flex items-center"},
                                            ["div", {
                                                "style": {
                                                    "backgroundColor": plotColors[cluster.id % 20],
                                                    "width": "24px",
                                                    "height": "24px",
                                                    "borderRadius": "4px",
                                                    "border": "1px solid rgba(0,0,0,0.2)",
                                                    "display": "inline-block",
                                                    "marginRight": "8px"
                                                }
                                            }],
                                            `Cluster ${cluster.id}`
                                        ]
                                    ],
                                    ["td", {"className": "py-1"}, cluster.weight.toFixed(4)],
                                    ["td", {"className": "py-1"}, cluster.count]
                                    ]
                                ),
                                ...placeholders
                            ]
                        ]
                    ]
                ];
            }()"""
            ),
        ],
    ])

)

"""
For the interested reader, here's some exercises to try out to make this model better:

1. Extend the model to infer the variance of the clusters by putting an inverse_gamma prior replacing the `OBS_VARIANCE` hyperparameter and doing block-Gibbs on it using the normal-inverse-gamma conjugacy
2. Try a better initialization of the datapoint assignment: pick a point a use something like k-means and assign all the surrounding points to the same initial cluster. Iterate on all the points until they all have some initial cluster.
3. Improve inference using SMC via data annealing: subssample 1/100 of the data and run inference on this, then run inference again on 1/10 of the data starting with the inferred choices for cluster means and weights from the previous trace, and finally repeat for the whole data.

Note that the model is still expected to get stuck in local minima (the clustering at the borders isn't great), and one way to improve upon it would be to use a split-merge move, via reversible-jump MCMC.
"""

================================================
FILE: docs/css/custom.css
================================================
@font-face {
font-family: 'Cal Sans';
src: url('../assets/font/CalSans-SemiBold.woff2') format('woff2');
font-style: normal;
}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');

@font-face {
font-family: 'Berkeley Mono';
src: url('../assets/font/BerkeleyMonoVariable-Regular.woff2') format('woff2');
font-weight: 100 900;
font-style: normal;
}

@font-face {
font-family: 'Berkeley Mono';
src: url('../assets/font/BerkeleyMonoVariable-Italic.woff2') format('woff2');
font-weight: 100 900;
font-style: italic;
}

.md-typeset h1,
.md-typeset h2,
.md-typeset h3,
.md-typeset h4,
.md-typeset h5,
.md-typeset h6 {
font-weight: bold;
font-family: "Cal Sans";
}

.md-typeset {
font-family: Inter, Roboto, 'Helvetica Neue', 'Arial Nova', 'Nimbus Sans', Arial, sans-serif;
font-weight: normal;
}

.md-typeset code {
font-family: 'Berkeley Mono', ui-monospace, 'Cascadia Code',
'Source Code Pro', Menlo, Consolas, 'DejaVu Sans Mono', monospace;
font-weight: 600;
}

.md-footer,
.md-source {
font-weight: 600;
}

.md-content {
margin-left: auto;
margin-right: auto;
max-width: 800px;
}

div.md-header\_\_ellipsis {
font-weight: bold;
}

treescope-container::part(treescope_root) {
font-family: "Berkeley Mono";
font-size: 13px;
word-wrap: break-word;
width: fit-content;
display: block;
}

================================================
FILE: docs/css/mkdocstrings.css
================================================
/_ Indentation. _/
div.doc-contents {
padding-left: 25px;
border-left: 0.1rem solid var(--md-typeset-table-color);
}

/_ Mark external links as such. _/
a.autorefs-external::after {
/_ https://primer.style/octicons/arrow-up-right-24 _/
background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="rgb(0, 0, 0)" d="M18.25 15.5a.75.75 0 00.75-.75v-9a.75.75 0 00-.75-.75h-9a.75.75 0 000 1.5h7.19L6.22 16.72a.75.75 0 101.06 1.06L17.5 7.56v7.19c0 .414.336.75.75.75z"></path></svg>');
content: ' ';

display: inline-block;
position: relative;
top: 0.1em;
margin-left: 0.2em;
margin-right: 0.1em;

height: 1em;
width: 1em;
border-radius: 100%;
background-color: var(--md-typeset-a-color);
}

a.autorefs-external:hover::after {
background-color: var(--md-accent-fg-color);
}

span.doc.doc-object-name {
font-family: "Berkeley Mono", monospace;
}

div.result {
font-size: 0.85em;
font-family: "Berkeley Mono", monospace;
}

================================================
FILE: docs/js/mathjax.js
================================================
window.MathJax = {
tex: {
inlineMath: [['$','$'], ["\\(", "\\)"]],
displayMath: [['$$','$$'], ["\\[", "\\]"]],
processEscapes: true,
processEnvironments: true
},
};

document$.subscribe(() => {
MathJax.startup.output.clearCache()
MathJax.typesetClear()
MathJax.texReset()
MathJax.typesetPromise()
})

================================================
FILE: docs/library/combinators.md
================================================

# Combinators: structured patterns of composition

While the programmatic [`genjax.StaticGenerativeFunction`][] language is powerful, its restrictions can be limiting. Combinators are a way to express common patterns of composition in a more concise way, and to gain access to effects which are common in JAX (like `jax.vmap`) for generative computations.

Each of the combinators below is implemented as a method on [`genjax.GenerativeFunction`][] and as a standalone decorator.

You should strongly prefer the method form. Here's an example of the `vmap` combinator created by the [`genjax.GenerativeFunction.vmap`][] method:

Here is the `vmap` combinator used as a method. `square_many` below accepts an array and returns an array:

```python exec="yes" html="true" source="material-block" session="combinators"
import jax, genjax

@genjax.gen
def square(x):
    return x * x

square_many = square.vmap()
```

Here is `square_many` defined with [`genjax.vmap`][], the decorator version of the `vmap` method:

```python exec="yes" html="true" source="material-block" session="combinators"
@genjax.vmap()
@genjax.gen
def square_many_decorator(x):
    return x * x
```

!!! warning
We do _not_ recommend this style, since the original building block generative function won't be available by itself. Please prefer using the combinator methods, or the transformation style shown below.

If you insist on using the decorator form, you can preserve the original function like this:

```python exec="yes" html="true" source="material-block" session="combinators"
@genjax.gen
def square(x):
    return x * x

# Use the decorator as a transformation:
square_many_better = genjax.vmap()(square)
```

## `vmap`-like Combinators

::: genjax.vmap
::: genjax.repeat

## `scan`-like Combinators

::: genjax.scan
::: genjax.accumulate
::: genjax.reduce
::: genjax.iterate
::: genjax.iterate_final
::: genjax.masked_iterate
::: genjax.masked_iterate_final

## Control Flow Combinators

::: genjax.or_else
::: genjax.switch

## Argument and Return Transformations

::: genjax.dimap
::: genjax.map
::: genjax.contramap

## The Rest

::: genjax.mask
::: genjax.mix

================================================
FILE: docs/library/core.md
================================================

# Journey to the center of `genjax.core`

This page describes the set of core concepts and datatypes in GenJAX, including Gen's generative datatypes and concepts ([`GenerativeFunction`][genjax.core.GenerativeFunction], [`Trace`][genjax.core.Trace], [`ChoiceMap`][genjax.core.ChoiceMap], and [`EditRequest`][genjax.core.EditRequest]), the core JAX compatibility datatypes ([`Pytree`][genjax.core.Pytree], [`Const`][genjax.core.Const], and [`Closure`][genjax.core.Closure]), as well as functionally inspired `Pytree` extensions ([`Mask`][genjax.core.Mask]), and GenJAX's approach to "static" (JAX tracing time) typechecking.

::: genjax.core.GenerativeFunction

Traces are data structures which record (execution and inference) data about the invocation of generative functions. Traces are often specialized to a generative function language, to take advantage of data locality, and other representation optimizations. Traces support a _trace interface_: a set of accessor methods designed to provide convenient manipulation when handling traces in inference algorithms. We document this interface below for the `Trace` data type.

::: genjax.core.Trace
::: genjax.core.EditRequest

## Generative functions with addressed random choices

Generative functions will often include _addressed_ random choices. These are random choices which are given a name via an addressing syntax, and can be accessed by name via extended interfaces on the `ChoiceMap` type which supports the addressing.

::: genjax.core.ChoiceMap
::: genjax.core.Selection

## JAX compatible data via `Pytree`

JAX natively works with arrays, and with instances of Python classes which can be broken down into lists of arrays. JAX's [`Pytree`](https://jax.readthedocs.io/en/latest/pytrees.html) system provides a way to register a class with methods that can break instances of the class down into a list of arrays (canonically referred to as _flattening_), and build an instance back up given a list of arrays (canonically referred to as _unflattening_).

GenJAX provides an abstract class called `Pytree` which automates the implementation of the `flatten` / `unflatten` methods for a class. GenJAX's `Pytree` inherits from [`penzai.Struct`](https://penzai.readthedocs.io/en/stable/_autosummary/leaf/penzai.core.struct.Struct.html), to support pretty printing, and some convenient methods to annotate what data should be part of the `Pytree` _type_ (static fields, won't be broken down into a JAX array) and what data should be considered dynamic.

::: genjax.core.Pytree
options:
members: - dataclass - static - field

::: genjax.core.Const

::: genjax.core.Closure

## Dynamism in JAX: masks and sum types

The semantics of Gen are defined independently of any particular computational substrate or implementation - but JAX (and XLA through JAX) is a unique substrate, offering high performance, the ability to transformation code ahead-of-time via program transformations, and ... _a rather unique set of restrictions_.

### JAX is a two-phase system

While not yet formally modelled, it's appropriate to think of JAX as separating computation into two phases:

-   The _statics_ phase (which occurs at JAX tracing / transformation time).
-   The _runtime_ phase (which occurs when a computation written in JAX is actually deployed via XLA and executed on a physical device somewhere in the world).

JAX has different rules for handling values depending on which phase we are in.

For instance, JAX disallows usage of runtime values to resolve Python control flow at tracing time (intuition: we don't actually know the value yet!) and will error if the user attempts to trace through a Python program with incorrect usage of runtime values.

In GenJAX, we take advantage of JAX's tracing to construct code which, when traced, produces specialized code _depending on static information_. At the same time, we are careful to encode Gen's interfaces to respect JAX's rules which govern how static / runtime values can be used.

The most primitive way to encode _runtime uncertainty_ about a piece of data is to attach a `bool` to it, which indicates whether the data is "on" or "off".

GenJAX contains a system for tagging data with flags, to indicate if the data is valid or invalid during inference interface computations _at runtime_. The key data structure which supports this system is `genjax.core.Mask`.

::: genjax.core.Mask
options:
show_root_heading: true
members: - unmask - match

## Static typing with `genjax.typing` a.k.a 🐻`beartype`🐻

GenJAX uses [`beartype`](https://github.com/beartype/beartype) to perform type checking _during JAX tracing / compile time_. This means that `beartype`, normally a fast _runtime_ type checker, operates _at JAX tracing time_ to ensure that the arguments and return values are correct, with zero runtime cost.

### Generative interface types

::: genjax.core.Arguments
::: genjax.core.Score
::: genjax.core.Weight
::: genjax.core.Retdiff
::: genjax.core.Argdiffs

================================================
FILE: docs/library/generative_functions.md
================================================

# The menagerie of `GenerativeFunction`

Generative functions are probabilistic building blocks. They allow you to express complex probability distributions, and automate several operations on them. GenJAX exports a standard library of generative functions, and this page catalogues them and their usage.

## The venerable & reliable `Distribution`

To start, distributions are generative functions.

::: genjax.Distribution
options:
show_root_heading: true
members: - random_weighted - estimate_logpdf

Distributions intentionally expose a permissive interface ([`random_weighted`](generative_functions.md#genjax.Distribution.random_weighted) and [`estimate_logpdf`](generative_functions.md#genjax.Distribution.estimate_logpdf) which doesn't assume _exact_ density evaluation. [`genjax.ExactDensity`](generative_functions.md#genjax.ExactDensity) is a more restrictive interface, which assumes exact density evaluation.

::: genjax.ExactDensity
options:
show_root_heading: true
members: - random_weighted - estimate_logpdf

GenJAX exports a long list of exact density distributions, which uses the functionality of [`tfp.distributions`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions). A list of these is shown below.

::: genjax.generative_functions.distributions
options:
show_root_heading: true
summary:
attributes: true

## `StaticGenerativeFunction`: a programmatic language

For any serious work, you'll want a way to combine generative functions together, mixing deterministic functions with sampling. `StaticGenerativeFunction` is a way to do that: it supports the use of a JAX compatible subset of Python to author generative functions. It also supports the ability _to invoke_ other generative functions: instances of this type (and any other type of generative function) can then be used in larger generative programs.

::: genjax.StaticGenerativeFunction
options:
show_root_heading: true
members: - source - simulate - assess - update

================================================
FILE: docs/library/inference.md
================================================

# Inference

Conditioning probability distributions is a commonly desired operation, allowing users to express Bayesian inference problems. Conditioning is also a subroutine in other desired operations, like marginalization.

## The language of inference

In GenJAX, inference problems are specified by constructing [`Target`][genjax.inference.Target] distributions. Their solutions are approximated using [`Algorithm`][genjax.inference.Algorithm] families.

::: genjax.inference.Target
options:
show_root_heading: true

Algorithms inherit from a class called [`SampleDistribution`][genjax.inference.SampleDistribution] - these are objects which implement the _stochastic probability interface_ [[Lew23](https://dl.acm.org/doi/abs/10.1145/3591290)], meaning they expose methods to produce samples and samples from _density estimators_ for density computations.

::: genjax.inference.SampleDistribution
options:
show_root_heading: true
members: - random_weighted - estimate_logpdf

`Algorithm` families implement the stochastic probability interface. Their [`Distribution`][genjax.Distribution] methods accept `Target` instances, and produce samples and density estimates for approximate posteriors.

::: genjax.inference.Algorithm
options:
show_root_heading: true
members: - random_weighted - estimate_logpdf

By virtue of the _stochastic probability interface_, GenJAX also exposes _marginalization_ as a first class concept.

::: genjax.inference.Marginal
options:
show_root_heading: true
members: - random_weighted - estimate_logpdf

## The SMC inference library

Sequential Monte Carlo (SMC) is a popular algorithm for performing approximate inference in probabilistic models.

::: genjax.inference.smc.SMCAlgorithm
options:
show_root_heading: true

::: genjax.inference.smc.Importance
options:
show_root_heading: true

::: genjax.inference.smc.ImportanceK
options:
show_root_heading: true

## The VI inference library

Variational inference is an approach to inference which involves solving optimization problems over spaces of distributions. For a posterior inference problem, the goal is to find the distribution in some parametrized family of distributions (often called _the guide family_) which is close to the posterior under some notion of distance.

Variational inference problems typically involve optimization functions which are defined as _expectations_, and these expectations and their analytic gradients are often intractable to compute. Therefore, unbiased gradient estimators are used to approximate the true gradients.

The `genjax.vi` inference module provides automation for constructing variational losses, and deriving gradient estimators. The architecture is shown below.

<figure markdown="span">
  ![GenJAX VI architecture](../assets/img/genjax-vi.png){ width = "300" }
  <figcaption><b>Fig. 1</b>: How variational inference works in GenJAX.</figcaption>
</figure>

::: genjax.inference.vi.adev_distribution
options:
show_root_heading: true

::: genjax.inference.vi.ELBO
options:
show_root_heading: true

::: genjax.inference.vi.IWELBO
options:
show_root_heading: true

::: genjax.inference.vi.PWake
options:
show_root_heading: true

::: genjax.inference.vi.QWake
options:
show_root_heading: true

================================================
FILE: tests/adev/test_adev.py
================================================

# Copyright 2024 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax
import pytest

from genjax.adev import Dual, add_cost, baseline, expectation, flip_enum, flip_reinforce

class TestADEVFlipCond:
def test*flip_cond_exact_forward_mode_correctness(self):
@expectation
def flip_exact_loss(p):
b = flip_enum(p)
return jax.lax.cond(
b,
lambda *: 0.0,
lambda p: -p / 2.0,
p,
)

        key = jax.random.key(314159)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            p_dual = jax.jit(flip_exact_loss.jvp_estimate)(key, Dual(p, 1.0))
            assert p_dual.tangent == pytest.approx(p - 0.5, rel=0.0001)

    def test_flip_cond_exact_reverse_mode_correctness(self):
        @expectation
        def flip_exact_loss(p):
            b = flip_enum(p)
            return jax.lax.cond(
                b,
                lambda _: 0.0,
                lambda p: -p / 2.0,
                p,
            )

        key = jax.random.key(314159)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            (p_grad,) = jax.jit(flip_exact_loss.grad_estimate)(key, (p,))
            assert p_grad == pytest.approx(p - 0.5, rel=0.0001)

    def test_flip_cond_smoke_test_symbolic_zeros(self):
        @expectation
        def flip_exact_loss(p):
            b = flip_enum(0.3)
            return jax.lax.cond(
                b,
                lambda _: 0.0,
                lambda p: -p / 2.0,
                p,
            )

        key = jax.random.key(314159)
        _ = jax.jit(flip_exact_loss.jvp_estimate)(key, Dual(0.1, 1.0))

    def test_add_cost(self):
        @expectation
        def flip_exact_loss(p):
            add_cost(p**2)
            return 0.0

        key = jax.random.key(314159)
        _ = jax.jit(flip_exact_loss.jvp_estimate)(key, Dual(0.1, 1.0))

class TestBaselineFlip:
def test_baseline_flip(self):
@expectation
def flip_reinforce_loss_no_baseline(p):
b = flip_reinforce(p)
v = jax.lax.cond(b, lambda: -1.0, lambda: 1.0)
return v

        @expectation
        def flip_reinforce_loss(p):
            b = baseline(flip_reinforce)(10.0, p)
            v = jax.lax.cond(b, lambda: -1.0, lambda: 1.0)
            return v + 10.0

        key = jax.random.key(314159)
        p_dual_no_baseline = jax.jit(flip_reinforce_loss_no_baseline.jvp_estimate)(
            key, Dual(0.1, 1.0)
        )

        p_dual = jax.jit(flip_reinforce_loss.jvp_estimate)(key, Dual(0.1, 1.0))

        assert p_dual.tangent == pytest.approx(p_dual_no_baseline.tangent, 1e-3)

================================================
FILE: tests/core/test_diff.py
================================================

# Copyright 2023 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

class TestDiff:
pass

================================================
FILE: tests/core/test_pytree.py
================================================
import jax.numpy as jnp

import genjax
from genjax.typing import FloatArray

class TestPytree:
def test_unwrap(self):
c = genjax.Pytree.const(5)
assert c.unwrap() == 5
assert genjax.Const.unwrap(10) == 10

class TestPythonic:
def test_pythonic(self):
@genjax.Pytree.dataclass
class Foo(genjax.PythonicPytree):
x: FloatArray
y: FloatArray

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])

        f = Foo(x, y)

        assert f[1] == Foo(x[1], y[1])
        assert f[jnp.array(1, dtype=int)] == Foo(x[1], y[1])

        assert jnp.all(f[:2].x == x[:2])
        assert jnp.all(f[:2].y == y[:2])

        assert jnp.all(f[2:].x == x[2:])
        assert jnp.all(f[2:].y == y[2:])

        assert jnp.all(f[1::4].x == x[1::4])
        assert jnp.all(f[1::4].y == y[1::4])

        assert len(f) == x.shape[0]

        fi = iter(f)
        assert next(fi) == Foo(x[0], y[0])
        assert next(fi) == Foo(x[1], y[1])

        ff = f + f

        assert len(ff) == 2 * len(f)
        assert jnp.allclose(ff.x, jnp.concatenate((x, x)))

        p = Foo(jnp.array(-1.0), jnp.array(-10.0))
        fp = f.prepend(p)

        assert len(fp) == 1 + len(f)
        assert jnp.all(fp[0].x == p.x)
        assert jnp.all(fp[0].y == p.y)
        assert fp[0] == p

================================================
FILE: tests/core/test_staging.py
================================================

# Copyright 2024 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax.numpy as jnp

from genjax.\_src.core.compiler.staging import FlagOp, multi_switch, tree_choose

class TestFlag:
def test*basic_operation(self):
true_flags = [
True,
jnp.array(True),
jnp.array([True, True]),
]
false_flags = [
False,
jnp.array(False),
jnp.array([False, False]),
]
for t in true_flags:
assert jnp.all(t)
assert not jnp.all(FlagOp.not*(t))
for f in false*flags:
assert not jnp.all(f)
assert not jnp.all(FlagOp.and*(t, f))
assert jnp.all(FlagOp.or*(t, f))
assert jnp.all(FlagOp.xor*(t, f))
for u in true*flags:
assert jnp.all(FlagOp.and*(t, u))
assert jnp.all(FlagOp.or*(t, u))
assert not jnp.all(FlagOp.xor*(t, u))
for f1 in false*flags:
for f2 in false_flags:
assert not jnp.all(FlagOp.xor*(f1, f2))

    def test_where(self):
        assert FlagOp.where(True, 3.0, 4.0) == 3
        assert FlagOp.where(False, 3.0, 4.0) == 4
        assert FlagOp.where(jnp.array(True), 3.0, 4.0) == 3
        assert FlagOp.where(jnp.array(False), 3.0, 4.0) == 4

class TestTreeChoose:
def test_static_integer_index(self):
result = tree_choose(1, [10, 20, 30])
assert result == 20

    def test_jax_array_index(self):
        """
        Test that tree_choose works correctly with JAX array indices.
        This test ensures that when given a JAX array as an index,
        the function selects the correct value from the list.
        """
        result = tree_choose(jnp.array(2), [10, 20, 30])
        assert jnp.array_equal(result, jnp.array(30))

    def test_heterogeneous_types(self):
        """
        Test that tree_choose correctly handles heterogeneous types.
        It should attempt to cast compatible types (like bool to int)
        and use the dtype of the result for consistency.
        """
        result = tree_choose(2, [True, 2, False])
        assert result == 0
        assert jnp.asarray(result).dtype == jnp.int32

    def test_wrap_mode(self):
        """
        Test that tree_choose wraps around when the index is out of bounds.
        This should work for both jnp.array indices and concrete integer indices.
        """
        # first, the jnp.array index case:
        result = tree_choose(jnp.array(3), [10, 20, 30])
        assert jnp.array_equal(result, jnp.array(10))

        # then the concrete index case:
        concrete_result = tree_choose(3, [10, 20, 30])
        assert jnp.array_equal(result, concrete_result)

class TestMultiSwitch:
def test_multi_switch(self):
def branch_0(x):
return {"result": x + 1, "extra": True}

        def branch_1(x, y):
            return {"result": x * y, "extra": [x, y]}

        def branch_2(x, y, z):
            return {
                "result": x + y + z,
                "extra": {"sum": x + y + z, "product": x * y * z},
            }

        branches = [branch_0, branch_1, branch_2]
        arg_tuples = [(5,), (3, 4), (1, 2, 3)]

        # Test with static index — the return value is the list of all possible shapes with only the selected one filled in.
        assert multi_switch(0, branches, arg_tuples) == [
            {"extra": True, "result": jnp.array(6, dtype=jnp.int32)},
            {
                "extra": [jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)],
                "result": jnp.array(0, dtype=jnp.int32),
            },
            {
                "extra": {
                    "product": jnp.array(0, dtype=jnp.int32),
                    "sum": jnp.array(0, dtype=jnp.int32),
                },
                "result": jnp.array(0, dtype=jnp.int32),
            },
        ]

        assert multi_switch(1, branches, arg_tuples) == [
            {"extra": False, "result": jnp.array(0, dtype=jnp.int32)},
            {
                "extra": [jnp.array(3, dtype=jnp.int32), jnp.array(4, dtype=jnp.int32)],
                "result": jnp.array(12, dtype=jnp.int32),
            },
            {
                "extra": {
                    "product": jnp.array(0, dtype=jnp.int32),
                    "sum": jnp.array(0, dtype=jnp.int32),
                },
                "result": jnp.array(0, dtype=jnp.int32),
            },
        ]

        assert multi_switch(2, branches, arg_tuples) == [
            {"extra": False, "result": jnp.array(0, dtype=jnp.int32)},
            {
                "extra": [jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32)],
                "result": jnp.array(0, dtype=jnp.int32),
            },
            {
                "extra": {
                    "product": jnp.array(6, dtype=jnp.int32),
                    "sum": jnp.array(6, dtype=jnp.int32),
                },
                "result": jnp.array(6, dtype=jnp.int32),
            },
        ]

        # Test with dynamic index
        dynamic_index = jnp.array(1)
        assert multi_switch(dynamic_index, branches, arg_tuples) == multi_switch(
            1, branches, arg_tuples
        )

        # Test with out of bounds index (should clamp)
        assert multi_switch(10, branches, arg_tuples) == multi_switch(
            2, branches, arg_tuples
        )

================================================
FILE: tests/core/generative/test_core.py
================================================

# Copyright 2023 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax
import jax.numpy as jnp
import pytest

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Selection
from genjax import SelectionBuilder as S
from genjax.\_src.generative_functions.static import StaticTrace

class TestTupleAddr:
def test_tupled_address(self):
@genjax.gen
def f():
x = genjax.normal(0.0, 1.0) @ ("x", "x0")
y = genjax.normal(x, 1.0) @ "y"
return y

        tr = f.simulate(jax.random.key(0), ())
        chm = tr.get_choices()
        x_score, _ = genjax.normal.assess(C.v(chm["x", "x0"]), (0.0, 1.0))
        assert x_score == tr.project(jax.random.key(1), Selection.at["x", "x0"])

    @pytest.mark.skip(reason="this check is not yet implemented")
    def test_tupled_address_conflict(self):
        @genjax.gen
        def submodel():
            return genjax.normal(0.0, 1.0) @ "y"

        @genjax.gen
        def model():
            _ = genjax.normal(0.0, 1.0) @ ("x", "y")
            return submodel() @ "x"

        with pytest.raises(Exception):
            tr = model.simulate(jax.random.key(0), ())
            tr.get_choices()

class TestProject:
def test_project(self):
@genjax.gen
def f():
x = genjax.normal(0.0, 1.0) @ "x"
y = genjax.normal(0.0, 1.0) @ "y"
return x, y

        # get a trace
        tr = f.simulate(jax.random.key(0), ())
        # evaluations
        x_score = tr.project(jax.random.key(1), S["x"])
        with pytest.deprecated_call():
            assert x_score == tr.get_subtrace(("x",)).get_score()
        assert x_score == tr.get_subtrace("x").get_score()

        y_score = tr.project(jax.random.key(1), S["y"])
        with pytest.deprecated_call():
            assert y_score == tr.get_subtrace(("y",)).get_score()
        assert y_score == tr.get_subtrace("y").get_score()

        assert tr.get_score() == x_score + y_score

class TestGetSubtrace:
def test_get_subtrace(self):
@genjax.gen
def f():
x = genjax.normal(0.0, 1.0) @ "x"
y = genjax.normal(0.0, 1.0) @ "y"
return x, y

        @genjax.gen
        def g():
            x, y = f() @ "f"
            return x + y

        @genjax.gen
        def h():
            return g() @ "g"

        tr = g.simulate(jax.random.key(1), ())
        f_tr = tr.get_subtrace("f")
        assert isinstance(f_tr, StaticTrace)
        assert (
            tr.get_subtrace("f", "x").get_score() == f_tr.get_subtrace("x").get_score()
        )
        assert (
            tr.get_subtrace("f", "y").get_score() == f_tr.get_subtrace("y").get_score()
        )

        tr = h.simulate(jax.random.key(2), ())
        assert (
            tr.get_subtrace("g").get_subtrace("f").get_subtrace("x").get_score()
            == tr.get_subtrace("g", "f", "x").get_score()
        )
        assert (
            tr.get_subtrace("g").get_subtrace("f", "x").get_score()
            == tr.get_subtrace("g", "f", "x").get_score()
        )
        assert (
            tr.get_subtrace("g", "f").get_subtrace("x").get_score()
            == tr.get_subtrace("g", "f", "x").get_score()
        )

    def test_get_subtrace_switch(self):
        @genjax.gen
        def f():
            return genjax.normal(0.0, 0.01) @ "x"

        @genjax.gen
        def g():
            return genjax.uniform(10.0, 11.0) @ "y"

        @genjax.gen
        def h():
            flip = genjax.flip(0.5) @ "flip"
            return f.or_else(g)(flip, (), ()) @ "z"

        tr = h.simulate(jax.random.key(0), ())
        flip_tr = tr.get_subtrace("flip")
        flip = flip_tr.get_retval()
        if flip:
            assert (
                tr.get_subtrace("z", "x").get_score()
                == tr.get_score() - flip_tr.get_score()
            )
        else:
            assert (
                tr.get_subtrace("z", "y").get_score()
                == tr.get_score() - flip_tr.get_score()
            )

    def test_get_subtrace_vmap(self):
        @genjax.vmap()
        @genjax.gen
        def f(x):
            return genjax.normal(x, 0.01) @ "y"

        tr = f.simulate(jax.random.key(0), (jnp.arange(5.0),))
        assert tr.get_subtrace("y").get_score().shape == (5,)
        assert tr.get_score() == jnp.sum(tr.get_subtrace("y").get_score())

    def test_get_subtrace_scan(self):
        @genjax.gen
        def f(state, step):
            return state + genjax.normal(step, 0.01) @ "y", None

        tr = f.scan().simulate(jax.random.key(0), (5.0, jnp.arange(3.0)))
        print(tr)
        assert tr.get_subtrace("y").get_score().shape == (3,)
        assert tr.get_score() == jnp.sum(tr.get_subtrace("y").get_score())

class TestCombinators:
"""Tests for the generative function combinator methods."""

    def test_vmap(self):
        key = jax.random.key(314159)

        @genjax.gen
        def model(x):
            v = genjax.normal(x, 1.0) @ "v"
            return (v, genjax.normal(v, 0.01) @ "q")

        vmapped_model = model.vmap()

        jit_fn = jax.jit(vmapped_model.simulate)

        tr = jit_fn(key, (jnp.array([10.0, 20.0, 30.0]),))
        chm = tr.get_choices()
        varr, qarr = tr.get_retval()

        # The : syntax groups everything under a sub-key:
        assert jnp.array_equal(chm[:, "v"], varr)
        assert jnp.array_equal(chm[:, "q"], qarr)

    def test_repeat(self):
        key = jax.random.key(314159)

        @genjax.gen
        def model(x):
            return genjax.normal(x, 1.0) @ "x"

        vmap_model = model.vmap()
        repeat_model = model.repeat(n=3)

        vmap_tr = jax.jit(vmap_model.simulate)(key, (jnp.zeros(3),))
        repeat_tr = jax.jit(repeat_model.simulate)(key, (0.0,))

        repeatarr = repeat_tr.get_retval()
        varr = vmap_tr.get_retval()

        # Check that we get 3 repeated values:
        assert jnp.array_equal(repeat_tr.get_choices()[:, "x"], repeatarr)

        # check that the return value matches the traced values (in this case)
        assert jnp.array_equal(repeat_tr.get_retval(), repeatarr)

        # vmap does as well, but they are different due to internal seed splitting:
        assert jnp.array_equal(vmap_tr.get_choices()[:, "x"], varr)

    def test_or_else(self):
        key = jax.random.key(314159)

        @genjax.gen
        def if_model(x):
            return genjax.normal(x, 1.0) @ "if_value"

        @genjax.gen
        def else_model(x):
            return genjax.normal(x, 5.0) @ "else_value"

        @genjax.gen
        def switch_model(toss: bool):
            return if_model.or_else(else_model)(toss, (1.0,), (10.0,)) @ "tossed"

        jit_fn = jax.jit(switch_model.simulate)
        if_tr = jit_fn(key, (True,))
        assert "if_value" in if_tr.get_choices()("tossed")

        else_tr = jit_fn(key, (False,))
        assert "else_value" in else_tr.get_choices()("tossed")

================================================
FILE: tests/core/generative/test_functional_types.py
================================================

# Copyright 2023 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import re

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from genjax.\_src.checkify import do_checkify
from genjax.\_src.core.generative.functional_types import Mask

class TestMask:
def test_mask_kwarg_constructor(self): # Test basic kwarg construction
mask1 = Mask(value=42, flag=True)
assert mask1.value == 42
assert mask1.flag is True

        # Test flag defaults to True when not specified
        mask4 = Mask(value=42)
        assert mask4.value == 42
        assert mask4.flag is True

    def test_mask_unmask_without_default(self):
        valid_mask = Mask(42, True)
        assert valid_mask.unmask() == 42

        invalid_mask = Mask(42, False)
        with do_checkify():
            with pytest.raises(Exception):
                invalid_mask.unmask()

    def test_mask_unmask_with_default(self):
        valid_mask = Mask(42, True)
        assert valid_mask.unmask(default=0) == 42

        invalid_mask = Mask(42, False)
        assert invalid_mask.unmask(default=0) == 0

    def test_mask_unmask_pytree(self):
        pytree = {"a": 1, "b": [2, 3], "c": {"d": 4}}
        valid_mask = Mask(pytree, True)
        assert valid_mask.unmask() == pytree

        invalid_mask = Mask(pytree, False)
        default = {"a": 0, "b": [0, 0], "c": {"d": 0}}
        result = invalid_mask.unmask(default=default)
        assert result == default

    def test_build(self):
        mask = Mask.build(42, True)
        assert isinstance(mask, Mask)
        assert mask.flag is True
        assert mask.value == 42

        nested_mask = Mask.build(Mask.build(42, True), False)
        assert isinstance(nested_mask, Mask)
        assert nested_mask.flag is False
        assert nested_mask.value == 42

        with pytest.raises(
            ValueError,
            match=re.escape("(1,) must be a prefix of all leaf shapes. Found ()"),
        ):
            jax.vmap(Mask.build)(
                jnp.arange(2), jnp.array([[True], [False]], dtype=bool)
            )

        # build a vectorized mask
        v_mask = jax.vmap(Mask.build)(jnp.arange(10), jnp.ones(10, dtype=bool))

        # nesting it with a scalar is fine
        nested = Mask.build(v_mask, False)
        assert jnp.array_equal(nested.value, jnp.arange(10))
        assert jnp.array_equal(nested.primal_flag(), jnp.zeros(10, dtype=bool))

        # building with a concrete vs non-concrete scalar is fine
        assert jtu.tree_map(
            jnp.array_equal, nested, Mask.build(v_mask, jnp.array(False))
        )

        # non-scalar flags have to match dimension
        with pytest.raises(
            AssertionError,
            match=re.escape(
                "Can't build a Mask with non-matching Flag shapes (2,) and (10,)"
            ),
        ):
            Mask.build(v_mask, jnp.array([False, True]))

    def test_scalar_flag_validation(self):
        # Boolean flags should be left unchanged
        mask = Mask.build(42, True)
        assert mask.flag is True

        mask = Mask.build([1, 2, 3], False)
        assert mask.flag is False

        # Array flags should only be allowed if they line up with a vectorized value
        value = jnp.array([1.0, 2.0, 3.0])

        with pytest.raises(
            ValueError,
            match=re.escape(
                "shape (1,) must be a prefix of all leaf shapes. Found (3,)"
            ),
        ):
            Mask.build(value, jnp.array([True]))

        mask = Mask.build(value, jnp.array(True))
        assert jnp.array_equal(mask.primal_flag(), jnp.array(True))

        # Works with pytrees
        value = {"a": jnp.ones((3, 2)), "b": jnp.ones((3, 2))}
        flag = jnp.array(False)
        mask = Mask.build(value, flag)
        assert jnp.array_equal(mask.primal_flag(), flag)

        # differing shapes in pytree leaves are fine
        value = {"a": jnp.ones((4, 8)), "b": jnp.ones((3, 2))}
        flag = jnp.array(True)
        mask = Mask.build(value, flag)

    def test_maybe_mask(self):
        result = Mask.maybe_mask(42, True)
        assert result == 42

        result = Mask.maybe_mask(42, False)
        assert result is None

        mask = Mask(42, True)
        assert Mask.maybe_mask(mask, True) == 42
        assert Mask.maybe_mask(mask, False) is None

        assert Mask.maybe_mask(None, jnp.asarray(True)) == Mask(
            None, jnp.asarray(True)
        ), "None survives maybe_mask"

    def test_mask_or_concrete_flags(self):
        # True | True = True
        mask1 = Mask(42, True)
        mask2 = Mask(43, True)
        result = mask1 | mask2
        assert result.primal_flag() is True
        assert result.value == 42

        # True | False = True (takes first value)
        mask1 = Mask(42, True)
        mask2 = Mask(43, False)
        result = mask1 | mask2
        assert result.primal_flag() is True
        assert result.value == 42

        # False | True = True (takes second value)
        mask1 = Mask(42, False)
        mask2 = Mask(43, True)
        result = mask1 | mask2
        assert result.primal_flag() is True
        assert result.value == 43

        # False | False = False
        mask1 = Mask(42, False)
        mask2 = Mask(43, False)
        result = mask1 | mask2
        assert result.primal_flag() is False

        # Array flags result in array flag
        mask1 = Mask(jnp.array([42, 42, 42, 42]), jnp.array([True, True, False, False]))
        mask2 = Mask(jnp.array([43, 43, 43, 43]), jnp.array([False, True, False, True]))
        result = mask1 | mask2
        jtu.tree_map(
            jnp.array_equal,
            result,
            Mask(jnp.array([42, 43, 43, 42]), jnp.array([True, True, False, True])),
        )

    def test_mask_xor_concrete_flags(self):
        # True ^ True = False
        mask1 = Mask(42, True)
        mask2 = Mask(43, True)
        result = mask1 ^ mask2
        assert result.primal_flag() is False

        # True ^ False = True (takes first value)
        mask1 = Mask(42, True)
        mask2 = Mask(43, False)
        result = mask1 ^ mask2
        assert result.primal_flag() is True
        assert result.value == 42

        # False ^ True = True (takes second value)
        mask1 = Mask(42, False)
        mask2 = Mask(43, True)
        result = mask1 ^ mask2
        assert result.primal_flag() is True
        assert result.value == 43

        # False ^ False = False
        mask1 = Mask(42, False)
        mask2 = Mask(43, False)
        result = mask1 ^ mask2
        assert result.primal_flag() is False

        # Array flags result in array flag
        mask1 = Mask(jnp.array([42, 42, 42, 42]), jnp.array([True, True, False, False]))
        mask2 = Mask(jnp.array([43, 43, 43, 43]), jnp.array([False, True, False, True]))
        result = mask1 ^ mask2
        jtu.tree_map(
            jnp.array_equal,
            result,
            Mask(jnp.array([42, 42, 43, 42]), jnp.array([True, False, False, True])),
        )

    def test_mask_combine_different_pytree_shapes(self):
        mask1 = Mask({"a": 1, "b": 2}, True)
        mask2 = Mask({"a": 1}, True)

        with pytest.raises(
            ValueError, match="Cannot combine masks with different tree structures"
        ):
            _ = mask1 | mask2

        with pytest.raises(
            ValueError, match="Cannot combine masks with different tree structures"
        ):
            _ = mask1 ^ mask2

    def test_mask_combine_different_array_shapes(self):
        # Array vs array with different shapes
        mask1 = Mask(jnp.ones((2, 3)), True)
        mask2 = Mask(jnp.ones((2, 2)), True)

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask1 | mask2

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask1 ^ mask2

        # Scalar vs array
        mask3 = Mask(jnp.asarray(1.0), True)
        mask4 = Mask(jnp.ones((2, 2)), True)

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask3 | mask4

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask3 ^ mask4

        # Different scalar shapes
        mask5 = Mask(1.0, True)
        mask6 = Mask(jnp.array(1.0), True)

        assert mask5 | mask6 == mask6  # pyright: ignore

        assert (mask5 ^ mask6).primal_flag() is False  # pyright: ignore

        # Same scalar shapes should work
        mask7 = Mask(1.0, True)
        mask8 = Mask(2.0, False)
        assert mask7 | mask8 == mask7
        assert mask7 ^ mask8 == mask7

        # Vectorized masks with same shape should work
        mask9 = Mask(jnp.array([1.0, 2.0]), jnp.array([True, False]))
        mask10 = Mask(jnp.array([3.0, 4.0]), jnp.array([True, True]))

        # vectorized or works correctly
        assert jtu.tree_map(
            jnp.array_equal,
            mask9 | mask10,
            Mask(jnp.array([1.0, 4.0]), jnp.array([True, True])),
        )

        # vectorized xor works correctly
        assert jtu.tree_map(
            jnp.array_equal,
            mask9 ^ mask10,
            Mask(jnp.array([1.0, 2.0]), jnp.array([False, True])),
        )

        # can't combine different shapes of value
        mask11 = Mask(jnp.array([[3.0, 4.0], [3.0, 4.0]]), jnp.array([True, True]))

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask9 | mask11

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask9 ^ mask11

        # can't combine vectorized with scalar flag
        mask12 = Mask(jnp.array([3.0, 4.0]), jnp.array(True))
        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask9 | mask12

        with pytest.raises(
            ValueError, match="Cannot combine masks with different array shapes"
        ):
            _ = mask9 ^ mask12

    def test_mask_not(self):
        # scalar not works correctly
        mask13 = Mask(1.0, True)
        assert ~mask13 == Mask(1.0, False)

        mask14 = Mask(2.0, False)
        assert ~mask14 == Mask(2.0, True)

        # vectorized not works correctly
        mask15 = Mask(jnp.array([1.0, 2.0]), jnp.array([True, False]))
        assert jtu.tree_map(
            jnp.array_equal,
            ~mask15,
            Mask(jnp.array([1.0, 2.0]), jnp.array([False, True])),
        )

    def test_mask_indexing(self):
        # Test indexing with scalar flag
        scalar_flag_mask = Mask(jnp.array([[1, 2], [3, 4]]), True)
        assert scalar_flag_mask[0, 1].value == 2
        assert scalar_flag_mask[0, 1].primal_flag() is True

        # Test indexing with vectorized flag
        vectorized_mask = Mask(jnp.array([[1, 2], [3, 4]]), jnp.array([True, False]))

        # When indexing with more elements than vectorized dimensions,
        # only first parts of path are applied to flag
        indexed = vectorized_mask[0, 1]
        assert indexed.value == 2
        assert jnp.array_equal(
            indexed.primal_flag(), jnp.array(True)
        )  # Flag only used first index [0]

        indexed2 = vectorized_mask[1, 0]
        assert indexed2.value == 3
        assert jnp.array_equal(
            indexed2.primal_flag(), jnp.array(False)
        )  # Flag only used first index [1]

================================================
FILE: tests/core/interpreters/test_incremental.py
================================================

# Copyright 2023 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

from genjax.\_src.core.compiler.interpreters.incremental import (
Diff,
NoChange,
UnknownChange,
)

class TestDiff:
def test_no_nested_diffs(self):
d1 = Diff.no_change(1.0)
d2 = Diff.unknown_change(d1)
assert not isinstance(d2.get_primal(), Diff)

        assert Diff.static_check_no_change(d1)
        assert not Diff.static_check_no_change(d2)

    def test_tree_diff(self):
        primal_tree = {"a": 1, "b": [2, 3]}
        tangent_tree = {"a": NoChange, "b": [UnknownChange, NoChange]}
        result = Diff.tree_diff(primal_tree, tangent_tree)
        assert isinstance(result["a"], Diff)
        assert isinstance(result["b"][0], Diff)
        assert isinstance(result["b"][1], Diff)
        assert result["a"].get_tangent() == NoChange
        assert result["b"][0].get_tangent() == UnknownChange
        assert result["b"][1].get_tangent() == NoChange

    def test_tree_primal(self):
        tree = {"a": Diff(1, NoChange), "b": [Diff(2, UnknownChange), 3]}
        result = Diff.tree_primal(tree)
        assert result == {"a": 1, "b": [2, 3]}

    def test_tree_tangent(self):
        tree = {"a": 1, "b": [Diff(2, UnknownChange), 3]}
        result = Diff.tree_tangent(tree)

        # note that non-Diffs are marked as UnknownChange, the default tangent value.
        assert result == {"a": NoChange, "b": [UnknownChange, NoChange]}

    def test_static_check_tree_diff(self):
        tree1 = {"a": Diff(1, NoChange), "b": [Diff(2, UnknownChange)]}
        tree2 = {"a": Diff(1, NoChange), "b": [2]}
        assert Diff.static_check_tree_diff(tree1)
        assert not Diff.static_check_tree_diff(tree2)

    def test_static_check_no_change(self):
        tree1 = {"a": Diff(1, NoChange), "b": [Diff(2, NoChange)]}
        tree2 = {"a": Diff(1, NoChange), "b": [Diff(2, UnknownChange)]}
        assert Diff.static_check_no_change(tree1)
        assert not Diff.static_check_no_change(tree2)

================================================
FILE: tests/generative_functions/test_dimap_combinator.py
================================================

# Copyright 2024 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax

import genjax
from genjax import ChoiceMapBuilder as C

class TestDimap:
def test*dimap_update_retval(self): # Define pre- and post-processing functions
def pre_process(x, y):
return (x + 1, y * 2, y \_ 3)

        def post_process(_args, _xformed, retval):
            assert len(_args) == 2, "post_process receives pre-transformed args..."
            assert len(_xformed) == 3, "...and post-transformed args."
            return retval + 2

        def invert_post(x):
            return x - 2

        @genjax.gen
        def model(x, y, _):
            return genjax.normal(x, y) @ "z"

        dimap_model = model.dimap(pre=pre_process, post=post_process)

        # Use the dimap model
        key = jax.random.key(0)
        trace = dimap_model.simulate(key, (2.0, 3.0))
        assert trace.get_retval() == trace.get_choices()["z"] + 2.0, (
            "initial retval is a square of random draw"
        )

        assert (trace.get_score(), trace.get_retval()) == dimap_model.assess(
            trace.get_choices(), (2.0, 3.0)
        ), "assess with the same args returns score, retval"

        assert (
            genjax.normal.logpdf(
                invert_post(trace.get_retval()), *pre_process(2.0, 3.0)
            )
            == trace.get_score()
        ), (
            "final score sees pre-processing but not post-processing (note the inverse). This is only true here because we are returning the sampled value."
        )

        updated_tr, _, _, _ = trace.update(key, C["z"].set(-2.0))
        assert 0.0 == updated_tr.get_retval(), (
            "updated 'z' must hit `post_process` before returning"
        )

        importance_tr, _ = dimap_model.importance(
            key, updated_tr.get_choices(), (1.0, 2.0)
        )
        assert importance_tr.get_retval() == updated_tr.get_retval(), (
            "importance shouldn't update the retval"
        )

        assert (
            genjax.normal.logpdf(
                invert_post(importance_tr.get_retval()), *pre_process(1.0, 2.0)
            )
            == importance_tr.get_score()
        ), (
            "with importance trace, final score sees pre-processing but not post-processing."
        )

================================================
FILE: tests/generative_functions/test_distributions.py
================================================

# Copyright 2023 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax
import jax.numpy as jnp
import pytest

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Diff, NoChange, UnknownChange

class TestDistributions:
def test_simulate(self):
key = jax.random.key(314159)
tr = genjax.normal(0.0, 1.0).simulate(key, ())
assert tr.get_score() == genjax.normal(0.0, 1.0).assess(tr.get_choices(), ())[0]

    def test_importance(self):
        key = jax.random.key(314159)

        # No constraint.
        (tr, w) = genjax.normal.importance(key, C.n(), (0.0, 1.0))
        assert w == 0.0

        # Constraint, no mask.
        (tr, w) = genjax.normal.importance(key, C.v(1.0), (0.0, 1.0))
        v = tr.get_choices()
        assert w == genjax.normal(0.0, 1.0).assess(v, ())[0]

        # Constraint, mask with True flag.
        (tr, w) = genjax.normal.importance(
            key,
            C.v(1.0).mask(jnp.array(True)),
            (0.0, 1.0),
        )
        v = tr.get_choices().get_value()
        assert v == 1.0
        assert w == genjax.normal.assess(C.v(v), (0.0, 1.0))[0]

        # Constraint, mask with False flag.
        (tr, w) = genjax.normal.importance(
            key,
            C.v(1.0).mask(jnp.array(False)),
            (0.0, 1.0),
        )
        v = tr.get_choices().get_value()
        assert v != 1.0
        assert w == 0.0

    def test_update(self):
        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = genjax.normal.simulate(sub_key, (0.0, 1.0))

        # No constraint, no change to arguments.
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.n(),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_choices().get_value() == tr.get_choices().get_value()
        assert (
            new_tr.get_score() == genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )
        assert w == 0.0

        # Constraint, no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.v(1.0),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_choices().get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.assess(C.v(1.0), (0.0, 1.0))[0]
        assert (
            w
            == genjax.normal.assess(C.v(1.0), (0.0, 1.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

        # No constraint, change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.n(),
            (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_choices().get_value() == tr.get_choices().get_value()
        assert (
            new_tr.get_score() == genjax.normal.assess(tr.get_choices(), (1.0, 1.0))[0]
        )
        assert (
            w
            == genjax.normal.assess(tr.get_choices(), (1.0, 1.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint, change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.v(1.0),
            (Diff(1.0, UnknownChange), Diff(2.0, UnknownChange)),
        )
        assert new_tr.get_choices().get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.assess(C.v(1.0), (1.0, 2.0))[0]
        assert (
            w
            == genjax.normal.assess(C.v(1.0), (1.0, 2.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint is masked (True), no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.v(1.0).mask(jnp.array(True)),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_choices().get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.assess(C.v(1.0), (0.0, 1.0))[0]
        assert (
            w
            == genjax.normal.assess(C.v(1.0), (0.0, 1.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint is masked (True), change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.v(1.0).mask(True),
            (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_choices().get_value() == 1.0
        assert new_tr.get_score() == genjax.normal.assess(C.v(1.0), (1.0, 1.0))[0]
        assert (
            w
            == genjax.normal.assess(C.v(1.0), (1.0, 1.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

        # Constraint is masked (False), no change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.v(1.0).mask(False),
            (Diff(0.0, NoChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_choices().get_value() == tr.get_choices().get_value()
        assert (
            new_tr.get_score() == genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )
        assert w == 0.0

        # Constraint is masked (False), change to arguments.
        key, sub_key = jax.random.split(key)
        (new_tr, w, _, _) = genjax.normal.update(
            sub_key,
            tr,
            C.v(1.0).mask(False),
            (Diff(1.0, UnknownChange), Diff(1.0, NoChange)),
        )
        assert new_tr.get_choices().get_value() == tr.get_choices().get_value()
        assert (
            new_tr.get_score() == genjax.normal.assess(tr.get_choices(), (1.0, 1.0))[0]
        )
        assert (
            w
            == genjax.normal.assess(tr.get_choices(), (1.0, 1.0))[0]
            - genjax.normal.assess(tr.get_choices(), (0.0, 1.0))[0]
        )

    def test_using_primitive_distributions(self):
        @genjax.gen
        def model():
            _ = (
                genjax.bernoulli(
                    probs=0.5,
                )
                @ "a"
            )
            _ = genjax.beta(1.0, 1.0) @ "b"
            _ = genjax.beta_binomial(1.0, 1.0, 1.0) @ "c"
            _ = genjax.beta_quotient(1.0, 1.0, 1.0, 1.0) @ "d"
            _ = genjax.binomial(1.0, 0.5) @ "e"
            _ = (
                genjax.cauchy(
                    0.0,
                    1.0,
                )
                @ "f"
            )
            _ = (
                genjax.categorical(
                    probs=[0.5, 0.5],
                )
                @ "g"
            )
            _ = (
                genjax.chi(
                    1.0,
                )
                @ "h"
            )
            _ = (
                genjax.chi2(
                    1.0,
                )
                @ "i"
            )
            _ = (
                genjax.dirichlet(
                    [
                        1.0,
                        1.0,
                    ],
                )
                @ "j"
            )
            _ = (
                genjax.dirichlet_multinomial(
                    1.0,
                    [
                        1.0,
                        1.0,
                    ],
                )
                @ "k"
            )
            _ = genjax.double_sided_maxwell(1.0, 1.0) @ "l"
            _ = (
                genjax.exp_gamma(
                    1.0,
                    1.0,
                )
                @ "m"
            )
            _ = (
                genjax.exp_inverse_gamma(
                    1.0,
                    1.0,
                )
                @ "n"
            )
            _ = (
                genjax.exponential(
                    1.0,
                )
                @ "o"
            )
            _ = (
                genjax.flip(
                    0.5,
                )
                @ "p"
            )
            _ = (
                genjax.gamma(
                    1.0,
                    1.0,
                )
                @ "q"
            )
            _ = (
                genjax.geometric(
                    0.5,
                )
                @ "r"
            )
            _ = (
                genjax.gumbel(
                    0.0,
                    1.0,
                )
                @ "s"
            )
            _ = genjax.half_cauchy(1.0, 1.0) @ "t"
            _ = genjax.half_normal(1.0) @ "u"
            _ = genjax.half_student_t(1.0, 1.0, 1.0) @ "v"
            _ = (
                genjax.inverse_gamma(
                    1.0,
                    1.0,
                )
                @ "w"
            )
            _ = (
                genjax.kumaraswamy(
                    1.0,
                    1.0,
                )
                @ "x"
            )
            _ = (
                genjax.laplace(
                    0.0,
                    1.0,
                )
                @ "y"
            )
            _ = (
                genjax.lambert_w_normal(
                    1.0,
                    1.0,
                )
                @ "z"
            )
            _ = (
                genjax.log_normal(
                    0.0,
                    1.0,
                )
                @ "aa"
            )
            _ = (
                genjax.logit_normal(
                    0.0,
                    1.0,
                )
                @ "bb"
            )
            _ = (
                genjax.moyal(
                    0.0,
                    1.0,
                )
                @ "cc"
            )
            _ = (
                genjax.multinomial(
                    1.0,
                    [0.5, 0.5],
                )
                @ "dd"
            )
            _ = (
                genjax.mv_normal(
                    [0.0, 0.0],
                    [[1.0, 0.0], [0.0, 1.0]],
                )
                @ "ee"
            )
            _ = (
                genjax.mv_normal_diag(
                    jnp.array([1.0, 1.0]),
                )
                @ "ff"
            )
            _ = (
                genjax.negative_binomial(
                    1.0,
                    0.5,
                )
                @ "gg"
            )
            _ = (
                genjax.non_central_chi2(
                    1.0,
                    1.0,
                )
                @ "hh"
            )
            _ = (
                genjax.normal(
                    0.0,
                    1.0,
                )
                @ "ii"
            )
            _ = (
                genjax.poisson(
                    1.0,
                )
                @ "kk"
            )
            _ = (
                genjax.power_spherical(
                    jnp.array([
                        0.0,
                        0.0,
                    ]),
                    1.0,
                )
                @ "ll"
            )
            _ = (
                genjax.skellam(
                    1.0,
                    1.0,
                )
                @ "mm"
            )
            _ = genjax.student_t(1.0, 1.0, 1.0) @ "nn"
            _ = (
                genjax.truncated_cauchy(
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                )
                @ "oo"
            )
            _ = (
                genjax.truncated_normal(
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                )
                @ "pp"
            )
            _ = (
                genjax.uniform(
                    0.0,
                    1.0,
                )
                @ "qq"
            )
            _ = (
                genjax.von_mises(
                    0.0,
                    1.0,
                )
                @ "rr"
            )
            _ = (
                genjax.von_mises_fisher(
                    jnp.array([
                        0.0,
                        0.0,
                    ]),
                    1.0,
                )
                @ "ss"
            )
            _ = (
                genjax.weibull(
                    1.0,
                    1.0,
                )
                @ "tt"
            )
            _ = (
                genjax.zipf(
                    2.0,
                )
                @ "uu"
            )
            return None

        key = jax.random.key(314159)
        _ = model.simulate(key, ())

    def test_distribution_repr(self):
        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.bernoulli(logits=0.0) @ "y"
            z = genjax.flip(0.5) @ "z"
            t = genjax.categorical(logits=[0.0, 0.0]) @ "t"
            return x, y, z, t

        tr = model.simulate(jax.random.key(0), ())
        assert str(tr.get_subtrace("x").get_gen_fn()) == "genjax.normal()"
        assert str(tr.get_subtrace("y").get_gen_fn()) == "genjax.bernoulli()"
        assert str(tr.get_subtrace("z").get_gen_fn()) == "genjax.flip()"
        assert str(tr.get_subtrace("t").get_gen_fn()) == "genjax.categorical()"

    def test_distribution_kwargs(self):
        @genjax.gen
        def model():
            c = genjax.categorical(logits=[-0.3, -0.5]) @ "c"
            p = genjax.categorical(probs=[0.3, 0.7]) @ "p"
            n = genjax.normal(loc=0.0, scale=0.1) @ "n"
            return c + p + n

        tr = model.simulate(jax.random.key(0), ())
        assert tr.get_subtrace("c").get_args() == ((), {"logits": [-0.3, -0.5]})
        assert tr.get_subtrace("p").get_args() == ((), {"probs": [0.3, 0.7]})
        assert tr.get_subtrace("n").get_args() == ((), {"loc": 0.0, "scale": 0.1})

    def test_deprecation_warnings(self):
        @genjax.gen
        def f():
            return genjax.categorical([-0.3, -0.5]) @ "c"

        @genjax.gen
        def g():
            return genjax.bernoulli(-0.4) @ "b"

        with pytest.warns(
            DeprecationWarning, match="bare argument to genjax.categorical"
        ):
            _ = f.simulate(jax.random.key(0), ())

        with pytest.warns(
            DeprecationWarning, match="bare argument to genjax.bernoulli"
        ):
            _ = g.simulate(jax.random.key(0), ())

    def test_switch_with_kwargs(self):
        prim = genjax.bernoulli(0.3)
        prim_kw = genjax.bernoulli(probs=0.3)

        key = jax.random.key(314159)
        with pytest.warns(DeprecationWarning):
            genjax.switch(prim, prim).simulate(key, (0, (), ()))
        genjax.switch(prim_kw, prim_kw).simulate(key, (0, (), ()))

================================================
FILE: tests/generative_functions/test_mask_combinator.py
================================================

# Copyright 2024 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax.\_src.generative_functions.combinators.vmap import VmapTrace

@genjax.mask
@genjax.gen
def model(x):
z = genjax.normal(x, 1.0) @ "z"
return z

class TestMaskCombinator:
@pytest.fixture
def key(self):
return jax.random.key(314159)

    def test_mask_simple_normal_true(self, key):
        tr = jax.jit(model.simulate)(key, (True, -4.0))
        assert tr.get_score() == tr.inner.get_score()
        assert tr.get_retval() == genjax.Mask(tr.inner.get_retval(), jnp.array(True))

        tr = jax.jit(model.simulate)(key, (False, -4.0))
        assert tr.get_score() == 0.0
        assert tr.get_retval() == genjax.Mask(tr.inner.get_retval(), jnp.array(False))

    def test_mask_simple_normal_false(self, key):
        tr = jax.jit(model.simulate)(key, (False, 2.0))
        assert tr.get_score() == 0.0
        assert not tr.get_retval().flag

        score, retval = jax.jit(model.assess)(tr.get_choices(), tr.get_args())
        assert score == 0.0
        assert not retval.flag

        _, w = jax.jit(model.importance)(key, C["z"].set(-2.0), tr.get_args())
        assert w == 0.0

    def test_mask_update_weight_to_argdiffs_from_true(self, key):
        # pre-update, the mask is True
        tr = model.simulate(key, (True, 2.0))

        # mask check arg transition: True --> True
        argdiffs = (Diff.unknown_change(True), Diff.no_change(tr.get_args()[1]))
        w = tr.update(key, C.n(), argdiffs)[1]
        assert w == tr.inner.update(key, C.n())[1]
        assert w == 0.0
        # mask check arg transition: True --> False
        argdiffs = (Diff.unknown_change(False), Diff.no_change(tr.get_args()[1]))
        w = tr.update(key, C.n(), argdiffs)[1]
        assert w == -tr.get_score()

    def test_mask_update_weight_to_argdiffs_from_false(self, key):
        # pre-update mask arg is False
        tr = jax.jit(model.simulate)(key, (False, 2.0))

        # mask check arg transition: False --> True
        w = tr.update(
            key,
            C.n(),
            (Diff.unknown_change(True), Diff.no_change(tr.get_args()[1])),
        )[1]
        assert w == tr.inner.update(key, C.n())[1] + tr.inner.get_score()
        assert w == tr.inner.update(key, C.n())[0].get_score()

        # mask check arg transition: False --> False
        w = tr.update(
            key,
            C.n(),
            (
                Diff.unknown_change(False),
                Diff.no_change(tr.get_args()[1]),
            ),
        )[1]
        assert w == 0.0
        assert w == tr.get_score()

    def test_mask_vmap(self, key):
        @genjax.gen
        def init():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        masks = jnp.array([True, False, True])

        @genjax.gen
        def model_2():
            vmask_init = init.mask().vmap(in_axes=(0))(masks) @ "init"
            return vmask_init

        tr = model_2.simulate(key, ())
        retval = tr.get_retval()
        retval_flag = retval.flag
        retval_val = retval.unmask()
        assert tr.get_score() == jnp.sum(
            retval_flag
            * jax.vmap(lambda v: genjax.normal.logpdf(v, 0.0, 1.0))(retval_val)
        )
        vmap_tr = tr.get_subtrace("init")
        assert isinstance(vmap_tr, VmapTrace)
        inner_scores = jax.vmap(lambda tr: tr.get_score())(vmap_tr.inner)
        # score should be sum of sub-scores masked True
        assert tr.get_score() == inner_scores[0] + inner_scores[2]

    def test_mask_update_weight_to_argdiffs_from_false_(self, key):
        # pre-update mask arg is False
        tr = jax.jit(model.simulate)(key, (False, 2.0))
        # mask check arg transition: False --> True
        argdiffs = (Diff.unknown_change(True), Diff.no_change(tr.get_args()[1]))
        w = tr.update(key, C.n(), argdiffs)[1]
        assert w == tr.inner.update(key, C.n())[1] + tr.inner.get_score()
        assert w == tr.inner.update(key, C.n())[0].get_score()
        # mask check arg transition: False --> False
        argdiffs = (Diff.unknown_change(False), Diff.no_change(tr.get_args()[1]))
        w = tr.update(key, C.n(), argdiffs)[1]
        assert w == 0.0
        assert w == tr.get_score()

    def test_masked_iterate_final_update(self, key):
        masks = jnp.array([True, True])

        @genjax.gen
        def step(x):
            _ = (
                genjax.normal.mask().vmap(in_axes=(0, None, None))(masks, x, 1.0)
                @ "rats"
            )
            return x

        # Create some initial traces:
        key = jax.random.key(0)
        mask_steps = jnp.arange(10) < 5
        model = step.masked_iterate_final()
        init_particle = model.simulate(key, (0.0, mask_steps))

        assert jnp.array_equal(init_particle.get_retval(), jnp.array(0.0))

        step_particle, step_weight, _, _ = model.update(
            key, init_particle, C.n(), Diff.no_change((0.0, mask_steps))
        )
        assert jnp.array_equal(step_weight, jnp.array(0.0))
        assert jnp.array_equal(step_particle.get_retval(), jnp.array(0.0))

        # Testing inference working when we extend the model by unmasking a value.
        argdiffs_ = (Diff.no_change(0.0), Diff.unknown_change(jnp.arange(10) < 6))
        step_particle, step_weight, _, _ = model.update(
            key, init_particle, C.n(), argdiffs_
        )
        assert step_weight != jnp.array(0.0)
        assert step_particle.get_score() == step_weight + init_particle.get_score()

    def test_masked_iterate(self, key):
        masks = jnp.array([True, True])

        @genjax.gen
        def step(x):
            _ = (
                genjax.normal.mask().vmap(in_axes=(0, None, None))(masks, x, 1.0)
                @ "rats"
            )
            return x

        # Create some initial traces:
        key = jax.random.key(0)
        mask_steps = jnp.arange(10) < 5
        model = step.masked_iterate()
        init_particle = model.simulate(key, (0.0, mask_steps))
        assert jnp.array_equal(init_particle.get_retval(), jnp.zeros(11)), (
            "0.0 is threaded through 10 times in addition to the initial value"
        )

    def test_mask_scan_update_type_error(self, key):
        @genjax.gen
        def model_inside():
            masks = jnp.array([True, False, True])
            return genjax.normal(0.0, 1.0).mask().vmap()(masks) @ "init"

        outside_mask = jnp.array([True, False, True])

        @genjax.gen
        def model_outside():
            return genjax.normal(0.0, 1.0).mask().vmap()(outside_mask) @ "init"

        # These tests guard against regression to a strange case where it makes a difference whether
        # a constant `jnp.array` of flags is created inside or outside of a generative function.
        # When inside, the array is recast by JAX into a numpy array, since it appears in the
        # literal pool of a compiled function, but not when outside, where it escapes such
        # treatment.
        inside_tr = model_inside.simulate(key, ())
        outside_tr = model_outside.simulate(key, ())

        assert outside_tr.get_score() == inside_tr.get_score()
        assert jtu.tree_map(
            jnp.array_equal, inside_tr.get_retval(), outside_tr.get_retval()
        )
        assert jtu.tree_map(
            jnp.array_equal, inside_tr.get_choices(), outside_tr.get_choices()
        )

        retval = outside_tr.get_retval()
        retval_masks = retval.flag
        retval_value = retval.unmask()
        assert outside_tr.get_score() == jnp.sum(
            retval_masks
            * jax.vmap(lambda v: genjax.normal.logpdf(v, 0.0, 1.0))(retval_value)
        )

    def test_mask_fails_with_vector_mask(self, key):
        @genjax.gen
        def model():
            return genjax.normal(0.0, 1.0) @ "x"

        masks = jnp.array([True, True, False])

        def simulate_masked(key, masks):
            return model.mask().simulate(key, (masks,))

        with pytest.raises(TypeError):
            simulate_masked(key, masks)

        tr = model.mask().vmap().simulate(key, (masks,))

        # note that it's still possible to vmap.
        assert jnp.all(tr.get_retval().flag == masks)

================================================
FILE: tests/generative_functions/test_mix_combinator.py
================================================

# Copyright 2023 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax
import jax.numpy as jnp

import genjax
from genjax import ChoiceMapBuilder as C

class TestMixture:
def test_mix_basic(self): # Define two simple component functions
@genjax.gen
def comp1(x):
return genjax.normal(x, 1.0) @ "y"

        @genjax.gen
        def comp2(x):
            return genjax.normal(x + 2.0, 0.5) @ "y"

        # Create mixture model
        mixture = genjax.mix(comp1, comp2)

        # Test simulation
        key = jax.random.key(0)
        logits = jnp.array([-0.1, -0.2])
        trace = mixture.simulate(key, (logits, (0.0,), (0.0,)))

        # Check structure
        chm = trace.get_choices()
        assert "mixture_component" in chm
        assert ("component_sample", "y") in chm

        # Test assessment
        choices = C["mixture_component"].set(0) | C["component_sample", "y"].set(1.0)
        score, _ = mixture.assess(choices, (logits, (0.0,), (0.0,)))
        assert jnp.isfinite(score)

    def test_mix_then_simulate(self):
        """GEN-1025 brought up an issue where calling `genjax.mix` between two calls to simulate
        would make the second call fail (somehow capturing the internal mix genfn)"""

        @genjax.gen
        def f():
            return genjax.uniform(0.0, 1.1) @ "uniform"

        @genjax.gen
        def g2():
            return f() @ "f"

        key = jax.random.key(0)
        tr = g2.simulate(key, ())
        genjax.mix(genjax.flip, genjax.flip)
        assert tr == g2.simulate(key, ())

================================================
FILE: tests/generative_functions/test_or_else.py
================================================

# Copyright 2023 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax
import pytest

import genjax

class TestOrElse:
@pytest.fixture
def key(self):
return jax.random.key(314159)

    def test_assess_or_else(self, key):
        @genjax.gen
        def f():
            return genjax.normal(0.0, 1.0) @ "value"

        f_or_f = f.or_else(f)
        args = (True, (), ())
        tr = f_or_f.simulate(key, args)
        score, ret = f_or_f.assess(f_or_f.simulate(key, args).get_choices(), args)

        assert tr.get_score() == score
        assert tr.get_retval() == ret

    def test_assess_or_else_inside_fn(self, key):
        p = 0.5

        @genjax.gen
        def f():
            flip = genjax.flip(p) @ "flip"
            return (
                genjax.normal(0.0, 1.0).or_else(genjax.normal(2.0, 1.0))(flip, (), ())
                @ "value"
            )

        args = ()
        tr = f.simulate(key, args)
        score, ret = f.assess(tr.get_choices(), args)

        assert tr.get_score() == score
        assert tr.get_retval() == ret

================================================
FILE: tests/generative_functions/test_repeat_combinator.py
================================================

# Copyright 2024 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax
import jax.numpy as jnp

from genjax import ChoiceMapBuilder as C
from genjax import gen, normal

class TestRepeatCombinator:
def test_repeat_combinator_importance(self):
@gen
def model():
return normal(0.0, 1.0) @ "x"

        key = jax.random.key(314)
        tr, w = model.repeat(n=10).importance(key, C[1, "x"].set(3.0), ())
        assert normal.assess(C.v(tr.get_choices()[1, "x"]), (0.0, 1.0))[0] == w

    def test_repeat_matches_vmap(self):
        @gen
        def square(x):
            return x * x

        key = jax.random.key(314)
        repeat_retval = square.repeat(n=10)(2)(key)

        assert repeat_retval.shape == (10,), "We asked for and received 10 squares"

        assert jnp.array_equal(square.vmap()(jnp.repeat(2, 10))(key), repeat_retval), (
            "Repeat 10 times matches vmap with 10 equal inputs"
        )

    def test_nested_lookup(self):
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            return x

        key = jax.random.key(0)
        big_model = model.repeat(n=10).repeat(n=10)

        chm = C[jnp.array(0), :, "x"].set(jnp.ones(10))

        tr, _ = big_model.importance(key, chm, ())
        assert jnp.array_equal(tr.get_choices()[0, :, "x"], jnp.ones(10))

================================================
FILE: tests/generative_functions/test_scan_combinator.py
================================================

# Copyright 2023 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import re

import jax
import jax.numpy as jnp
import pytest

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Diff, IndexRequest, Regenerate, StaticRequest
from genjax import Selection as S
from genjax.\_src.core.typing import ArrayLike
from genjax.typing import FloatArray

@genjax.iterate(n=10)
@genjax.gen
def scanner(x):
z = genjax.normal(x, 1.0) @ "z"
return z

class TestIterateSimpleNormal:
@pytest.fixture
def key(self):
return jax.random.key(314159)

    def test_iterate_simple_normal(self, key):
        @genjax.iterate(n=10)
        @genjax.gen
        def scanner(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key, sub_key = jax.random.split(key)
        tr = jax.jit(scanner.simulate)(sub_key, (0.01,))
        scan_score = tr.get_score()
        sel = genjax.Selection.all()
        assert tr.project(key, sel) == scan_score

    def test_iterate_simple_normal_importance(self, key):
        key, sub_key = jax.random.split(key)
        for i in range(1, 5):
            tr, w = jax.jit(scanner.importance)(sub_key, C[i, "z"].set(0.5), (0.01,))
            value = tr.get_choices()[i, "z"]
            assert value == 0.5
            prev = tr.get_choices()[i - 1, "z"]
            assert w == genjax.normal.assess(C.v(value), (prev, 1.0))[0]

    def test_iterate_simple_normal_update(self, key):
        @genjax.iterate(n=10)
        @genjax.gen
        def scanner(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key, sub_key = jax.random.split(key)
        for i in range(1, 5):
            tr, _w = jax.jit(scanner.importance)(sub_key, C[i, "z"].set(0.5), (0.01,))
            new_tr, _w, _rd, _bwd_request = jax.jit(scanner.update)(
                sub_key,
                tr,
                C[i, "z"].set(1.0),
                Diff.no_change((0.01,)),
            )
            assert new_tr.get_choices()[i, "z"] == 1.0

@genjax.gen
def inc(prev: ArrayLike) -> ArrayLike:
return prev + 1

@genjax.gen
def inc_tupled(arg: tuple[ArrayLike, ArrayLike]) -> tuple[ArrayLike, ArrayLike]:
"""Takes a pair, returns a pair."""
prev, offset = arg
return (prev + offset, offset)

class TestIterate:
@pytest.fixture
def key(self):
return jax.random.key(314159)

    def test_inc(self, key):
        """Baseline test that `inc` works!"""
        result = inc.simulate(key, (0,)).get_retval()
        assert result == 1

    def test_iterate(self, key):
        """
        `iterate` returns a generative function that applies the original
        function `n` times and returns an array of each result (not including
        the initial value).
        """
        result = inc.iterate(n=4).simulate(key, (0,)).get_retval()
        assert jnp.array_equal(jnp.asarray(result), jnp.array([0, 1, 2, 3, 4]))

        # same as result, with a jnp.array-wrapped accumulator
        result_wrapped = inc.iterate(n=4).simulate(key, (jnp.array(0),)).get_retval()
        assert jnp.array_equal(jnp.asarray(result), result_wrapped)

    def test_iterate_final(self, key):
        """
        `iterate_final` returns a generative function that applies the original
        function `n` times and returns the final result.
        """

        result = inc.iterate_final(n=10).simulate(key, (0,)).get_retval()
        assert jnp.array_equal(result, 10)

    def test_inc_tupled(self, key):
        """Baseline test demonstrating `inc_tupled`."""
        result = inc_tupled.simulate(key, ((0, 2),)).get_retval()
        assert jnp.array_equal(jnp.asarray(result), jnp.array((2, 2)))

    def test_iterate_tupled(self, key):
        """
        `iterate` on function from tuple => tuple passes the tuple correctly
        from invocation to invocation.
        """
        result = inc_tupled.iterate(n=4).simulate(key, ((0, 2),)).get_retval()
        assert jnp.array_equal(
            jnp.asarray(result),
            jnp.array([[0, 2, 4, 6, 8], [2, 2, 2, 2, 2]]),
        )

    def test_iterate_final_tupled(self, key):
        """
        `iterate` on function from tuple => tuple passes the tuple correctly
        from invocation to invocation. Same idea as above, but with
        `iterate_final`.
        """
        result = inc_tupled.iterate_final(n=10).simulate(key, ((0, 2),)).get_retval()
        assert jnp.array_equal(jnp.asarray(result), jnp.array((20, 2)))

    def test_iterate_array(self, key):
        """
        `iterate` on function with an array-shaped initial value works correctly.
        """

        @genjax.gen
        def double(prev):
            return prev + prev

        result = double.iterate(n=4).simulate(key, (jnp.ones(4),)).get_retval()

        assert jnp.array_equal(
            result,
            jnp.array([
                [1, 1, 1, 1],
                [2, 2, 2, 2],
                [4, 4, 4, 4],
                [8, 8, 8, 8],
                [16, 16, 16, 16],
            ]),
        )

    def test_iterate_matrix(self, key):
        """
        `iterate` on function with matrix-shaped initial value works correctly.
        """

        fibonacci_matrix = jnp.array([[1, 1], [1, 0]])

        @genjax.gen
        def fibonacci_step(prev):
            return fibonacci_matrix @ prev

        iterated_fib = fibonacci_step.iterate(n=5)
        result = iterated_fib.simulate(key, (fibonacci_matrix,)).get_retval()

        # sequence of F^n fibonacci matrices
        expected = jnp.array([
            [[1, 1], [1, 0]],
            [[2, 1], [1, 1]],
            [[3, 2], [2, 1]],
            [[5, 3], [3, 2]],
            [[8, 5], [5, 3]],
            [[13, 8], [8, 5]],
        ])

        assert jnp.array_equal(result, expected)

@genjax.gen
def add(carry, x):
return carry + x

@genjax.gen
def add_tupled(acc, x):
"""accumulator state is a pair."""
carry, offset = acc
return (carry + x + offset, offset)

class TestAccumulateReduceMethods:
@pytest.fixture
def key(self):
return jax.random.key(314159)

    def test_add(self, key):
        """Baseline test that `add` works!"""
        result = add.simulate(key, (0, 2)).get_retval()
        assert result == 2

    def test_accumulate(self, key):
        """
        `accumulate` on a generative function of signature `(accumulator, v) -> accumulator` returns a generative function that

        - takes `(accumulator, jnp.array(v)) -> accumulator`
        - and returns an array of each intermediate accumulator value seen (not including the initial value).
        """
        result = add.accumulate().simulate(key, (0, jnp.ones(4))).get_retval()

        assert jnp.array_equal(result, jnp.array([0, 1, 2, 3, 4]))

        # same as result, but with a wrapped scalar vs a bare `0`.
        result_wrapped = (
            add.accumulate().simulate(key, (jnp.array(0), jnp.ones(4))).get_retval()
        )
        assert jnp.array_equal(result, result_wrapped)

    def test_reduce(self, key):
        """
        `reduce` on a generative function of signature `(accumulator, v) -> accumulator` returns a generative function that

        - takes `(accumulator, jnp.array(v)) -> accumulator`
        - and returns the final `accumulator` produces by folding in each element of `jnp.array(v)`.
        """

        result = add.reduce().simulate(key, (0, jnp.ones(10))).get_retval()
        assert jnp.array_equal(result, 10)

    def test_add_tupled(self, key):
        """Baseline test demonstrating `add_tupled`."""
        result = add_tupled.simulate(key, ((0, 2), 10)).get_retval()
        assert jnp.array_equal(jnp.asarray(result), jnp.array((12, 2)))

    def test_accumulate_tupled(self, key):
        """
        `accumulate` on function with tupled carry state works correctly.
        """
        result = (
            add_tupled.accumulate().simulate(key, ((0, 2), jnp.ones(4))).get_retval()
        )
        assert jnp.array_equal(
            jnp.asarray(result), jnp.array([[0, 3, 6, 9, 12], [2, 2, 2, 2, 2]])
        )
        jax.numpy.hstack

    def test_reduce_tupled(self, key):
        """
        `reduce` on function with tupled carry state works correctly.
        """
        result = add_tupled.reduce().simulate(key, ((0, 2), jnp.ones(10))).get_retval()
        assert jnp.array_equal(jnp.asarray(result), jnp.array((30, 2)))

    def test_accumulate_array(self, key):
        """
        `accumulate` with an array-shaped accumulator works correctly, including the initial value.
        """
        result = add.accumulate().simulate(key, (jnp.ones(4), jnp.eye(4))).get_retval()

        assert jnp.array_equal(
            result,
            jnp.array([
                [1, 1, 1, 1],
                [2, 1, 1, 1],
                [2, 2, 1, 1],
                [2, 2, 2, 1],
                [2, 2, 2, 2],
            ]),
        )

    def test_accumulate_matrix(self, key):
        """
        `accumulate` on function with matrix-shaped initial value works correctly.
        """

        fib = jnp.array([[1, 1], [1, 0]])
        repeated_fib = jnp.broadcast_to(fib, (5, 2, 2))

        @genjax.gen
        def matmul(prev, next):
            return prev @ next

        fib_steps = matmul.accumulate()
        result = fib_steps.simulate(key, (fib, repeated_fib)).get_retval()

        # sequence of F^n fibonacci matrices
        expected = jnp.array([
            [[1, 1], [1, 0]],
            [[2, 1], [1, 1]],
            [[3, 2], [2, 1]],
            [[5, 3], [3, 2]],
            [[8, 5], [5, 3]],
            [[13, 8], [8, 5]],
        ])

        assert jnp.array_equal(result, expected)

class TestScanUpdate:
@pytest.fixture
def key(self):
return jax.random.key(314159)

    def test_scan_update(self, key):
        @genjax.Pytree.dataclass
        class A(genjax.Pytree):
            x: FloatArray

        @genjax.gen
        def step(b, a):
            return genjax.normal(b + a.x, 1e-6) @ "b", None

        @genjax.gen
        def model(k):
            return step.scan(n=3)(k, A(jnp.array([1.0, 2.0, 3.0]))) @ "steps"

        k1, k2 = jax.random.split(key)
        tr = model.simulate(k1, (jnp.array(1.0),))
        u, w, _, _ = tr.update(k2, C["steps", 1, "b"].set(99.0))
        assert jnp.allclose(
            u.get_choices()["steps", :, "b"], jnp.array([2.0, 99.0, 7.0]), atol=0.1
        )
        assert w < -100.0

class TestScanWithParameters:
@pytest.fixture
def key(self):
return jax.random.key(314159)

    @genjax.gen
    @staticmethod
    def step(data, state, update):
        new_state = state + genjax.normal(update, data["noise"]) @ "state"
        return new_state, new_state

    @genjax.gen
    @staticmethod
    def model(data):
        stepper = TestScanWithParameters.step.partial_apply(data)
        return stepper.scan(n=3)(data["initial"], data["updates"]) @ "s"

    def test_scan_with_parameters(self, key):
        tr = TestScanWithParameters.model.simulate(
            key,
            (
                {
                    "initial": jnp.array(3.0),
                    "updates": jnp.array([5.0, 6.0, 7.0]),
                    "noise": 1e-6,
                },
            ),
        )

        end, steps = tr.get_retval()

        assert jnp.allclose(steps, jnp.array([8.0, 14.0, 21.0]), atol=0.1)
        assert jnp.allclose(end, jnp.array(21.0), atol=0.1)

    def test_scan_length_inferred(self, key):
        @genjax.gen
        def walk_step(x, std):
            new_x = genjax.normal(x, std) @ "x"
            return new_x, new_x

        args = (0.0, jnp.array([2.0, 4.0, 3.0, 5.0, 1.0]))
        tr = walk_step.scan(n=5).simulate(key, args)
        _, expected = tr.get_retval()
        assert jnp.allclose(
            tr.get_choices()[:, "x"],
            expected,
        )

        tr = walk_step.scan().simulate(key, args)
        assert jnp.allclose(tr.get_choices()[:, "x"], expected)

        # now with jit
        jitted = jax.jit(walk_step.scan().simulate)
        tr = jitted(key, args)
        assert jnp.allclose(tr.get_choices()[:, "x"], expected)

    def test_zero_length_scan(self, key):
        # GEN-333
        @genjax.gen
        def step(state, sigma):
            new_x = genjax.normal(state, sigma) @ "x"
            return (new_x, new_x + 1)

        trace = step.scan(n=0).simulate(key, (2.0, jnp.arange(0, dtype=float)))

        assert trace.get_choices().static_is_empty(), (
            "zero-length scan produces empty choicemaps."
        )

        key, subkey = jax.random.split(key)
        step.scan().importance(
            subkey,
            trace.get_choices(),
            (2.0, 2.0 + jnp.arange(0, dtype=float)),
        )

    def test_scan_validation(self, key):
        @genjax.gen
        def foo(shift, d):
            loc = d["loc"]
            scale = d["scale"]
            x = genjax.normal(loc, scale) @ "x"
            return x + shift, None

        d = {
            "loc": jnp.array([10.0, 12.0]),
            "scale": jnp.array([1.0]),
        }
        with pytest.raises(
            ValueError,
            match=re.escape("scan got values with different leading axis sizes: 2, 1."),
        ):
            jax.jit(foo.scan().simulate)(key, (jnp.array([1.0]), d))

    def test_vmap_key_scan(self, key):
        @genjax.gen
        def model(x, _):
            y = genjax.normal(x, 1.0) @ "y"
            return y, None

        vmapped = model.scan()

        keys = jax.random.split(key, 10)
        xs = jnp.arange(5, dtype=float)
        args = (jnp.array(1.0), xs)

        results = jax.vmap(lambda k: vmapped.simulate(k, args))(jnp.array(keys))

        chm = results.get_choices()

        # the inner scan aggregates a score, while the outer vmap does not accumulate anything
        assert results.get_score().shape == (10,)

        # the inner scan has scanned over the y's
        assert chm[:, "y"].shape == (10, 5)

class TestScanRegenerate:
@pytest.fixture
def key(self):
return jax.random.key(314159)

    def test_scan_regenerate(self, key):
        @genjax.gen
        def scanned_normal():
            @genjax.gen
            def kernel(carry, _):
                z = genjax.normal(0.0, 1.0) @ "z"
                return z, None

            y1 = genjax.normal(0.0, 1.0) @ "y1"
            _ = genjax.normal(0.0, 1.0) @ "y2"
            return kernel.scan(n=10)(y1, None) @ "kernel"

        key, sub_key = jax.random.split(key)
        tr = scanned_normal.simulate(sub_key, ())
        # First, try y1 and test for correctness.
        old_y1 = tr.get_choices()["y1"]
        old_target_density = genjax.normal.logpdf(old_y1, 0.0, 1.0)
        request = genjax.Regenerate(S.at["y1"])
        new_tr, fwd_w, _, _ = request.edit(key, tr, ())
        new_y1 = new_tr.get_choices()["y1"]
        new_target_density = genjax.normal.logpdf(new_y1, 0.0, 1.0)
        assert fwd_w == new_target_density - old_target_density

class TestScanIndexRequest:
@pytest.fixture
def key(self):
return jax.random.key(314159)

    def test_scan_regenerate(self):
        @genjax.gen
        def scanned_normal():
            @genjax.gen
            def kernel(carry, _):
                z = genjax.normal(0.0, 1.0) @ "z"
                return z, None

            y1 = genjax.normal(0.0, 1.0) @ "y1"
            _ = genjax.normal(0.0, 1.0) @ "y2"
            return kernel.scan(n=10)(y1, None) @ "kernel"

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = scanned_normal.simulate(sub_key, ())
        # Try all indices and test for correctness.
        for idx in range(10):
            old_z = tr.get_choices()["kernel", idx, "z"]
            old_target_density = genjax.normal.logpdf(old_z, 0.0, 1.0)
            request = StaticRequest({
                "kernel": IndexRequest(jnp.array(idx), Regenerate(S.at["z"])),
            })
            new_tr, fwd_w, _, _ = request.edit(key, tr, ())
            new_z = new_tr.get_choices()["kernel", idx, "z"]
            new_target_density = genjax.normal.logpdf(new_z, 0.0, 1.0)
            assert fwd_w == new_target_density - old_target_density

        with pytest.raises(AssertionError):
            idx = 11
            old_z = tr.get_choices()["kernel", idx, "z"]
            old_target_density = genjax.normal.logpdf(old_z, 0.0, 1.0)
            request = StaticRequest({
                "kernel": IndexRequest(jnp.array(idx), Regenerate(S.at["z"])),
            })
            new_tr, fwd_w, _, _ = request.edit(key, tr, ())
            new_z = new_tr.get_choices()["kernel", idx, "z"]
            new_target_density = genjax.normal.logpdf(new_z, 0.0, 1.0)
            assert fwd_w == new_target_density - old_target_density

================================================
FILE: tests/generative_functions/test_switch_combinator.py
================================================

# Copyright 2023 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import re

import jax
import pytest
from jax import numpy as jnp

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Diff
from genjax.\_src.core.typing import Array

class TestSwitch:
def test_switch_combinator_simulate_in_gen_fn(self):
@genjax.gen
def f():
x = genjax.normal(0.0, 1.0) @ "x"
return x

        @genjax.gen
        def model():
            b = genjax.flip(0.5) @ "b"
            s = f.switch(f)(jnp.int32(b), (), ()) @ "s"
            return s

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = model.simulate(sub_key, ())
        assert tr.get_retval() == tr.get_choices()["s", "x"].unmask()

    def test_switch_combinator_simulate(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        @genjax.gen
        def simple_flip():
            _y3 = genjax.flip(0.3) @ "y3"

        switch = simple_normal.switch(simple_flip)

        key = jax.random.key(314159)
        jitted = jax.jit(switch.simulate)
        key, sub_key = jax.random.split(key)
        tr = jitted(sub_key, (0, (), ()))
        v1 = tr.get_choices()["y1"]
        v2 = tr.get_choices()["y2"]
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        v1_score, _ = genjax.normal.assess(C.v(v1), (0.0, 1.0))
        key, sub_key = jax.random.split(key)
        v2_score, _ = genjax.normal.assess(C.v(v2), (0.0, 1.0))
        assert score == v1_score + v2_score
        assert tr.get_args() == (0, (), ())
        key, sub_key = jax.random.split(key)
        tr = jitted(sub_key, (1, (), ()))
        b = tr.get_choices().get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (flip_score, _) = genjax.flip.assess(b, (0.3,))
        assert score == flip_score
        (idx, *_) = tr.get_args()
        assert idx == 1

    def test_switch_combinator_choice_map_behavior(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        @genjax.gen
        def simple_flip():
            _y3 = genjax.flip(0.3) @ "y3"

        switch = simple_normal.switch(simple_flip)

        key = jax.random.key(314159)
        jitted = jax.jit(switch.simulate)
        tr = jitted(key, (0, (), ()))
        chm = tr.get_choices()
        assert "y1" in chm
        assert "y2" in chm
        assert "y3" in chm
        assert chm["y3"] == genjax.Mask(jnp.array(False), jnp.array(False))

    def test_switch_combinator_importance(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        @genjax.gen
        def simple_flip():
            _y3 = genjax.flip(0.3) @ "y3"

        switch = simple_normal.switch(simple_flip)

        key = jax.random.key(314159)
        chm = C.n()
        jitted = jax.jit(switch.importance)
        key, sub_key = jax.random.split(key)
        (tr, w) = jitted(sub_key, chm, (0, (), ()))
        v1 = tr.get_choices().get_submap("y1")
        v2 = tr.get_choices().get_submap("y2")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        v1_score, _ = genjax.normal.assess(v1, (0.0, 1.0))
        key, sub_key = jax.random.split(key)
        v2_score, _ = genjax.normal.assess(v2, (0.0, 1.0))
        assert score == v1_score + v2_score
        assert w == 0.0
        key, sub_key = jax.random.split(key)
        (tr, w) = jitted(sub_key, chm, (1, (), ()))
        b = tr.get_choices().get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (flip_score, _) = genjax.flip.assess(b, (0.3,))
        assert score == flip_score
        assert w == 0.0
        chm = C["y3"].set(1)
        key, sub_key = jax.random.split(key)
        (tr, w) = jitted(sub_key, chm, (1, (), ()))
        b = tr.get_choices().get_submap("y3")
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (flip_score, _) = genjax.flip.assess(b, (0.3,))
        assert score == flip_score
        assert w == score

    def test_switch_combinator_update_single_branch_no_change(self):
        @genjax.gen
        def simple_normal():
            _y1 = genjax.normal(0.0, 1.0) @ "y1"
            _y2 = genjax.normal(0.0, 1.0) @ "y2"

        switch = simple_normal.switch()
        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(switch.simulate)(sub_key, (0, ()))
        v1 = tr.get_choices()["y1"]
        v2 = tr.get_choices()["y2"]
        score = tr.get_score()
        key, sub_key = jax.random.split(key)
        (tr, _, _, _) = jax.jit(switch.update)(
            sub_key,
            tr,
            C.n(),
            (Diff.no_change(0), ()),
        )
        assert score == tr.get_score()
        assert v1 == tr.get_choices()["y1"]
        assert v2 == tr.get_choices()["y2"]

    def test_switch_combinator_update_updates_score(self):
        regular_stddev = 1.0
        outlier_stddev = 10.0
        sample_value = 2.0

        @genjax.gen
        def regular():
            x = genjax.normal(0.0, regular_stddev) @ "x"
            return x

        @genjax.gen
        def outlier():
            x = genjax.normal(0.0, outlier_stddev) @ "x"
            return x

        key = jax.random.key(314159)
        switch = regular.switch(outlier)
        key, importance_key = jax.random.split(key)

        (tr, wt) = switch.importance(
            importance_key, C["x"].set(sample_value), (0, (), ())
        )
        (idx, *_) = tr.get_args()
        assert idx == 0
        assert (
            tr.get_score()
            == genjax.normal.assess(C.v(sample_value), (0.0, regular_stddev))[0]
        )
        assert wt == tr.get_score()

        key, update_key = jax.random.split(key)
        (new_tr, new_wt, _, _) = switch.update(
            update_key,
            tr,
            C.n(),
            (Diff.unknown_change(1), (), ()),
        )
        (idx, *_) = new_tr.get_args()
        assert idx == 1
        assert new_tr.get_score() != tr.get_score()
        assert tr.get_score() + new_wt == pytest.approx(new_tr.get_score(), 1e-5)

    def test_switch_combinator_vectorized_access(self):
        @genjax.gen
        def f1():
            return genjax.normal(0.0, 1.0) @ "y"

        @genjax.gen
        def f2():
            return genjax.normal(0.0, 2.0) @ "y"

        s = f1.switch(f2)

        keys = jax.random.split(jax.random.key(17), 3)
        # Just select 0 in all branches for simplicity:
        tr = jax.vmap(s.simulate, in_axes=(0, None))(keys, (0, (), ()))
        y = tr.get_choices()["y"].unmask()
        assert y.shape == (3,)

    def test_switch_combinator_with_empty_gen_fn(self):
        @genjax.gen
        def f():
            x = genjax.normal(0.0, 1.0) @ "x"
            return x

        @genjax.gen
        def empty():
            return jnp.asarray(0.0)

        @genjax.gen
        def model():
            b = genjax.flip(0.5) @ "b"
            s = f.switch(empty)(jnp.int32(b), (), ()) @ "s"
            return s

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        chm = C["b"].set(1)
        tr, _ = model.importance(sub_key, chm, ())
        assert 0.0 == tr.get_retval()

    def test_switch_combinator_with_different_return_types(self):
        @genjax.gen
        def identity(x: int) -> Array:
            return jnp.asarray(x)

        @genjax.gen
        def bool_branch(_: int) -> Array:
            return jnp.asarray(True)

        k = jax.random.key(0)

        switch_model = genjax.switch(identity, bool_branch)

        bare_idx_result = switch_model(1, (10,), (10,))(k)
        assert bare_idx_result == jnp.asarray(1)
        assert bare_idx_result.dtype == jnp.int32

        # this case returns 1
        array_idx_result = switch_model(jnp.array(1), (10,), (10,))(k)
        assert array_idx_result == jnp.asarray(1)
        assert array_idx_result.dtype == bare_idx_result.dtype

    def test_runtime_incompatible_types(self):
        @genjax.gen
        def three_branch(x: int):
            return jax.numpy.ones(3)

        @genjax.gen
        def four_branch(_: int):
            return jax.numpy.ones(4)

        k = jax.random.key(0)
        switch_model = three_branch.switch(four_branch)

        with pytest.raises(ValueError, match="Incompatible shapes for broadcasting"):
            switch_model(0, (10,), (10,))(k)

    def test_switch_distinct_addresses(self):
        @genjax.gen
        def x_z():
            x = genjax.normal(0.0, 1.0) @ "x"
            _ = genjax.normal(x, jnp.ones(3)) @ "z"
            return x

        @genjax.gen
        def x_y():
            x = genjax.normal(0.0, 2.0) @ "x"
            _ = genjax.normal(x, jnp.ones(20)) @ "y"
            return x

        model = x_z.switch(x_y)
        k = jax.random.key(0)
        tr = model.simulate(k, (jnp.array(0), (), ()))

        # both xs match, so it's fine to combine across models
        assert tr.get_choices()["x"].unmask().shape == ()

        # y and z only show up on one side of the `switch` so any shape is fine
        assert tr.get_choices()["y"].unmask().shape == (20,)
        assert tr.get_choices()["z"].unmask().shape == (3,)

        @genjax.gen
        def arr_x():
            _ = genjax.normal(0.0, jnp.array([2.0, 2.0])) @ "x"
            _ = genjax.normal(0.0, jnp.ones(20)) @ "y"
            return jnp.array(1.0)

        mismatched_tr = x_z.switch(arr_x).simulate(k, (jnp.array(0), (), ()))

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Cannot combine masks with different array shapes: () vs (2,)"
            ),
        ):
            mismatched_tr.get_choices()["x"]

================================================
FILE: tests/generative_functions/test_vmap_combinator.py
================================================

# Copyright 2023 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import re

import jax
import jax.numpy as jnp
import pytest

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import IndexRequest, Regenerate, Selection, StaticRequest, Update
from genjax import Selection as S
from genjax.\_src.core.typing import IntArray

class TestVmap:
def test_vmap_combinator_simple_normal(self):
@genjax.vmap(in_axes=(0,))
@genjax.gen
def model(x):
z = genjax.normal(x, 1.0) @ "z"
return z

        key = jax.random.key(314159)
        map_over = jnp.arange(0, 50, dtype=float)
        tr = jax.jit(model.simulate)(key, (map_over,))
        map_score = tr.get_score()
        assert map_score == jnp.sum(tr.inner.get_score())

    def test_vmap_simple_normal_project(self):
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        vmapped = model.vmap(in_axes=(0,))

        key = jax.random.key(314159)
        means = jnp.arange(0, 10, dtype=float)

        tr = jax.jit(vmapped.simulate)(key, (means,))

        vmapped_score = tr.get_score()

        assert tr.project(key, Selection.all()) == vmapped_score
        assert tr.project(key, Selection.none()) == 0.0

    def test_vmap_combinator_vector_choice_map_importance(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def kernel(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.key(314159)
        map_over = jnp.arange(0, 3, dtype=float)
        chm = jax.vmap(lambda idx, v: C[idx, "z"].set(v))(
            jnp.arange(3), jnp.array([3.0, 2.0, 3.0])
        )

        (_, w) = jax.jit(kernel.importance)(key, chm, (map_over,))
        assert (
            w
            == genjax.normal.assess(C.v(3.0), (0.0, 1.0))[0]
            + genjax.normal.assess(C.v(2.0), (1.0, 1.0))[0]
            + genjax.normal.assess(C.v(3.0), (2.0, 1.0))[0]
        )

    def test_vmap_combinator_indexed_choice_map_importance(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def kernel(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.key(314159)
        map_over = jnp.arange(0, 3, dtype=float)
        chm = C[0, "z"].set(3.0)
        key, sub_key = jax.random.split(key)
        (_, w) = jax.jit(kernel.importance)(sub_key, chm, (map_over,))
        assert w == genjax.normal.assess(C.v(3.0), (0.0, 1.0))[0]

        key, sub_key = jax.random.split(key)
        zv = jnp.array([3.0, -1.0, 2.0])
        chm = jax.vmap(lambda idx, v: C[idx, "z"].set(v))(jnp.arange(3), zv)
        (tr, _) = kernel.importance(sub_key, chm, (map_over,))
        for i in range(0, 3):
            v = tr.get_choices()[i, "z"]
            assert v == zv[i]

    def test_vmap_combinator_nested_indexed_choice_map_importance(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def higher_model(x):
            return model(x) @ "outer"

        key = jax.random.key(314159)
        map_over = jnp.ones((3, 3), dtype=float)
        chm = C[0, "outer", 1, "z"].set(1.0)
        (_, w) = jax.jit(higher_model.importance)(key, chm, (map_over,))
        assert w == genjax.normal.assess(C.v(1.0), (1.0, 1.0))[0]

    def test_vmap_combinator_vmap_pytree(self):
        @genjax.gen
        def model2(x):
            _ = genjax.normal(x, 1.0) @ "y"
            return x

        model_mv2 = model2.mask().vmap()
        masks = jnp.array([
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ])
        xs = jnp.arange(0.0, 10.0, 1.0)

        key = jax.random.key(314159)

        # show that we don't error if we map along multiple axes via the default.
        tr = jax.jit(model_mv2.simulate)(key, (masks, xs))
        assert jnp.array_equal(tr.get_retval().value, xs)
        assert jnp.array_equal(tr.get_retval().flag, masks)

        @genjax.vmap(in_axes=(None, (0, None)))
        @genjax.gen
        def foo(y, args):
            loc, (scale, _) = args
            x = genjax.normal(loc, scale) @ "x"
            return x + y

        _ = jax.jit(foo.simulate)(key, (10.0, (jnp.arange(3.0), (1.0, jnp.arange(3)))))

    def test_vmap_combinator_assess(self):
        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def model(x):
            z = genjax.normal(x, 1.0) @ "z"
            return z

        key = jax.random.key(314159)
        map_over = jnp.arange(0, 50, dtype=float)
        tr = jax.jit(model.simulate)(key, (map_over,))
        sample = tr.get_choices()
        map_score = tr.get_score()
        assert model.assess(sample, (map_over,))[0] == map_score

    def test_vmap_validation(self):
        @genjax.gen
        def foo(loc: float, scale: float):
            return genjax.normal(loc, scale) @ "x"

        key = jax.random.key(314159)

        with pytest.raises(
            ValueError,
            match="vmap was requested to map its argument along axis 0, which implies that its rank should be at least 1, but is only 0",
        ):
            jax.jit(foo.vmap(in_axes=(0, None)).simulate)(key, (10.0, jnp.arange(3.0)))

        # in_axes doesn't match args
        with pytest.raises(
            ValueError,
            match="vmap in_axes specification must be a tree prefix of the corresponding value",
        ):
            jax.jit(foo.vmap(in_axes=(0, (0, None))).simulate)(
                key, (10.0, jnp.arange(3.0))
            )

        with pytest.raises(
            IndexError,
        ):
            jax.jit(foo.vmap(in_axes=0).simulate)(key, (jnp.arange(2), jnp.arange(3)))

        # in_axes doesn't match args
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Found incompatible dtypes, <class 'numpy.float32'> and <class 'numpy.int32'>"
            ),
        ):
            jax.jit(foo.vmap(in_axes=(None, 0)).simulate)(key, (10.0, jnp.arange(3)))

    def test_vmap_key_vmap(self):
        @genjax.gen
        def model(x):
            y = genjax.normal(x, 1.0) @ "y"
            return y

        vmapped = model.vmap(in_axes=(0,))

        key = jax.random.key(314159)
        keys = jax.random.split(key, 10)
        xs = jnp.arange(5, dtype=float)

        results = jax.vmap(lambda k: vmapped.simulate(k, (xs,)))(jnp.array(keys))

        chm = results.get_choices()

        # the inner vmap aggregates a score, while the outer vmap does not accumulate anything
        assert results.get_score().shape == (10,)

        # the inner vmap has vmap'd over the y's
        assert chm[:, "y"].shape == (10, 5)

    def test_zero_length_vmap(self):
        @genjax.gen
        def step(state, sigma):
            new_x = genjax.normal(state, sigma) @ "x"
            return (new_x, new_x + 1)

        trace = step.vmap(in_axes=(None, 0)).simulate(
            jax.random.key(20), (2.0, jnp.arange(0, dtype=float))
        )

        assert trace.get_choices().static_is_empty(), (
            "zero-length vmap produces empty choicemaps."
        )

@genjax.Pytree.dataclass
class MyClass(genjax.PythonicPytree):
x: IntArray

class TestVmapPytree:
def test_vmap_pytree(self):
batched_val = jax.vmap(lambda x: MyClass(x))(jnp.arange(5))

        def regular_function(mc: MyClass):
            return mc.x + 5

        assert jnp.array_equal(
            jax.vmap(regular_function)(batched_val), jnp.arange(5) + 5
        )

        @genjax.gen
        def generative_function(mc: MyClass):
            return mc.x + 5

        key = jax.random.key(0)

        # check that we can vmap over a vectorized pytree.
        assert jnp.array_equal(
            generative_function.vmap(in_axes=0)(batched_val)(key), jnp.arange(5) + 5
        )

class TestVmapIndexRequest:
@pytest.fixture
def key(self):
return jax.random.key(314159)

    def test_vmap_regenerate(self):
        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            _ = genjax.normal.vmap()(jnp.zeros(1000), jnp.ones(1000)) @ "a"
            return x

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = model.simulate(sub_key, ())
        for idx in range(10):
            old_a = tr.get_choices()["a", idx]
            old_target_density = genjax.normal.logpdf(old_a, 0.0, 1.0)

            request = StaticRequest({
                "a": IndexRequest(jnp.array(idx), Regenerate(S.all()))
            })

            new_tr, fwd_w, _, _ = request.edit(key, tr, ())
            new_a = new_tr.get_choices()["a", idx]
            new_target_density = genjax.normal.logpdf(new_a, 0.0, 1.0)

            assert fwd_w == new_target_density - old_target_density

    def test_vmap_update(self):
        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            _ = genjax.normal.vmap()(jnp.zeros(1000), jnp.ones(1000)) @ "a"
            return x

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = model.simulate(sub_key, ())
        for idx in range(10):
            old_a = tr.get_choices()["a", idx]
            old_target_density = genjax.normal.logpdf(old_a, 0.0, 1.0)

            request = StaticRequest({
                "a": IndexRequest(jnp.array(idx), Update(C.v(idx + 7.0)))
            })

            new_tr, fwd_w, _, _ = request.edit(key, tr, ())
            new_a = new_tr.get_choices()["a", idx]
            new_target_density = genjax.normal.logpdf(new_a, 0.0, 1.0)

            assert new_a == idx + 7.0
            assert fwd_w == new_target_density - old_target_density

================================================
FILE: tests/inference/test_requests.py
================================================

# Copyright 2024 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu
import pytest

import genjax
from genjax import (
ChoiceMap,
Diff,
DiffAnnotate,
EmptyRequest,
Regenerate,
Selection,
Update,
)
from genjax import ChoiceMap as C
from genjax import SelectionBuilder as S
from genjax.\_src.generative_functions.static import StaticRequest
from genjax.inference.requests import HMC, Rejuvenate, SafeHMC

class TestRegenerate:
def test_simple_normal_regenerate(self):
@genjax.gen
def simple_normal():
y1 = genjax.normal(0.0, 1.0) @ "y1"
y2 = genjax.normal(0.0, 1.0) @ "y2"
return y1 + y2

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal.simulate(sub_key, ())

        # First, try y1 and test for correctness.
        old_v = tr.get_choices()["y1"]
        request = genjax.Regenerate(S["y1"])
        new_tr, fwd_w, _, bwd_request = request.edit(key, tr, ())
        old_density = genjax.normal.logpdf(old_v, 0.0, 1.0)
        new_density = genjax.normal.logpdf(new_tr.get_choices()["y1"], 0.0, 1.0)
        assert fwd_w != 0.0
        assert fwd_w == new_density - old_density
        new_v = new_tr.get_choices()["y1"]
        assert old_v != new_v
        old_tr, bwd_w, _, bwd_request = bwd_request.edit(sub_key, new_tr, ())
        assert bwd_w != 0.0
        assert (fwd_w + bwd_w) == 0.0
        old_old_v = old_tr.get_choices()["y1"]
        assert old_old_v == old_v

        # Now, do y2
        old_v = tr.get_choices()["y2"]
        request = genjax.Regenerate(S["y2"])
        new_tr, fwd_w, _, bwd_request = request.edit(key, tr, ())
        old_density = genjax.normal.logpdf(old_v, 0.0, 1.0)
        new_density = genjax.normal.logpdf(new_tr.get_choices()["y2"], 0.0, 1.0)
        assert fwd_w != 0.0
        assert fwd_w == new_density - old_density
        new_v = new_tr.get_choices()["y2"]
        assert old_v != new_v
        old_tr, bwd_w, _, bwd_request = bwd_request.edit(key, new_tr, ())
        assert bwd_w != 0.0
        assert (fwd_w + bwd_w) == 0.0
        old_old_v = old_tr.get_choices()["y2"]
        assert old_old_v == old_v

        # What about both?
        old_v = tr.get_choices()["y2"]
        request = genjax.Regenerate(
            S["y1"] | S["y2"],
        )
        new_tr, fwd_w, _, bwd_request = request.edit(key, tr, ())
        new_v = new_tr.get_choices()["y2"]
        assert old_v != new_v
        old_tr, bwd_w, _, bwd_request = bwd_request.edit(key, new_tr, ())
        assert (fwd_w + bwd_w) == 0.0
        old_old_v = old_tr.get_choices()["y2"]
        assert old_old_v == old_v

    def test_linked_normal_regenerate(self):
        @genjax.gen
        def linked_normal():
            y1 = genjax.normal(0.0, 1.0) @ "y1"
            _ = genjax.normal(y1, 1.0) @ "y2"

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = linked_normal.simulate(sub_key, ())

        # First, try y1 and test for correctness.
        old_y1 = tr.get_choices()["y1"]
        old_y2 = tr.get_choices()["y2"]
        old_target_density = genjax.normal.logpdf(
            old_y1, 0.0, 1.0
        ) + genjax.normal.logpdf(old_y2, old_y1, 1.0)
        request = genjax.Regenerate(S["y1"])
        new_tr, fwd_w, _, _ = request.edit(key, tr, ())
        new_y1 = new_tr.get_choices()["y1"]
        new_y2 = new_tr.get_choices()["y2"]
        new_target_density = genjax.normal.logpdf(
            new_y1, 0.0, 1.0
        ) + genjax.normal.logpdf(new_y2, new_y1, 1.0)
        assert fwd_w != 0.0
        assert fwd_w == pytest.approx(new_target_density - old_target_density, 1e-6)

    def test_linked_normal_convergence(self):
        @genjax.gen
        def linked_normal():
            y1 = genjax.normal(0.0, 3.0) @ "y1"
            _ = genjax.normal(y1, 0.01) @ "y2"

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr, _ = linked_normal.importance(sub_key, C.kw(y2=3.0), ())
        request = Regenerate(S["y1"])

        # Run Metropolis-Hastings for 200 steps.
        for _ in range(200):
            key, sub_key = jax.random.split(key)
            new_tr, w, _, _ = request.edit(sub_key, tr, ())
            key, sub_key = jax.random.split(key)
            check = jnp.log(genjax.uniform.sample(sub_key, 0.0, 1.0)) < w
            tr = jtu.tree_map(lambda v1, v2: jnp.where(check, v1, v2), new_tr, tr)

        assert tr.get_choices()["y1"] == pytest.approx(3.0, 1e-2)

class TestRejuvenate:
def test\*simple_normal_correctness(self):
@genjax.gen
def simple_normal():

-   = genjax.normal(0.0, 1.0) @ "y1"

          key = jax.random.key(314159)
          key, sub_key = jax.random.split(key)
          tr = simple_normal.simulate(sub_key, ())
          old_v = tr.get_choices()["y1"]

          #####
          # Suppose the proposal is the prior, and its symmetric.
          #####

          request = StaticRequest({
              "y1": Rejuvenate(
                  genjax.normal,
                  lambda chm: (0.0, 1.0),
              )
          })
          new_tr, w, _, _ = request.edit(sub_key, tr, ())
          new_v = new_tr.get_choices()["y1"]
          assert old_v != new_v
          assert w == 0.0

    def test*linked_normal_rejuvenate_convergence(self):
    @genjax.gen
    def linked_normal():
    y1 = genjax.normal(0.0, 3.0) @ "y1"
    * = genjax.normal(y1, 0.001) @ "y2"

          key = jax.random.key(314159)
          key, sub_key = jax.random.split(key)
          tr, _ = linked_normal.importance(sub_key, C.kw(y2=3.0), ())

          request = StaticRequest({
              "y1": Rejuvenate(
                  genjax.normal,
                  lambda chm: (chm.get_value(), 0.3),
              )
          })

          # Run Metropolis-Hastings for 100 steps.
          for _ in range(100):
              key, sub_key = jax.random.split(key)
              new_tr, w, _, _ = request.edit(sub_key, tr, ())
              key, sub_key = jax.random.split(key)
              check = jnp.log(genjax.uniform.sample(sub_key, 0.0, 1.0)) < w
              tr = jtu.tree_map(lambda v1, v2: jnp.where(check, v1, v2), new_tr, tr)

          assert tr.get_choices()["y1"] == pytest.approx(3.0, 5e-3)

class TestHMC:
def test_simple_normal_hmc(self):
@genjax.gen
def model():
x = genjax.normal(0.0, 1.0) @ "x"
y = genjax.normal(x, 0.01) @ "y"
return y

        key = jrand.key(0)
        key, sub_key = jrand.split(key)
        tr, _ = model.importance(sub_key, ChoiceMap.kw(y=3.0), ())
        request = HMC(Selection.at["x"], jnp.array(1e-2))
        editor = jax.jit(request.edit)

        # First, try moving x and test for correctness.
        old_x = tr.get_choices()["x"]
        old_y = tr.get_choices()["y"]
        old_target_density = genjax.normal.logpdf(
            old_x, 0.0, 1.0
        ) + genjax.normal.logpdf(old_y, old_x, 0.01)
        new_tr, fwd_w, _, _ = editor(key, tr, ())
        new_x = new_tr.get_choices()["x"]
        new_y = new_tr.get_choices()["y"]
        new_target_density = genjax.normal.logpdf(
            new_x, 0.0, 1.0
        ) + genjax.normal.logpdf(new_y, new_x, 0.01)
        assert fwd_w != 0.0
        # The change in the target scores corresponds to the non-momenta terms in the HMC alpha computation.
        assert (new_tr.get_score() - tr.get_score()) == pytest.approx(
            new_target_density - old_target_density, 1e-6
        )
        # The weight factors in the target score change and the momenta, so removing the change in the target scores should leave us with a non-zero contribution from the momenta.
        assert fwd_w - (new_tr.get_score() - tr.get_score()) != 0.0

        # Check for gradient convergence.
        new_tr = tr
        for _ in range(20):
            key, sub_key = jrand.split(key)
            new_tr, *_ = editor(sub_key, new_tr, ())
        assert new_tr.get_choices()["x"] == pytest.approx(3.0, 5e-3)

    def test_simple_scan_hmc(self):
        @genjax.gen
        def kernel(z, scanned_in):
            z = genjax.normal(z, 1.0) @ "x"
            _ = genjax.normal(z, 0.01) @ "y"
            return z, None

        key = jrand.key(0)
        key, sub_key = jrand.split(key)
        model = kernel.scan(n=10)
        vchm = ChoiceMap.empty().at["y"].set(3.0 * jnp.ones(10))
        tr, _ = model.importance(sub_key, vchm, (0.0, None))
        request = HMC(Selection.at["x"], jnp.array(1e-2))
        editor = jax.jit(request.edit)
        new_tr = tr
        for _ in range(50):
            key, sub_key = jrand.split(key)
            new_tr, *_ = editor(sub_key, new_tr, Diff.no_change((0.0, None)))
        assert new_tr.get_choices()[:, "x"] == pytest.approx(3.0, 8e-3)

    @pytest.mark.skip(reason="needs more work")
    def test_hmm_hmc(self):
        @genjax.gen
        def simulate_motion_step(carry, scanned_in):
            (pos, pos_noise, obs_noise, dt) = carry
            new_latent_position = genjax.mv_normal_diag(pos, pos_noise) @ "pos"
            _ = genjax.mv_normal_diag(new_latent_position, obs_noise) @ "obs_pos"
            return (
                new_latent_position,
                pos_noise,
                obs_noise,
                dt,
            ), new_latent_position

        @genjax.gen
        def simple_hmm(position_noise, observation_noise, dt):
            initial_y_pos = genjax.normal(0.5, 0.01) @ "init_pos"
            initial_position = jnp.array([0.0, initial_y_pos])
            _ = (
                genjax.mv_normal_diag(initial_position, observation_noise)
                @ "init_obs_pos"
            )
            _, tracks = (
                simulate_motion_step.scan(n=10)(
                    (
                        initial_position,
                        position_noise,
                        observation_noise,
                        dt,
                    ),
                    None,
                )
                @ "tracks"
            )
            return jnp.vstack([initial_position, tracks])

        # Simulate ground truth from the model.
        key = jrand.key(0)
        key, sub_key = jax.random.split(key)
        ground_truth = simple_hmm.simulate(
            sub_key,
            (jnp.array([1e-1, 1e-1]), jnp.array([1e-1, 1e-1]), 0.1),
        )

        # Create an initial importance sample.
        obs = ChoiceMap.empty()
        obs = obs.at["tracks", :, "obs_pos"].set(
            ground_truth.get_choices()["tracks", :, "obs_pos"]
        )
        obs = obs.at["init_obs_pos"].set(ground_truth.get_choices()["init_obs_pos"])
        key, sub_key = jax.random.split(key)
        init_tr, _ = simple_hmm.importance(
            sub_key,
            obs,
            (jnp.array([1e-1, 1e-1]), jnp.array([1e-1, 1e-1]), 0.1),
        )

        def _rejuvenation(eps):
            def _inner(carry, _):
                (key, tr) = carry
                key, sub_key = jax.random.split(key)
                request = HMC(Selection.at["init_pos"], eps)
                new_tr, w, _, _ = request.edit(
                    sub_key, tr, Diff.no_change(tr.get_args())
                )
                key, sub_key = jax.random.split(key)
                check = jnp.log(genjax.uniform.sample(sub_key, 0.0, 1.0)) < w
                tr = jtu.tree_map(
                    lambda v1, v2: jnp.where(check, v1, v2),
                    new_tr,
                    tr,
                )
                request = HMC(Selection.at["tracks", ..., "pos"], eps)
                key, sub_key = jax.random.split(key)
                new_tr, w, _, _ = request.edit(
                    sub_key, tr, Diff.no_change(tr.get_args())
                )
                key, sub_key = jax.random.split(key)
                check = jnp.log(genjax.uniform.sample(sub_key, 0.0, 1.0)) < w
                tr = jtu.tree_map(
                    lambda v1, v2: jnp.where(check, v1, v2),
                    new_tr,
                    tr,
                )
                return (key, tr), None

            return _inner

        def rejuvenation(length: int):
            def inner(key, tr, eps):
                (_, new_tr), _ = jax.lax.scan(
                    _rejuvenation(eps),
                    (key, tr),
                    length=length,
                )
                return new_tr

            return inner

        # Run MH with HMC.
        key, sub_key = jrand.split(key)
        rejuvenator = jax.jit(rejuvenation(3000))
        new_tr = rejuvenator(sub_key, init_tr, jnp.array(1e-4))
        assert init_tr.get_choices()["tracks", 0, "pos"] != pytest.approx(
            ground_truth.get_choices()["tracks", 0, "pos"], 1e-5
        )
        assert init_tr.get_choices()["tracks", -1, "pos"] != pytest.approx(
            ground_truth.get_choices()["tracks", -1, "pos"], 1e-5
        )
        assert new_tr.get_choices()["init_pos"] != pytest.approx(
            init_tr.get_choices()["init_pos"], 1e-5
        )
        assert new_tr.get_choices()["tracks", 0, "pos"] != pytest.approx(
            init_tr.get_choices()["tracks", 0, "pos"], 1e-5
        )
        assert new_tr.get_choices()["tracks", 0, "pos"] == pytest.approx(
            ground_truth.get_choices()["tracks", 0, "pos"], 5e-2
        )
        assert new_tr.get_choices()["tracks", -1, "pos"] == pytest.approx(
            ground_truth.get_choices()["tracks", -1, "pos"], 5e-2
        )

    def test_safe_hmc(self):
        @genjax.gen
        def submodel():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(x, 0.01) @ "y"
            return y

        @genjax.gen
        def model():
            _ = submodel() @ "x"
            _ = submodel() @ "y"

        key = jrand.key(0)
        key, sub_key = jrand.split(key)
        tr, _ = model.importance(sub_key, ChoiceMap.kw(y=3.0), ())
        request = StaticRequest(
            {"x": SafeHMC(Selection.at["x"], jnp.array(1e-2))},
        )
        editor = jax.jit(request.edit)
        key, sub_key = jrand.split(key)
        new_tr, w, *_ = editor(sub_key, tr, ())
        assert new_tr.get_choices()["x", "x"] != tr.get_choices()["x", "x"]
        assert w != 0.0

        # Compositional request _including_ HMC.
        request = StaticRequest(
            {
                "x": SafeHMC(Selection.at["x"], jnp.array(1e-2)),
                "y": StaticRequest({
                    "x": Regenerate(Selection.all()),
                    "y": Update(C.choice(3.0)),
                }),
            },
        )
        editor = jax.jit(request.edit)
        key, sub_key = jrand.split(key)
        new_tr, w, *_ = editor(sub_key, tr, ())
        assert new_tr.get_choices()["x", "x"] != tr.get_choices()["x", "x"]
        assert new_tr.get_choices()["y", "x"] != tr.get_choices()["y", "x"]
        assert w != 0.0

        request = StaticRequest(
            {"x": SafeHMC(Selection.at["y"], jnp.array(1e-2))},
        )
        editor = jax.jit(request.edit)
        key, sub_key = jrand.split(key)
        with pytest.raises(Exception):
            new_tr, w, *_ = editor(sub_key, tr, ())

class TestDiffCoercion:
def test_diff_coercion(self):
@genjax.gen
def simple_normal():
y1 = genjax.normal(0.0, 1.0) @ "y1"
y2 = genjax.normal(y1, 1.0) @ "y2"
return y1 + y2

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = simple_normal.simulate(sub_key, ())

        # Test that DiffCoercion.edit is being
        # properly used compositionally.
        def assert_no_change(v):
            assert Diff.static_check_no_change(v)
            return v

        request = StaticRequest({
            "y1": Regenerate(Selection.all()),
            "y2": DiffAnnotate(
                EmptyRequest(),
                argdiff_fn=assert_no_change,
            ),
        })

        with pytest.raises(Exception):
            request.edit(key, tr, ())

        # Test equivalent between requests which use
        # DiffCoercion in trivial ways.
        unwrapped_request = StaticRequest({
            "y1": Regenerate(Selection.all()),
        })
        wrapped_request = StaticRequest({
            "y1": Regenerate(Selection.all()).contramap(assert_no_change),
            "y2": EmptyRequest().map(assert_no_change),
        })
        _, w, _, _ = unwrapped_request.edit(key, tr, ())
        _, w_, _, _ = wrapped_request.edit(key, tr, ())
        assert w == w_

================================================
FILE: tests/inference/test_smc.py
================================================

# Copyright 2024 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import logsumexp

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import SelectionBuilder as S
from genjax.\_src.core.typing import Any
from genjax.\_src.inference.sp import Target

def logpdf(v):
return lambda c, \*args: v.assess(C.v(c), args)[0]

class TestSMC:
def test\*exact_flip_flip_trivial(self):
@genjax.gen
def flip_flip_trivial():

-   = genjax.flip(0.5) @ "x"
    \_ = genjax.flip(0.7) @ "y"

            def flip_flip_exact_log_marginal_density(target: genjax.Target[Any]):
                y = target.constraint.get_submap("y")
                return genjax.flip.assess(y, (0.7,))[0]

            key = jax.random.key(314159)
            inference_problem = genjax.Target(flip_flip_trivial, (), C["y"].set(True))

            # Single sample IS.
            Z_est = genjax.inference.smc.Importance(
                inference_problem
            ).log_marginal_likelihood_estimate(key)
            Z_exact = flip_flip_exact_log_marginal_density(inference_problem)
            assert Z_est == pytest.approx(Z_exact, 1e-1)

            # K-sample sample IS.
            Z_est = genjax.inference.smc.ImportanceK(
                inference_problem, k_particles=1000
            ).log_marginal_likelihood_estimate(key)
            Z_exact = flip_flip_exact_log_marginal_density(inference_problem)
            assert Z_est == pytest.approx(Z_exact, 1e-3)

        def test_exact_flip_flip(self):
            @genjax.gen
            def flip_flip():
                v1 = genjax.flip(0.5) @ "x"
                p = jax.lax.cond(v1, lambda: 0.9, lambda: 0.3)
                _ = genjax.flip(p) @ "y"

            def flip_flip_exact_log_marginal_density(target: genjax.Target[Any]):
                y = target["y"]
                x_prior = jnp.array([
                    logpdf(genjax.flip)(True, 0.5),
                    logpdf(genjax.flip)(False, 0.5),
                ])
                y_likelihood = jnp.array([
                    logpdf(genjax.flip)(y, 0.9),
                    logpdf(genjax.flip)(y, 0.3),
                ])
                y_marginal = logsumexp(x_prior + y_likelihood)
                return y_marginal

            key = jax.random.key(314159)
            inference_problem = genjax.Target(flip_flip, (), C["y"].set(True))

            # K-sample IS.
            Z_est = genjax.inference.smc.ImportanceK(
                inference_problem, k_particles=2000
            ).log_marginal_likelihood_estimate(key)
            Z_exact = flip_flip_exact_log_marginal_density(inference_problem)
            assert Z_est == pytest.approx(Z_exact, 1e-1)

        def test_non_marginal_target(self):
            @genjax.gen
            def model():
                idx = genjax.categorical(probs=[0.5, 0.25, 0.25]) @ "idx"
                # under the prior, 50% chance to be in cluster 1 and 50% chance to be in cluster 2.
                means = jnp.array([0.0, 10.0, 11.0])
                vars = jnp.array([1.0, 1.0, 1.0])
                x = genjax.normal(means[idx], vars[idx]) @ "x"
                y = genjax.normal(means[idx], vars[idx]) @ "y"
                return x, y

            marginal_model = model.marginal(
                selection=S["x"] | S["y"]
            )  # This means we are projection onto the variables x and y, marginalizing out the rest

            obs1 = C["x"].set(1.0)
            with pytest.raises(TypeError):
                Target(marginal_model, (), obs1)

================================================
FILE: tests/inference/test_vi.py
================================================

# Copyright 2024 MIT Probabilistic Computing Project

#

# Licensed under the Apache License, Version 2.0 (the "License");

# you may not use this file except in compliance with the License.

# You may obtain a copy of the License at

#

# https://www.apache.org/licenses/LICENSE-2.0

#

# Unless required by applicable law or agreed to in writing, software

# distributed under the License is distributed on an "AS IS" BASIS,

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# See the License for the specific language governing permissions and

# limitations under the License.

import jax
import pytest

import genjax
from genjax import ChoiceMapBuilder as C

class TestVI:
def test\*normal_normal_tight_variance(self):
@genjax.gen
def model(v):
mu = genjax.normal(0.0, 10.0) @ "mu"

-   = genjax.normal(mu, 0.1) @ "v"

          @genjax.marginal()
          @genjax.gen
          def guide(target):
              (v,) = target.args
              _ = genjax.vi.normal_reparam(v, 0.1) @ "mu"

          key = jax.random.key(314159)
          elbo_grad = genjax.vi.ELBO(
              guide, lambda v: genjax.Target(model, (v,), C["v"].set(3.0))
          )
          v = 0.1
          jitted = jax.jit(elbo_grad)
          for _ in range(200):
              (v_grad,) = jitted(key, (v,))
              v -= 1e-3 * v_grad
          assert v == pytest.approx(3.0, 5e-2)
