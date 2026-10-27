# binference

**Population and hierarchical inference with PDFs.**

Many objects are each measured with noise: galaxies, merger events, patients,
classified documents. From each object's likelihood, posterior, samples, binned count or
classifier output, binference infers the distribution of the population the objects come
from, with calibrated uncertainty, and each object's posterior under that population.

![Population posteriors for a mock catalogue](https://raw.githubusercontent.com/joshspeagle/binference/main/docs/_static/quickstart.png)

## Install

binference is in alpha. Until it is released on PyPI, install it from GitHub:

```bash
pip install "binference[plot] @ git+https://github.com/joshspeagle/binference"
```

The [installation page](https://joshspeagle.github.io/binference/start/install.html)
explains the extras.

## Example

```python
import binference

bins = binference.Bins.linear(0.0, 3.0, 15)

# a simulated stand-in for your arrays: measured values and their Gaussian errors
mock = binference.simulate.mock_1d(2000, bins, rng=42)
z_obs, z_err = mock.measurements["mean"], mock.measurements["sigma"]

lk = binference.Likelihoods.from_gaussian(z_obs, z_err, bins)
result = binference.infer(lk, rng=1)  # method="auto" chooses the engine
print(result.summary())
```

## Documentation

The [documentation](https://joshspeagle.github.io/binference/) is the home of
everything else:

- [What binference does](https://joshspeagle.github.io/binference/start/what-it-does.html),
  with the same model under the names of several fields, and
  [whether it suits your data](https://joshspeagle.github.io/binference/start/is-it-right.html);
- the [quick start](https://joshspeagle.github.io/binference/start/quickstart.html) and the
  [tutorials](https://joshspeagle.github.io/binference/tutorials/index.html);
- the [guides](https://joshspeagle.github.io/binference/guides/index.html), the
  [theory](https://joshspeagle.github.io/binference/how-it-works/theory.html), the
  [fields](https://joshspeagle.github.io/binference/fields/index.html) and the
  [API reference](https://joshspeagle.github.io/binference/reference/api/index.html);
- [how to cite binference](https://joshspeagle.github.io/binference/reference/citing.html).

To contribute, see [`CONTRIBUTING.md`](https://github.com/joshspeagle/binference/blob/main/CONTRIBUTING.md).

## License

MIT; see [LICENSE](https://github.com/joshspeagle/binference/blob/main/LICENSE).
