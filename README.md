# binference

**Population and hierarchical inference with PDFs.**

Many objects are each measured with noise: galaxies, merger events, patients,
classified documents. binference reads each object's likelihood, posterior, samples,
binned measurement or classifier output. From these it infers the population the objects
come from, with calibrated uncertainty, and each object's posterior under that
population.

![Population posteriors for a mock catalogue][figure]

[figure]: https://raw.githubusercontent.com/joshspeagle/binference/main/docs/_static/quickstart.png

*The population inferred from 2000 simulated objects, under the default Dirichlet prior
(left) and a Gaussian-process prior (right). Each panel shows the posterior median, its
68% and 95% intervals, the true density (dashed) and the stacked estimate (dotted).*

## Install

```bash
pip install "binference[plot] @ git+https://github.com/joshspeagle/binference"
```

The [installation page](https://joshspeagle.github.io/binference/start/install.html)
lists the extras and the supported Python versions.

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

- [Quick start][quickstart]
- [Is binference right for my data?][is-it-right]
- [Tutorials][tutorials]
- [Full documentation][docs]

[quickstart]: https://joshspeagle.github.io/binference/start/quickstart.html
[is-it-right]: https://joshspeagle.github.io/binference/start/is-it-right.html
[tutorials]: https://joshspeagle.github.io/binference/tutorials/index.html
[docs]: https://joshspeagle.github.io/binference/

To contribute, see [`CONTRIBUTING.md`][contributing].

[contributing]: https://github.com/joshspeagle/binference/blob/main/CONTRIBUTING.md

## License

MIT; see [LICENSE](https://github.com/joshspeagle/binference/blob/main/LICENSE).
