# Parity fixtures

Golden fixtures generated from the **reference implementation** — the ClusterX
package the R operator depends on (JinmiaoChenLab/ClusterX 0.99.1, vendored in
`reference/`) — by `generate_goldens.R`. The fixtures are committed; regenerate
only when the reference changes:

```sh
Rscript generate_goldens.R   # needs the plyr, pdist and Rtsne R packages
```

Numeric values are written with `sprintf("%.17g")` so an exact port can be
compared exactly (`write.csv`'s 15 digits make an exact port look wrong on
ties). `tests/parity.rs` in the crate root consumes them. What each fixture
pins is documented in the header of that test file and in
`../../CLAUDE.md` (§ Parity).

Note: the tiny-input section was removed from the generator because the R
reference loops forever there (`estimateDc`'s neighbour-rate band is
unreachable for a few points); the port refuses such shapes with an error
instead, asserted in `the_dc_search_refuses_shapes_the_reference_loops_forever_on`.
