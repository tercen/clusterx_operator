//! Student-t quantiles for the generalised ESD test (`detect_anoms_sd` calls
//! R's `qt` on every iteration).
//!
//! R's `qt` is AS 91 with Newton polishing; this computes the same quantile
//! by inverting the t CDF, whose upper tail is a regularised incomplete beta
//! function evaluated with the standard continued fraction (Lentz), then
//! bisecting to machine precision. Agreement with R's `qt` is at the 1e-11
//! level (checked against `tests/parity/qt_grid.csv`, dumped from R at
//! %.17g), which is orders of magnitude finer than any margin the ESD test
//! actually decides on.

/// Regularised incomplete beta function I_x(a, b).
fn betainc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let front = lbeta.exp();
    // Continued fraction (Numerical Recipes 6.4), switched at x > (a+1)/(a+b+2)
    let switch = (a + 1.0) / (a + b + 2.0);
    if x < switch {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_IT: usize = 300;
    const EPS: f64 = 3e-16;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_IT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let mut aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// ln Γ(z) — Lanczos, g = 7, matching the standard 9-term coefficients.
fn ln_gamma(z: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_13,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_5e-7,
    ];
    if z < 0.5 {
        // reflection
        std::f64::consts::PI
            / ((std::f64::consts::PI * z).sin() * ln_gamma(1.0 - z))
                .abs()
                .ln()
    } else {
        let z = z - 1.0;
        let mut x = C[0];
        for (i, &c) in C.iter().enumerate().skip(1) {
            x += c / (z + i as f64);
        }
        let t = z + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
    }
}

/// Two-sided? No — R's `qt(p, df)` is the plain inverse CDF at `p`.
pub fn qt(p: f64, df: f64) -> f64 {
    if !((0.0..1.0).contains(&p) && df > 0.0) {
        return f64::NAN;
    }
    // The t CDF: F(t) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2) for t > 0,
    // mirrored for t < 0. Invert by bisection; 200 halvings exhaust f64.
    let cdf = |t: f64| -> f64 {
        if t == 0.0 {
            return 0.5;
        }
        let x = df / (df + t * t);
        let tail = 0.5 * betainc(df / 2.0, 0.5, x);
        if t > 0.0 { 1.0 - tail } else { tail }
    };
    // bracket
    let (mut lo, mut hi) = (-1.0f64, 1.0f64);
    while cdf(lo) > p {
        lo *= 2.0;
    }
    while cdf(hi) < p {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break;
        }
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::qt;

    #[test]
    fn known_values() {
        // R 4.3.3: qt(c(.9, .95, .975, .995), 10). The port agrees with R's
        // AS 91 to ~1e-11 relative (the Lanczos lgamma limits the continued
        // fraction), far inside any margin the ESD test decides on.
        for (p, df, want) in [
            (0.90, 10.0, 1.372_183_641_110_336),
            (0.95, 10.0, 1.812_461_122_811_676),
            (0.975, 10.0, 2.228_138_851_986_274),
            (0.995, 10.0, 3.169_272_672_616_95),
        ] {
            let got = qt(p, df);
            assert!(
                (got - want).abs() / want.abs() < 2e-11,
                "qt({p}, {df}) = {got}, want {want}"
            );
        }
    }
}
