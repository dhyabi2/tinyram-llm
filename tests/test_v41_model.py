"""v41_model: calibration against the measured Qwen3-30B Mac runs, monotonic properties, a control that must fail. No model is run."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "tools")); from harness import T
import v41_model as m
def run():
    t = T("T21 v41_model projection")
    t.check("calibration: worst error against the 7 measured Qwen3 configurations is under 15%", m_validate_quiet() < 0.15)
    a, _ = m.tok_s("qwen3", hit_override=0.59); b, _ = m.tok_s("qwen3", hit_override=0.80)
    t.check("higher hit rate is never slower", b > a)
    r1, _ = m.tok_s("v41", ram=16, rate=1.4); r2, _ = m.tok_s("v41", ram=16, rate=3.0)
    t.check("faster reads are never slower", r2 >= r1)
    t.check("fewer experts per token is never slower", m.tok_s("v41", K=4, ram=16)[0] >= m.tok_s("v41", K=6, ram=16)[0])
    t.check("more RAM is never slower", m.tok_s("v41", ram=32)[0] >= m.tok_s("v41", ram=16)[0])
    ceiling = 1.0 / (16e9 * 2 / 1e9 / m.G_CPU)
    t.check("no setting at 6 experts exceeds the compute ceiling %.2f tok/s" % ceiling, all(m.tok_s("v41", ram=64, rate=50.0, hit_override=1.0, route=b)[0] <= ceiling + 1e-9 for b in (0, 1, 3)))
    t.check("plain mmap is slower than the pool at the same RAM", m.tok_s("v41", plain=True, ram=16)[0] < m.tok_s("v41", ram=16)[0])
    t.check("bias 3.0 removes more misses than bias 1.0", m.tok_s("v41", ram=16, route=3)[1] > m.tok_s("v41", ram=16, route=1)[1] > m.tok_s("v41", ram=16)[1])
    t.check("cache size is never negative", m.cache_gb(m.GEOM["v41"], 4) == 0.0)
    t.check("risk score orders 4 of 6 experts above 5 of 6 above 6", m.risk(dict(K=4, bits=4.3, route=0, spec=0)) > m.risk(dict(K=5, bits=4.3, route=0, spec=0)) > m.risk(dict(K=6, bits=4.3, route=0, spec=0)))
    saved = m.R_EFF; m.R_EFF = 20.0
    t.check("control: a wrong read rate (20 GB/s) breaks the calibration (must exceed 15% error)", m_validate_quiet() > 0.15); m.R_EFF = saved
    t.check("after restoring the rate the calibration passes again", m_validate_quiet() < 0.15)
    return t
def m_validate_quiet():
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()): return m.validate()
if __name__ == "__main__": sys.exit(0 if run().summary() else 1)
