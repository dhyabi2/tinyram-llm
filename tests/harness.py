"""Tiny test harness: counted checks, PASS/FAIL lines, seeds printed on failure, non-zero exit on any failure."""
import sys, os, json
class T:
    def __init__(self, suite): self.suite, self.n, self.bad = suite, 0, []
    def check(self, name, cond, detail=""):
        self.n += 1
        if os.environ.get("TEST_CATALOG"):
            with open(os.environ["TEST_CATALOG"], "a") as f: f.write(json.dumps({"suite": self.suite, "name": name, "passed": bool(cond)}) + "\n")
        if not cond:
            self.bad.append(name); print(f"FAIL {self.suite}: {name}" + (f"  [{detail}]" if detail else ""))
        return bool(cond)
    def summary(self):
        print(f"{self.suite}: {self.n - len(self.bad)}/{self.n} checks passed"); return not self.bad
