#!/usr/bin/env python3
"""Gate a docs build on things Sphinx will not fail on by itself.

Run after building:

    python3 -m sphinx -b html -d _build/doctrees . _build/html 2>&1 | tee build.log
    python3 check_docs_build.py --log build.log --html _build/html

Why this exists, concretely. Two separate incidents shipped an API reference
that was partly empty while the build reported success:

* `iris/host/iris.py` imports `xio` at module load. Docs CI installs only
  `docs/sphinx/requirements.txt`, so every autodoc directive raised
  ModuleNotFoundError and the reference pages published as headings with no
  content. Sphinx logged 71 warnings and exited 0.
* A directive carried `:no-undoc-members:`, which is autoclass-only. On an
  autofunction Sphinx logged
  `ERROR: An option to autofunction is either unknown or has an invalid value`
  and then emitted the section heading with nothing under it. Exit code 0, job
  green, the function gone from the page.

So the log is worth failing on -- but it is not sufficient. A separate class of
defect produces perfectly valid HTML and zero log output: mock objects rendered
into signatures, bullet lists and code examples collapsed into run-on
paragraphs, members documented twice, members rendered with no description. A
build log scan finds none of those. They are only visible in the output.

Hence two layers: LOG for things Sphinx noticed and discarded, HTML for things
it never noticed.
"""

import argparse
import re
import sys
from pathlib import Path

# Warnings that are not this repo's fault and not worth failing on. Anything
# else in the log is fatal. Keep this list short and justify every entry --
# a long allowlist is how the xio failure stayed invisible for so long.
LOG_ALLOWLIST = [
    # rocm-docs-core looks the project up in an external registry.
    r"Current project 'Iris' not found in projects",
    # Four external inventories that 404 or serve a malformed header. Network
    # dependent, so failing on these would make CI flaky rather than strict.
    r"failed to reach any of the inventories",
    r"intersphinx inventory .* not (fetchable|readable)",
]


def scan_log(text):
    out = []
    for line in text.splitlines():
        if not re.search(r"\b(WARNING|ERROR|SEVERE)\b", line):
            continue
        if any(re.search(p, line) for p in LOG_ALLOWLIST):
            continue
        out.append(line.strip()[:160])
    return out


# --------------------------------------------------------------------------
# HTML checks. Each returns a list of offending items.
# --------------------------------------------------------------------------

def _article(html):
    m = re.search(r"<article[^>]*>(.*?)</article>", html, re.S)
    return m.group(1) if m else ""


def c_mock(a):
    """Mocked modules rendered into a signature, e.g.
    tracing (<MagicMock name='mock.constexpr' id='139779281861264'>)."""
    return re.findall(r"MagicMock name='[^']*'", a)


def c_collapsed_list(a):
    """A bullet list with no blank line before it, so RST folds the dashes
    into the preceding paragraph."""
    return [x for x in re.findall(r"<p>([^<]{25,}?)</p>", a)
            if re.search(r":\s+[-*]\s", x)]


def c_runon_code(a):
    """An Example written as bare indented text rather than >>> or a literal
    block, so consecutive statements render as one line."""
    return [x for x in re.findall(r"<p>([^<]{30,}?)</p>", a)
            if len(re.findall(r"=\s*\w+[.(]", x)) >= 2]


def c_undocumented(a):
    """An API object rendered with an empty body -- present on the page with
    no description at all."""
    out = []
    for m in re.finditer(r'<dl class="py (\w+)">(.*?)</dl>', a, re.S):
        kind, blk = m.group(1), m.group(2)
        dt = re.search(r'<span class="sig-name descname"><span class="pre">([^<]+)</span>', blk)
        for dd in re.findall(r"<dd>(.*?)</dd>", blk, re.S):
            if not re.sub(r"<[^>]+>|\s", "", dd):
                out.append("%s %s" % (kind, dt.group(1) if dt else "?"))
                break
    return out


def c_empty_section(a):
    """A section whose entire body is its own heading -- the directive under
    it rendered nothing."""
    out = []
    for m in re.finditer(r'<section id="([^"]+)">(.*?)</section>', a, re.S):
        sid, blk = m.group(1), m.group(2)
        if "<section" in blk:
            continue
        rest = re.sub(r"<h[1-6]>.*?</h[1-6]>", "", blk, flags=re.S)
        if not re.sub(r"<[^>]+>|\s|&#?\w+;", "", rest):
            out.append(sid)
    return out


def c_dup_member(a):
    """The same method rendered twice on one page, which happens when a page
    spells out automethod sections under an autoclass that already has
    members enabled."""
    heads = re.findall(r"<h3>([A-Za-z_]+\.[A-Za-z_]+)<a class=\"headerlink\"", a)
    return heads


def c_raw_markup(a):
    """Directive source leaking into the output instead of being executed."""
    return re.findall(r"(eval-rst|\.\.\s+auto(?:class|method|function)::)", a)


CHECKS = [
    ("mock-in-signature", c_mock),
    ("collapsed-list", c_collapsed_list),
    ("runon-example", c_runon_code),
    ("undocumented-member", c_undocumented),
    ("empty-section", c_empty_section),
    ("duplicate-member", c_dup_member),
    ("raw-markup", c_raw_markup),
]

# Fixtures captured verbatim from pages that really shipped these defects.
# An earlier version of this file used hand-written fixtures; one of them
# passed the potency gate while the check it guarded matched nothing real and
# reported a clean site with twenty defects on screen. Do not replace these
# with invented markup.
BAD = {
    "mock-in-signature":
        "<p>tracing (&lt;MagicMock name='mock.constexpr' id='139779281861264'&gt;)</p>",
    "collapsed-list":
        "<p>The context tensor has the format: - [cur_rank, num_ranks, heap_base_0] "
        "- If tracing=True: extracts trace buffer pointers</p>",
    "runon-example":
        "<p>view = make_tensor_view(ptr, M, N, stride_m, stride_n) tile_ptr, mask "
        "= view.tile_ptr(tile)</p>",
    "undocumented-member":
        '<dl class="py attribute"> <dt class="sig sig-object py"> '
        '<span class="sig-name descname"><span class="pre">ptr</span></span>'
        '<span class="property"><span class="p"><span class="pre">:</span></span>'
        '<span class="w"> </span><span class="pre">tensor</span></span></dt>'
        "<dd></dd></dl>",
    "empty-section":
        '<section id="make-tensor-view"> <h2>make_tensor_view'
        '<a class="headerlink" href="#make-tensor-view" title="Link">#</a></h2> </section>',
    "duplicate-member":
        '<h3>TensorView.tile_ptr<a class="headerlink" href="#x">#</a></h3>',
    "raw-markup":
        "<p>```{eval-rst} .. autoclass:: iris.mem.triton.types.Tile</p>",
}

# A correctly rendered section. Every check must stay silent on it, or it is
# flagging normal output and its hits mean nothing.
GOOD = (
    '<section id="tensorview"> <h2>TensorView<a class="headerlink" href="#t">#</a></h2>'
    '<dl class="py class"> <dt class="sig sig-object py">'
    '<span class="sig-name descname"><span class="pre">TensorView</span></span></dt>'
    "<dd><p>TensorView storing pointer and tensor metadata.</p></dd></dl>"
    "<p>Compute tile pointer and mask using the stored dimensions and strides.</p>"
    "</section>"
)


def potency():
    """Every check must fire on real defective markup and stay quiet on good
    markup. A check that cannot do both is reported as dead rather than
    silently contributing a zero."""
    dead = []
    for name, fn in CHECKS:
        if not fn(BAD[name]) or fn(GOOD):
            dead.append(name)
    return dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, help="sphinx build log")
    ap.add_argument("--html", type=Path, help="built html directory")
    ap.add_argument("--selftest", action="store_true",
                    help="run the potency gate and exit")
    args = ap.parse_args()

    dead = potency()
    print("potency: %d/%d checks can detect their own defect"
          % (len(CHECKS) - len(dead), len(CHECKS)))
    if dead:
        print("FAIL: these checks are inert: %s" % dead)
        return 2
    if args.selftest:
        return 0

    failures = 0

    if args.log:
        bad = scan_log(args.log.read_text(errors="replace"))
        print("\nlog: %d non-allowlisted warnings/errors" % len(bad))
        for b in bad[:25]:
            print("   %s" % b)
        failures += len(bad)

    if args.html:
        pages = sorted(args.html.rglob("*.html"))
        print("\nhtml: %d pages" % len(pages))
        for p in pages:
            a = _article(p.read_text(errors="replace"))
            if not a:
                continue
            hits = {n: fn(a) for n, fn in CHECKS}
            hits = {n: v for n, v in hits.items() if v}
            if hits:
                rel = p.relative_to(args.html)
                print("   %s" % rel)
                for n, v in hits.items():
                    print("      %-22s %d  e.g. %s"
                          % (n, len(v), re.sub(r"\s+", " ", str(v[0]))[:70]))
                    failures += len(v)

    print("\n%s (%d issues)" % ("FAIL" if failures else "PASS", failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
