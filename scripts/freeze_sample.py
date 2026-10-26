"""Build the M2 sample from a real ASR run, BEFORE inspecting evaluated outputs.

Annotations are coding-agent authored from the supplied conspect and ASR; they
are NOT independently human-verified gold labels. See eval/README.md.
Run from the repository root: python scripts/freeze_sample.py FULL_TRANSCRIPT.json
"""

import hashlib
import json
import sys
from pathlib import Path


# Short aliases intentionally measure lexical mention, not semantic correctness.
# References: user-supplied Vetrov conspect, Lecture 1, PDF pp. 4–10;
# seminar corrections in the user-supplied recording for the last two tasks.
DEFINITIONS = [
    (184, 332, "conditional probability and independence", "4, section 1.1", [
        ("conditional distribution", ["conditional", "conditioning"]),
        ("joint over marginal", ["joint"]), ("marginal distribution", ["marginal"]),
        ("independence removes information", ["independent", "independence"])]),
    (332, 402, "the variance interpretation of observing another variable", "ASR 05:31–06:41; related to 1.1", [
        ("unconditional variance", ["unconditional variance", "variance of x"]),
        ("conditional variance", ["conditional variance"]),
        ("averaging over Y", ["average", "averaged", "expectation", "expected", "mean"]),
        ("information reduces uncertainty", ["information", "uncertainty"])]),
    (367, 515, "the product/chain rule and alternative variable orderings", "4, section 1.1", [
        ("joint distribution", ["joint"]), ("product decomposition", ["product", "multiply", "multiplication"]),
        ("conditional factors", ["conditional"]), ("different orderings", ["order", "ordering", "orderings", "orders"])]),
    (547, 764, "marginalization and its expectation interpretation", "5, section 1.1", [
        ("marginalization", ["marginalization", "marginal", "unconditional"]),
        ("integrating variables out", ["integrate", "integrating", "integration", "integrated", "integral"]),
        ("expectation", ["expectation", "averaging", "average"]),
        ("joint distribution", ["joint"])]),
    (764, 928, "predicting an unobserved variable from a joint probabilistic model", "5, section 1.1", [
        ("joint model", ["joint"]), ("conditional prediction", ["conditional"]),
        ("observed Y", ["observed", "observe", "observations"]),
        ("eliminating irrelevant Z", ["integrate", "integrating", "marginalize", "marginalizing", "marginalization"])]),
    (928, 1064, "Bayes' theorem: prior, likelihood, posterior and evidence", "5, section 1.1", [
        ("prior", ["prior"]), ("likelihood", ["likelihood"]),
        ("posterior", ["posterior"]), ("normalizing evidence", ["evidence", "normalization", "normalizing", "normalisation"])]),
    (1188, 1323, "the maximum-likelihood objective and why log likelihood is used", "6, section 1.2", [
        ("independent observations", ["independent", "iid", "i i d"]),
        ("likelihood as parameter function", ["parameter", "parameters", "theta"]),
        ("product of densities", ["product"]), ("log converts product to sum", ["log", "logarithm", "logarithmic"])]),
    (1323, 1698, "the asymptotic properties of maximum-likelihood estimators and their limits", "6–7, section 1.2", [
        ("asymptotic unbiasedness", ["unbiased", "unbiasedness"]), ("consistency", ["consistency", "consistent"]),
        ("asymptotic normality", ["normal", "normality"]), ("efficiency", ["efficient", "efficiency"])]),
    (1698, 1933, "why the two-heads coin example exposes small-sample MLE limitations", "6–8, section 1.2; coin example in ASR", [
        ("empirical head frequency", ["frequency", "proportion", "k n", "k over n", "k/n"]),
        ("two trials", ["two", "twice", "2"]), ("probability one", ["one", "1", "100"]),
        ("small sample caveat", ["small", "limited", "insufficient"])]),
    (1933, 2094, "Bayesian parameter estimation versus a point estimate", "6–7, section 1.2", [
        ("prior", ["prior"]), ("likelihood", ["likelihood"]), ("posterior", ["posterior"]),
        ("point estimate contrast", ["point estimate", "point estimation", "point estimates"])]),
    (2094, 2364, "the lecture's frequentist versus Bayesian interpretation of randomness", "7, section 1.2", [
        ("objective uncertainty", ["objective"]), ("subjective ignorance", ["subjective"]),
        ("deterministic physical process", ["deterministic", "mechanics", "physics"]),
        ("missing information", ["information", "knowledge", "ignorance"])]),
    (2446, 2790, "sample size relative to parameter count, and Bayesian updating with no observations", "7–8, section 1.2", [
        ("parameter dimension", ["dimension", "parameters", "parameter"]),
        ("overfitting", ["overfitting", "overfit"]), ("no observations", ["zero", "0", "no observations", "no data"]),
        ("posterior equals prior, not necessarily uniform", ["prior"])]),
    (2829, 3202, "combining observations sequentially in the elephant example", "8, section 1.3; elephant example in ASR", [
        ("different information sources", ["sources", "source", "sensors", "tactile", "acoustic"]),
        ("posterior reused as prior", ["prior"]), ("likelihood for new observation", ["likelihood"]),
        ("posterior preserves combined information", ["posterior"])]),
    (3202, 3476, "the large-data limit and practical advantages of Bayesian methods", "8, sections 1.2–1.3", [
        ("MLE limiting case", ["maximum likelihood", "mle"]), ("posterior concentration", ["delta", "converges", "convergence"]),
        ("regularization", ["regularize", "regularization", "regularisation"]),
        ("uncertainty estimates", ["uncertainty", "uncertain"])]),
    (3476, 3824, "how Bayesian reasoning extends Boolean logic using the football example", "8, section 1.4", [
        ("modus ponens", ["modus ponens"]), ("Boolean limitation", ["boolean", "binary"]),
        ("probabilistic rather than certain reverse inference", ["probability", "probabilities", "probabilistic"]),
        ("alternative explanation for celebration", ["alternative", "contract", "mbappe", "mbappé"])]),
    (3824, 4286, "the alarm example's variables, causal graph, assumptions and conditional probabilities", "9, section 1.5", [
        ("thief", ["thief", "burglary", "burglar"]), ("earthquake", ["earthquake", "earthquakes"]),
        ("radio announcement", ["radio"]), ("independence assumption", ["independent", "independence"])]),
    (4561, 5027, "the alarm example's posterior calculations and the effect of changing the burglary prior", "9, section 1.5; ASR has conflicting numbers", [
        ("false alarm marginalization", ["0.001", "10^-3", "10 to the negative 3", "10⁻³", "10^{-3}"]),
        ("initial posterior about one sixth", ["1/6", "one sixth", "one over six", "17", "16.7", "16.67"]),
        ("higher burglary prior", ["prior"]),
        ("updated posterior about two thirds", ["2/3", "two thirds", "two-thirds", "67", "66.7", "0.67", "0.667"])]),
    (5027, 5374, "radio evidence, the low burglary posterior, and model misspecification", "10, section 1.5", [
        ("radio evidence", ["radio"]), ("posterior about two percent", ["2", "two percent", "0.02"]),
        ("incorrect independence assumption", ["independent", "independence"]),
        ("correct inference can use a wrong model", ["model"])]),
    (7070, 7360, "the corrected rare-disease test calculation; distinguish student errors from the final explanation", "Seminar 1:58–2:02; related to Bayes in PDF p.5", [
        ("false positive rate is one percent, not 99 percent", ["0.01", "one percent", "1"]),
        ("rare disease base rate", ["rare", "rarity", "base rate", "prevalence", "10,000", "10000"]),
        ("posterior about one percent", ["1", "one percent", "0.01"]),
        ("positive test is not certainty of disease", ["positive"])]),
    (11400, 12027, "the Gamma-Poisson Bayesian update, retaining only the corrected final derivation", "Seminar 3:10–3:20; PDF pp.11–12, section 2.1", [
        ("Gamma prior and posterior", ["gamma"]), ("Poisson likelihood", ["poisson"]),
        ("updated shape", ["a plus", "a +", "a_n", "shape", "α", "alpha"]),
        ("updated rate", ["b plus", "b +", "b_n", "rate", "β", "beta"])]),
]


def main():
    source = Path(sys.argv[1])
    data = json.loads(source.read_text())
    segments = data["segments"]
    boundaries = [s["start"] for s in segments] + [segments[-1]["end"]]
    tasks = []
    for i, (start, end, topic, reference, concepts) in enumerate(DEFINITIONS, 1):
        start = min(boundaries, key=lambda t: abs(t - start))
        end = min(boundaries, key=lambda t: abs(t - end))
        tasks.append({"id": f"t{i:02d}", "start": start, "end": end,
                      "task": f"Make concise, timestamped study notes on {topic}. "
                              "Preserve key definitions, reasoning and examples from this interval; flag unclear speech.",
                      "reference": reference,
                      "concepts": [{"name": name, "aliases": aliases} for name, aliases in concepts]})
    excerpt = {"source": "https://www.youtube.com/watch?v=XwR767IfQHI", "asr": data["asr"],
               "full_asr_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "segments": [s for s in segments if any(t["start"] <= s["start"] < t["end"] for t in tasks)]}
    folder = Path("eval")
    folder.mkdir(exist_ok=True)
    path = folder / "lecture_excerpt.json"
    if path.exists() or (folder / "tasks.json").exists():
        raise SystemExit("Sample already frozen; do not overwrite it.")
    path.write_text(json.dumps(excerpt, ensure_ascii=False, indent=2) + "\n")
    spec = {"version": 1, "transcript": "lecture_excerpt.json",
            "transcript_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "annotation_status": "Coding-agent authored from supplied reference/ASR; not independently human verified.",
            "tasks": tasks}
    (folder / "tasks.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    print(f"Frozen {len(tasks)} tasks and {len(excerpt['segments'])} ASR segments.")


if __name__ == "__main__":
    main()
