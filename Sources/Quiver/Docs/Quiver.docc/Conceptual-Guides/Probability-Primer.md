# Probability Primer

Reason about uncertainty by computing probabilities, updating beliefs, and weighing evidence with Quiver.

## Overview

Probability is the practice of putting a number on uncertainty. Every event sits somewhere on a scale from impossible to certain, and probability gives that scale a unit: a value in `[0, 1]`, where `0` means the event cannot happen and `1` means it must. The interesting questions in probability are rarely about a single number. They are about how that number changes when new evidence arrives. A test result, a sensor reading, a flagged email: each is a piece of evidence that should move what is believed about the underlying truth. Bayes' theorem is the rule that turns belief plus evidence into updated belief, and it is what most of this primer builds toward.

### From counts to probability

The first probability a developer computes is usually empirical. Given a list of past events, the **probability** of a specific outcome is the fraction of observations matching it. If 84 emails out of 420 received last month were spam, the empirical probability of spam in this account's inbox is `84 / 420 = 0.20`. Quiver exposes this directly on any `[Double]` array:

```swift
import Quiver

// Past message labels: 1.0 = spam, 0.0 = legitimate.
let labels = [/* 420 observed messages */]

let pSpam = labels.probability(of: 1.0)         // 0.20
```

For the full picture across every distinct outcome, `frequencyDistribution()` returns a dictionary whose values sum to `1.0`:

```swift
let distribution = labels.frequencyDistribution()
// [1.0: 0.20, 0.0: 0.80]
```

The result is a **probability distribution**: an assignment of probabilities to every distinct outcome, summing to `1.0`. This is the empirical foundation on which everything else in this primer rests. See <doc:Statistics-Primer> for the descriptive vocabulary (central tendency, spread, and the frequency surface) that this empirical distribution sits inside.

> Tip: For categorical counts and the underlying frequency surface, see <doc:Frequency-Tables>. Empirical probabilities are the building block for **priors** later in this primer.

### Joint and conditional probability

Two events at once produce a **joint probability**, written `P(A ∩ B)`: the probability that both events occur. If 20% of messages are spam and 60% of spam contains the phrase "verify your account," then the joint probability of "spam *and* contains the phrase" is `0.20 × 0.60 = 0.12`.

A **conditional probability**, written `P(A | B)`, asks how likely `A` is once `B` is known. Restricting the universe to messages containing "verify your account," the fraction that are spam, `P(spam | "verify your account")`, is rarely equal to `P(spam)` alone. Evidence narrows the universe, and the probability inside that narrowed universe is the answer the developer wants. Bayes' theorem inverts the question.

### Independence

Two events are **independent** when conditioning on one tells nothing about the other: `P(A | B) = P(A)`. Knowing the day of the week tells nothing about whether a coin will land heads, so the two are independent. Knowing a message contains "verify your account" tells a lot about whether it is spam, so those two are not. Independence is the special case where conditional probability collapses back to the unconditional value, and it is what makes the Naive Bayes classifier "naive": the model assumes features are conditionally independent given the class, even when they are not, because the assumption simplifies the math and works surprisingly well in practice.

### Bayes' theorem

Bayes' theorem is the rule for updating what was believed before seeing the evidence into what should be believed after seeing it. The form is:

```
P(H | E) = P(E | H) · P(H) / P(E)
```

Four named quantities carry the meaning. The **prior** `P(H)` is the probability of the hypothesis before any evidence is seen. The **likelihood** `P(E | H)` is the probability of seeing this evidence assuming the hypothesis is true. The **evidence** `P(E)` is the total probability of seeing this evidence under any hypothesis. The **posterior** `P(H | E)` is the updated probability of the hypothesis once we have weighed the evidence.

Quiver exposes this directly when the three inputs are known. See <doc:Working-With-Distributions> for the broader family of distribution-based reasoning that contains `Bayes.posterior`.

```swift
import Quiver

Bayes.posterior(
    prior: 0.20,         // 20% of inbox is spam
    likelihood: 0.60,    // 60% of spam contains the phrase
    evidence: 0.15       // 15% of all mail contains the phrase
)  // ≈ 0.80
```

Eighty percent. The phrase moved the probability of spam from 20% to 80%, because the phrase is much more common in spam than in legitimate mail.

### The expanded form

In practice the marginal probability `P(E)` is rarely known directly. What is known is how often the evidence appears under each hypothesis. The expanded form computes `P(E)` from those two rates using the **law of total probability**:

```
P(H | E) = sensitivity · prior / (sensitivity · prior + falsePositiveRate · (1 − prior))
```

Quiver's expanded overload takes the three rates the developer actually has:

```swift
import Quiver

Bayes.posterior(
    prior: 0.02,                  // 2% phishing base rate for this account
    trueRate: 0.70,               // 70% of phishing uses "verify your account"
    falsePositiveRate: 0.03       // 3% of legitimate mail uses it too
)  // ≈ 0.3226
```

Thirty-two percent. The phrase is a strong signal (seventy percent of phishing uses it) but the base rate of phishing is so low that even after the evidence arrives, the message is still more likely legitimate than phishing. This is the famous counterintuitive payoff of Bayesian reasoning: rare conditions stay rare unless the evidence is overwhelming, and "rare" beats "specific" more often than intuition expects. The same math is what makes a positive medical test for a 1-in-100 disease leave the patient with only a 32% probability of actually having the disease, even when the test is 95% accurate.

> Note: When `P(E) = 0` the calculation has no answer: division by zero. `Bayes.posterior` returns `nil` in that case, and a `nil` result should be read as "the evidence is impossible under the model," not as an error to catch. The same `nil` appears when any input falls outside `[0, 1]` or the implied marginal collapses to zero.

### Comparing competing hypotheses

Real applications rarely have one hypothesis. A clinical decision compares flu, cold, and COVID; a fraud check compares fraud, account-takeover, and legitimate purchase; a spam filter weighs spam against phishing against legitimate. Multi-hypothesis Bayes generalizes the same rule to `n` mutually exclusive hypotheses, each with its own prior and its own likelihood under the observed evidence. See <doc:Inferential-Statistics-Primer> for the companion framing: confidence intervals, test statistics, and the language of weighing competing claims against evidence.

The typed inputs prevent the most common error: a prior whose probabilities do not sum to `1.0`. `BayesPrior` is a failable initializer that rejects any input whose probabilities sum outside `1.0 ± 1e-9`, so the data error is caught at construction rather than producing silently wrong posteriors:

```swift
import Quiver

guard let prior = BayesPrior(
    hypotheses: ["Flu", "Cold", "COVID"],
    probabilities: [0.60, 0.30, 0.10]
) else { return }

guard let likelihood = BayesLikelihood([
    0.10,   // P(loss of smell | Flu)
    0.05,   // P(loss of smell | Cold)
    0.70    // P(loss of smell | COVID)
]) else { return }

guard let posterior = Bayes.posterior(
    prior: prior,
    likelihood: likelihood
) else { return }

print(posterior)
// BayesPosterior(Flu: 0.4138, Cold: 0.1034, COVID: 0.4828)
```

The shift is the lesson. The prior favored flu at 60% and gave COVID only 10%. The likelihood of loss-of-smell was seven times higher under COVID than under flu. The posterior flipped: COVID now sits at 48%, flu at 41%, and cold remains the least likely. Evidence overrode the base rate, but not by as much as the likelihood ratio alone would suggest — the prior continued to pull on the answer. Both ingredients shaped the posterior, which is exactly what Bayesian reasoning delivers.

> Note: The posterior probabilities sum to `1.0` by construction, which means they can be ranked, compared, and rendered as a probability bar chart directly. Quiver computes the normalization in log-space internally (`log(prior) + log(likelihood)` followed by softmax) so the calculation stays numerically stable across hypotheses with very small or very large likelihoods.

### From a single update to a model

A Bayes update on one piece of evidence is a calculation. A Bayes update on every feature of every row in a dataset is a classifier. The <doc:Naive-Bayes> model in Quiver applies the rule from this primer independently across features and combines the per-feature likelihoods with the class prior to produce a posterior over every class for every row. The vocabulary stays exact: `ClassStats.prior` is the same `count / totalSamples` empirical probability we started with, the per-feature sum that makes Naive Bayes "naive" is the independence assumption from earlier, and the final softmax that normalizes class probabilities is the same `softMax()` call the multi-hypothesis `Bayes.posterior(prior:likelihood:)` API uses. See <doc:Naive-Bayes> for the model documentation and <doc:Machine-Learning-Primer> for the broader features-labels-training-evaluation framing that wraps the classifier.

> Experiment: **The Quiver Notebook** is the right place to feel how the prior pulls on the posterior. Hold the likelihood and false-positive rate fixed at `0.70` and `0.03`, then sweep the prior across `0.01`, `0.10`, `0.50`, and `0.90`. Watch the posterior swing from a small fraction to near-certainty as the base rate grows. The same evidence carries radically different conclusions when the underlying frequency changes. See <doc:Quiver-Notebook>.
