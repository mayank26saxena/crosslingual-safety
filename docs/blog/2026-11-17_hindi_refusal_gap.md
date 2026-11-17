# Same question, different language: why two AI models refuse less in Hindi, and why the fix depends on the model

*Draft blog post. Not published. Links marked [link] are placeholders until the repository and preprint are public. No harmful request or
model reply is quoted anywhere in this post.*

---

Ask an open chat model something harmful in English and it will usually say no. Ask the *same* thing in Hindi and, quite often, it won't.
We spent the last few weeks measuring how big that gap is in two popular open models, Qwen3-8B and Llama-3.1-8B-Instruct. We looked inside
the models to see why it happens, and tried to fix it. Here's what we found.

## 1. The gap is large, and it isn't just "no" turning into "yes"

We took 404 harmful requests, each written natively in Hindi with an English version, and asked each model both versions. The comparison
is item by item: same request, different language.

| | Qwen3-8B | Llama-3.1-8B |
|---|---|---|
| Says a clear "no" in English | 51% | 89% |
| Says a clear "no" in Hindi | 19% | 47% |

Most safety evaluations stop at "refused or not". We graded every reply into four bins instead:

- **complied**: gave what was asked, even with a warning attached;
- **refused explicitly**: "I can't help with that";
- **refused softly**: didn't help, but didn't say no either; it lectured, warned, or changed the subject;
- **broken**: incoherent, or in the wrong language.

That's where it gets interesting. When **Qwen** stops refusing explicitly in Hindi, it mostly hedges: soft refusals replace clear ones.
When **Llama** stops, it splits between hedging and simply answering. Llama leaves Hindi harmful requests unrefused noticeably more often
than Qwen does.

A note on grading: replies were graded by an AI judge, which we checked against a fluent Hindi speaker. The boundary between "warned, then
answered" and "warned instead of answering" turned out to be genuinely hard, even for a careful human. We report where the human and judge
disagreed, and how that was resolved, in the write-up [link].

## 2. Inside the models: two different stories

Interpretability research has found that chat models carry a "refusal direction", a single direction in their internal activations that
lights up when a request should be refused. We found that direction using English prompts only, then measured how strongly Hindi prompts
activate it.

- **Qwen** activates it about as strongly for harmful Hindi requests as for English ones. The model registers that something is off; it
  just words its refusal softly.
- **Llama** activates it only about a quarter as strongly in Hindi. The signal itself is weak.

We also found a second direction that predicts *how* a model refuses: an explicit "no" versus a soft deflection. Hindi prompts sit toward
the soft end, and adding this direction to the model's activations turns soft refusals into explicit ones. As far as we know, nobody had
isolated refusal *style* this way before.

## 3. Trying to fix it

We used "gated steering". A small check looks at the refusal signal for each prompt. If the prompt looks harmful, we nudge the model's
internal state in a chosen direction while it writes its reply.

- **For Qwen**, where the problem is wording, we pushed toward explicit refusal. Hindi explicit refusals rose by 17 points, but English rose
  by almost as much, so the *gap* barely changed. When we tested the same fix on Bengali, a language we'd kept sealed, it passed its main
  test but damaged too many replies, so it failed our pre-set bar.
- **For Llama**, where the problem is the signal itself, we pushed along the refusal direction, and set the check's threshold using Hindi
  rather than English prompts. Hindi explicit refusals rose by about 15 points and the gap narrowed by about 7 points. That's real progress,
  but it fell just short of the bars we set in advance (15 and 10 points).

Neither fix is language-specific: both also nudge English. And neither made the models refuse harmless requests or do worse on a general
knowledge test, though the harmless prompts rarely triggered the gate at all, so that's a weak test.

We also searched for "Hindi neurons" that might act as a Hindi-only switch. We found a small set that carries much of the internal
English–Hindi difference, but changing them made the model refuse more in English, not in Hindi.

## 4. How we tried to keep ourselves honest

- **We wrote the rules down first.** Before each experiment we committed what we'd measure, which settings we'd try, how we'd pick one, and
  what would count as success. Changes went into dated addenda.
- **We tuned on one set of requests and tested once on another**, which is why some of our numbers are a bit lower than the first
  impressive results.
- **We report failures.** Several experiments didn't work: the Bengali confirmation, the neuron search, and the Llama fix narrowly missing
  its bars. Those results tell you as much as the successes.

## What we take away

1. **"Refuses less in Hindi" hides different failures.** Measure *what replaces* the refusal, not just whether there is one.
2. **The same gap can have different causes inside different models,** and a fix that suits one may not suit the other.
3. **Closing the gap is harder than raising refusal.** Both fixes made the models refuse more in Hindi; neither fully closed the gap,
   because they also acted on English.

Limits: two 8B models, one decoding setting, one AI judge and one human checker, and no comparison yet against simpler fixes such as a
safety system prompt. Code, pre-registrations and every number's derivation are in the repository [link]. Harmful prompts and model
replies are not redistributed.
