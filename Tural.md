Can you hear me?

Slide 1. Title

Good afternoon, everyone. Glad to see you all on my webinar.

Today I will present a research paper written by a group of scientists from China. The work is called BeautyGRPO.

It is about face retouching. It is a task that may seem simple, but is actually very difficult to do well.


· Why existing methods for face retouching do not work well
· What solution the authors propose
· How they collected their data
· What the formulas mean
· And what results they achieved

Let's begin.

Slide 2. The problem — why old methods are bad

[Show slide 2]

So, what is the problem?

The authors explain it like this. Face retouching needs two things at the same time — and these two things often fight each other.

First. We need to remove small imperfections: pimples, redness, dark circles under the eyes.

Second. We need to keep unique facial features: moles, natural skin texture, the shape of the eyes and lips.

If you remove too much — the face looks like a plastic mask. If you remove too little — the defects remain.

And existing methods, as the authors show, choose one extreme or the other. They cannot find the middle way.

Interactive moment 1 — Ask the audience

Quick question for you. Type + in the chat if you have ever used an automatic retouch tool on your phone — like a beauty filter or a one‑click skin smoother or type - if you haven’t. Now send another + if you were always happy with the result. Much fewer hands. Exactly. This is the problem we are talking about.

[pause]

The authors divide existing methods into two groups.

Group one — Supervised Fine‑Tuning, or SFT.

How does it work? Researchers take pairs of photos. One photo is the original face with defects. The other photo is the same face after a professional retoucher worked on it. Then they force the neural network to copy the result. Pixel by pixel.

What is wrong with this? The model learns to repeat, but it does not understand why the result is good. It does not know the difference between a pimple (which should be removed) and a mole (which should stay). For the model, both are just spots to remove.

As a result, the model gets stuck in copying. It can never find a solution that is better than what it saw in training. The skin becomes unnaturally smooth, texture disappears, the face looks rigid and fake.

Group two — Reinforcement Learning, or RL.

Here the model does not copy. It experiments. To make it try new things, the researchers add random noise at every step. The authors refer to a method called FlowGRPO, which adapts reinforcement learning for flow‑matching models.

This sounds good, but there is a new problem.

The noise is getting bigger. The authors call this stochastic drift. The trajectory slowly moves away from the correct path. At the end of the process, instead of a beautiful face, we get artifacts — unnatural spots, blurry areas.

So the authors identify a fundamental conflict.

Supervised learning gives you accuracy, but kills naturalness and freedom. Standard reinforcement learning gives you freedom, but creates noise and artifacts.

The question the researchers ask is: how can we combine the best of both worlds? How can we let the model explore new solutions, but not let it drift into noise?

The answer is BeautyGRPO.

[pause for transition]

---

Slide 3. FRPref‑10K — how they collected the data


Before I explain the algorithm, I need to tell you where the data comes from.

The authors created their own dataset. They called it FRPref‑10K. This stands for Face Retouching Preference — 10,000 examples.

Let me explain what is inside this dataset, step by step.


First. Source portraits.

The authors took 70 percent of the images from an open dataset called FFHQR — these are high‑quality portraits. The other 30 percent came from their own collection. They did this on purpose to have diversity: different ages, different ethnic groups, different lighting conditions.

Second. Generating candidates to compare.

For each original portrait, the authors generated many retouched versions. They used several modern models: NanoBanana, Flux.1 Kontext, RetouchFormer. And for each model, they used different random seeds — this means they ran the same model many times with slightly different starting points.

Then they formed pairs for comparison in two ways. First way — compare two different models against each other. Second way — compare a model's result against a high‑quality example from the training set.

Third. Five quality dimensions.

Here the authors did something very important. They did not just ask "like or don't like". They defined five specific, measurable dimensions.

Here they are:

1. Skin smoothing — is the skin too smooth or just right?
2. Blemish removal — are pimples gone, but moles still there?
3. Texture quality — can you see pores and natural shine?
4. Clarity — is the face sharp or blurry?
5. Identity preservation — do you still recognize the person?


Fourth. Three‑level annotation.

The authors did not trust a single person or a single neural network. They built a system with three levels.

Level one. An ensemble of three large vision‑language models. Each of them looked at a pair of images, wrote a short reasoning, and gave a score. The three opinions were averaged.

Level two. Trained human annotators checked the scores using the same criteria. If the human agreed with the AI consensus, the annotation was accepted.

Level three. If there was disagreement between humans and AIs, then professional retouchers made the final decision.


Interactive moment 2 — Ask the audience

So every one of the 10,000 pairs went through this three‑stage check. This gave the authors very clean, reliable annotations.

Slide 4. Dynamic Path Guidance (DPG) — the idea

Now we come to the heart of the method. The authors call it Dynamic Path Guidance, or DPG for short.

On the slide you see the key idea, written by the authors themselves. It says:

"BeautyGRPO offers a third path. Not copying and not making random noise, but softly guiding the model by giving it an anchor — good examples of retouching — while not forbidding it to deviate if this leads to a more beautiful result."

What does the word "anchor" mean here? Let me explain.

[pause]

An anchor is a concrete photo from the FRPref‑10K dataset. A photo that has already been retouched and is considered very good. The skin looks natural and defects are removed.

But here is the important point. The anchor is not a target image like in supervised learning. It is just a soft guide.

The model does not have to match the anchor. It can deviate if this leads to a higher score from the reward model. The anchor is only there to prevent the model from drifting too far into noise.

So at each step, the model does two things.

First. It calculates where it wants to go by itself, based on its own intuition. This is what it has learned so far.

Second. It looks at the anchor and calculates where it would go if it followed the ideal path toward the anchor.

Then it mixes these two directions.

And here is the key: the mixing proportion changes over time.

At the beginning of the process, when the image is still chaotic noise, the model listens more to the anchor.

At the end of the process, when the image already looks like a face and only fine details remain — pores, natural shine, small irregularities — the model gets more freedom. It can deviate a little, try new things, search for more beautiful results.

The authors call this "controlled stochasticity".

The main difference between DPG and the old FlowGRPO method is that the noise does not accumulate uncontrollably. At every step, there is a kind of leash that softly pulls the model back toward the safe trajectory. But the leash does not pull hard, and it does not forbid exploration.

This is what allows, in the authors' words, "reconciling exploration with high‑fidelity requirements".

[pause for transition]

---

Slide 5. What the formulas look like

[Show slide 5]

Now I will show you how these ideas look in mathematics. There are five blocks on the slide. I will explain each one in simple words. [pause]

Block 1. Step in the old FlowGRPO method

```latex
x_{t-\Delta t} = \mu_t + \sigma_{\text{step}} z_t, \quad z_t \sim \mathcal{N}(0, I)
```

What does each symbol mean?

· x — the image. A set of pixels.
· t — the current time. The process goes from t=1 (pure noise) to t=0 (clean face).
· \Delta t — the step size. A small number.
· x_{t-\Delta t} — the image at the next step. What we get after one step.
· \mu_t — the predicted drift. Where the model wants to go based on its intuition. No noise yet.
· \sigma_{\text{step}} — the noise scale for one step. How strong the random noise can be.
· z_t — random noise. A vector of random numbers from a standard normal distribution.
· \mathcal{N}(0, I) — the standard normal distribution. Zero is the average. I means each coordinate is independent.

Simple meaning: new image = where the model wants to go + random noise. This noise accumulates and causes drift.

[pause]

Block 2. The anchor‑guided target

```latex
x_{t-\Delta t}^* = \left(\frac{\Delta t}{t}\right)x_0^{\text{anchor}} + \left(1 - \frac{\Delta t}{t}\right)x_t
```

New symbols:

· x_{t-\Delta t}^* — the ideal target for the next step. Where we would go if we moved straight toward the anchor.
· x_0^{\text{anchor}} — the anchor. A good retouched image from FRPref‑10K.
· \frac{\Delta t}{t} — a simple fraction. It tells us how far we move toward the anchor in one step.

Simple meaning: draw a straight line from current state to anchor. Look where you land after one small step. That is x_{t-\Delta t}^*.

[pause]

Block 3. Correction vector

```latex
z_t^{\text{anchor}} = \frac{x_{t-\Delta t}^* - \mu_t}{\sigma_{\text{step}}}
```

New symbols:

· z_t^{\text{anchor}} — the correction vector. It shows how much and in what direction we need to change \mu_t to match the anchor path.
· x_{t-\Delta t}^* - \mu_t — the difference between "ideal anchor target" and "where the model wants to go". If this difference is large, the model has drifted far. If it is small, the model is close to the right path.
· Dividing by \sigma_{\text{step}} makes the correction comparable in size to the random noise.

Simple meaning: calculate how far the model has drifted away from the anchor path. Turn that into a correction vector.

[pause]

Block 4. Mixing the correction with random noise

```latex
z_t^{\text{mix}} = \lambda(t) z_t^{\text{anchor}} + (1 - \lambda(t)) z_t^{\text{std}}
```

New symbols:

· z_t^{\text{mix}} — the mixed, hybrid noise. This is what we will actually use.
· \lambda(t) — a time‑dependent weight. It decides how much of the correction we take versus how much random noise.
· z_t^{\text{std}} — standard random noise, the same as z_t in block 1.

How does \lambda(t) work?

```latex
\lambda(t) = \frac{t}{\max(1, T-t)}
```

Here T is the total number of steps (for example, 30 or 50).

How \lambda(t) changes over time:

· When t is large (beginning of the process), the numerator is large, denominator is small. \lambda is close to 1. The model mostly follows the anchor correction.
· When t is small (end of the process), the numerator is small, denominator is large. \lambda is close to 0. The model mostly follows random noise.

Simple meaning: take correction and random noise. Mix them. At the beginning, listen more to the anchor. At the end, listen more to randomness.

[pause]

Block 5. Final DPG step

```latex
x_{t-\Delta t}^{\text{guided}} = \mu_t + \lambda \sigma_{\text{step}} z_t^{\text{anchor}} + (1 - \lambda) \sigma_{\text{step}} z_t^{\text{std}}
```

New symbol:

· x_{t-\Delta t}^{\text{guided}} — the final image at the next step, obtained with DPG.

What this formula does:

It takes the model's intuition \mu_t. It adds the anchor correction multiplied by \lambda and the noise scale. It adds random noise multiplied by (1-\lambda) and the noise scale.



Slide 6. Results

[Show slide 6]

Now — the most important part. What results did the authors get?

In the paper there is a large table — Table 1. I will not list every number, but I will give you the main findings.

[pause]

First. Comparison with existing methods.

The authors compared BeautyGRPO with five specialized retouching models and four general editing models. 

They evaluated the results using six no‑reference quality metrics. These metrics try to measure how natural and beautiful an image looks without needing a "perfect" reference image.They did user study to compare these models

On all of these metrics, BeautyGRPO beats every existing method.



[pause]

Thank you for your attention. I am ready to answer your questions.
 Why is those 5 dimensions important? Because most existing datasets only measure global aesthetics — is the picture bright, is it colorful, is it nice overall. Here we have five different aspects. A model can be good on one dimension and bad on another. The authors wanted to capture this difference.

What VLM authors used for rating photos? GPT‑4o, Qwen2.5‑VL‑72B, and Gemini 2.5 Pro

Why does the model listen more to anchor at the beginning?
Because at the beginning, it is very easy to drift into complete nonsense. If you give the model full freedom, it will wander into noise artifacts. So early steps need more discipline.

Among the five dimensions which one do users care about most? According to the user study results in the paper, all five dimensions matter. However, the authors do not publish a ranking of importance.

100 people participated. They had different ages and different skill levels. There were two questionnaires.
Questionnaire I: Participants saw one original portrait and five retouched results from different methods. They chose the one they liked best. BeautyGRPO won 63% of the time.
Questionnaire II: Participants saw one original portrait and one retouched result. They rated the result on five dimensions (skin smoothing, blemish removal, texture, clarity, identity) using a 1–5 scale. These human scores were then compared to the reward model's predictions.
