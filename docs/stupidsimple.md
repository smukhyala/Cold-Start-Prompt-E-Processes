# The stupid-simple version (read this out loud)

## What I was trying to find out

I have an AI that does chores on websites, like "star the email from Sarah."
I give it instructions, called a prompt.
I wanted to know: if I write a lot of different prompts, can I find one that makes it clearly better?

## What I did

I wrote a lot of prompts: 50 here, 40 there, 12 there.
I tested each one on the same chores, in fake Gmail, fake GitLab and a fake online store.
Then I checked whether the best one really beat the normal, default prompt.

## What I found

**Almost never.**
The prompt that looked best usually wasn't really better.
When I tried it on new chores, its lead disappeared.

There was one exception: a prompt I wrote on purpose that secretly had the answers in it.
That one stayed better on new chores, about 20 points better.
So my test *can* spot a real winner. There just weren't any real winners to spot.

## Why

**It's like flipping coins.**
If 50 kids each flip a coin 30 times, someone will get a lot of heads.
That doesn't mean their coin is special. They just got lucky.

The AI is like that. Run the same prompt on the same chore twice, and it can do it once and fail once.
That luck is way bigger than any real difference between prompts, at least 40 times bigger.
So the "best" prompt is mostly the luckiest one.

Two more things:
- **Different words didn't change what the AI did.** Prompts that sounded totally different made it click the same buttons.
- **It's easy to make it worse, hard to make it better.** Telling it to "explore more" made it slow and it ran out of time. Nothing made it smarter. Only giving it the actual answers helped.

## How sure am I?

**Pretty sure about this:** if you pick the top-scoring prompt from a small test, you're mostly picking luck.
It happened in all 5 real groups of prompts, across 3 different websites.

**Not sure about this:** "a better prompt doesn't exist."
My tests were too small, 30 to 80 chores each.
A somewhat better prompt could still be hiding.

## Is it a big deal?

**For people building AI agents: yes, it's useful.**
If you test prompts on a small set of tasks, the winner will look about 10 points better than it really is.
Always re-test it on new tasks against your normal prompt.

**For science: medium.**
People in other fields already knew that what you search through matters more than how you search.
What's new is showing it carefully for AI agents, and comparing against the default prompt.

## Ian's idea: it's like testing new medicines

Testing medicines works the same way.
You try many versions of a drug, check whether any beats the normal treatment, and stop early if none will.

| Medicine trial | My project |
|---|---|
| Different versions of a drug | Different prompts |
| The normal treatment | The default prompt |
| Patients | Chores (tasks) |
| "Stop, this drug works!" | "Stop, we found a better prompt!" |
| "Stop, nothing here will work." | "Stop, no prompt beats the default." |

I built both "stop" rules.
They're safe: they almost never give a wrong answer.
But right now they need lots of chores, around 60 to 240, and my tests only had 30 to 80.

## I tried making the stop rules faster

I tried three tricks:

1. **Kick out the losers early.** Once a prompt is clearly no better, stop testing it. That's about 2 times fewer runs.
2. **Don't re-test the default.** We already know how it does, so don't pay to run it again. That's about 1.5 times fewer chores.
3. **Test the promising prompts more.** That's about 2 times fewer runs, but it needs way more different chores, which we don't have.

All the tricks are still safe. They're wrong less than 5% of the time.

**The big surprise:** proving "don't bother searching" is 2 to 4 times cheaper than proving "I found a winner."
Medicine trials work the same way.
With 12 prompts, I can prove "nothing here beats the default by 15 points" in about 600 runs and 80 chores.
Proving "this one is 15 points better" takes about 2,200 runs and 180 chores.

**The real limit is chores, not money.** Benchmarks only have 30 to 80 chores per website.

## What I want to do next

1. **Write it up:** "Prompt search only pays if something beats your default, and here's how to check."
2. **Put the stop rules inside it.** The cheap "don't bother searching" proof is the stats part, and it uses Ian's math.

## What I'd need to make it airtight

| What | Why | Cost |
|---|---|---|
| Re-test the 3 "best" prompts on ~110 brand-new GitLab chores | To be sure they're not secretly better | ~$60 |
| Re-run some old tests today | To check whether the AI changed since September | ~$5 |
| Try a second AI model | To show it isn't just this one AI | ~$50–150 |
| Build a group with a real, fair great prompt (hints, not answers) | To show the method finds a real winner when one exists | ~$90 |

The first two cost about $65 and fix the biggest weakness.

## My questions for you

1. Is "does anything beat the default?" a good headline, or too negative?
2. The real limit is the number of chores. Could we make more, for example by auto-generating new chores from templates?
3. Which medicine-trial papers should I read first?
4. Should I spend ~$65 on more tests before writing?
