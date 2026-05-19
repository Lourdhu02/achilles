# Coding During Interview — The Communication Framework

The framework you use during a live coding interview. Print it. Practice it. Use it every mock.

---

## The 6-step framework

```
1. UNDERSTAND   (2-3 min)
2. EXAMPLES     (2 min)
3. APPROACH     (3-5 min)
4. CODE         (15-20 min)
5. TEST         (3-5 min)
6. ANALYZE      (2 min)
```

If a 45-min interview goes:
- 0-3: Understand
- 3-5: Examples
- 5-10: Approach
- 10-30: Code
- 30-38: Test (and fix bugs found)
- 38-40: Complexity
- 40-45: Discussion / follow-ups

That's a good interview.

---

## Step 1: UNDERSTAND (2-3 min)

**Goal:** Make sure you and the interviewer agree on what's being asked.

### Script:
1. After the interviewer states the problem: pause 5 seconds, don't react
2. Re-state the problem in your own words: "So, given an array of integers and a target value, I need to return the indices of two numbers that add up to the target. Is that right?"
3. Ask clarifying questions:
   - Input constraints: "Can the array be empty? How large can it be?"
   - Data types: "Are the integers positive only, or can they be negative? Floats?"
   - Edge cases: "What if there are multiple valid answers? Return any, or all?"
   - Output format: "Return indices in any order, or sorted?"
   - Assumptions: "Can I assume exactly one solution exists?"

### Mistakes to avoid:
- Skipping this step ("just diving in" looks impulsive)
- Asking too many — 2-4 sharp questions is right; 8 is too many
- Asking obvious things ("Is this Python?" — you can see the editor)

---

## Step 2: EXAMPLES (2 min)

**Goal:** Confirm understanding with a concrete trace + identify edge cases.

### Script:
- "Let me trace through your example: [step through]"
- "And an edge case might be: empty array, returns []. Single element, returns []. All same numbers, returns first match. Negatives + positives both work."
- "Does that look right?"

### Bonus:
- If you spot a subtle ambiguity ("what if input has duplicates?"), this is when to mention it.

---

## Step 3: APPROACH (3-5 min)

**Goal:** Talk through 1-2 approaches before coding. NEVER code without first describing approach.

### Script:
1. "I see two approaches. Approach 1: brute force — nested loop, O(n²). Approach 2: hashmap — single pass, O(n) time and O(n) space."
2. "I'll go with Approach 2 because it's optimal."
3. "Pseudocode: maintain a dict of value → index. For each element, check if (target - element) is in dict. If yes, return both indices. Otherwise add current to dict."
4. "Sound good?"

Wait for interviewer to confirm or push back.

### When you don't know the optimal approach:
- "The brute force is O(n²). I'm trying to think if there's a sliding window or two-pointer approach... let me think for 30 seconds."
- Then either land on the optimal or proceed with brute force + plan to optimize: "Let me start with brute force, then we can optimize together."

This is **completely normal** and interviewers respect honest, structured thinking.

---

## Step 4: CODE (15-20 min)

**Goal:** Implement cleanly while narrating.

### Narration patterns:

While typing:
- "I'll create a dict to map values to indices."
- "Looping through the array with enumerate to get both index and value..."
- "For each element, I check if its complement is already in the dict..."
- "If found, I return the pair. Otherwise, I add to dict and continue."

When you hit a snag:
- "Hmm, let me think — if the same value appears twice, this dict lookup would overwrite. Let me re-read the problem."
- "Actually, since we need indices BEFORE current, this is fine — we check before adding."

When you spot a bug:
- "Wait, there's a bug here. If [condition], this would [problem]. Let me fix."

### Code quality rules:
- Use meaningful variable names: `val_to_idx` not `d`
- Function signature: type hints if Python
- One operation per line for complex steps
- Add a 1-line docstring at top: approach + complexity
- Don't optimize while writing — clarity first, optimization second

### What NOT to do:
- Silent for >30 seconds (interviewer thinks you're stuck)
- Code, erase, code, erase repeatedly (looks panicked)
- Use overly clever syntax (one-liner list comprehensions for complex logic)
- Skip comments where the why isn't obvious

---

## Step 5: TEST (3-5 min)

**Goal:** Walk through your code with an example. Find bugs before the interviewer does.

### Script:
- "Let me trace through with the example: [1, 3, 4, 2], target 6"
- "Iteration 0: val=1, complement=5, not in dict. Add {1: 0}."
- "Iteration 1: val=3, complement=3, not in dict. Add {1: 0, 3: 1}."
- "Iteration 2: val=4, complement=2, not in dict. Add ...]"
- "Iteration 3: val=2, complement=4, FOUND at index 2. Return [2, 3]."
- "Looks correct."

Then:
- "Let me try an edge case: empty array. Loop doesn't execute. Returns... hmm, we never reach the return. Let me add a default return or handle this upfront."
- Make the fix.

### Always run through:
- The given example (sanity check)
- One edge case (empty, single, max-size, all-same)
- One "weird" case (negatives, zero, very large numbers)

---

## Step 6: ANALYZE (2 min)

**Goal:** State complexity correctly.

### Script:
- "Time complexity: O(n) — single pass through array."
- "Space complexity: O(n) — dict can hold up to n entries."
- "Could we do better space? Not without sorting first, which would make it O(n log n) time. So this is the optimal time/space tradeoff."

### What to know:
- Time and space separately
- The exact reason (which operation, which data structure)
- Whether there's a tradeoff
- Worst case AND average case if they differ

---

## Common interviewer pushbacks (and how to handle)

### "What if the input is too large to fit in memory?"
- Discuss streaming/chunking: "I'd process in batches of k elements, maintaining state across batches."
- Or external sort + two-pointer over file pointers.

### "What if we need to do this in a distributed system?"
- Discuss MapReduce-style: "Partition by hash of value. Each worker handles a subset. Aggregate at end."

### "Can you optimize space?"
- Two-pointer instead of hashmap (requires sorting first; tradeoff is time)
- Bit manipulation if values bounded
- In-place modification if allowed

### "How would you handle concurrency?"
- "If the array is read-only, no concern. If it's being modified during read, I'd use a snapshot or a lock."
- Discuss reader-writer locks if relevant.

### "What if requirements change to find all pairs?"
- "I'd need to modify to collect all matches rather than return on first. Adjust dict to store list of indices per value."

### "Can you write tests?"
- "Sure. I'd test: standard case, empty array, single element, no solution exists, multiple solutions, negative numbers, very large input."

---

## When you don't know the answer

This will happen. The way you handle it is judged.

**Bad:** Silence. Or random guessing. Or "I don't know."

**Good:**
- "Let me think out loud. The problem reminds me of [pattern X], but I'm not sure that applies here because [reason]. What if I tried [naive approach] first to get something working?"
- "I haven't seen this exact problem before. Can I take 90 seconds to think?"

Interviewers grade your reasoning under uncertainty more than your raw correctness.

---

## When you're WRONG

Interviewer points out a bug or suboptimal approach.

**Bad:** Defensive ("Actually, it works because...")
**Bad:** Frozen panic

**Good:**
- "Oh, you're right. Let me look at that..."
- "Hmm, let me trace through. [traces]. Yes, that's wrong. The fix would be..."
- "Good catch. Let me fix it."

Composure under correction is a strong signal.

---

## Time management

If you're running out of time:

- **At 30 min, only at brute force:** "Let me discuss optimization at high level even if I don't have time to code it." That salvages.
- **At 35 min, code not working:** "Let me commit to the current approach and walk through what I'd fix with more time."
- **At 40 min, no testing yet:** Walk through with example NOW. Tests find more bugs than re-reading.

---

## Specific tips for different interview platforms

### CoderPad (Google, Meta, many others)
- Real syntax highlighting
- Can run code (limited)
- Use simple Python constructs; not all libraries available
- Practice on CoderPad.io's free sandbox before real interview

### CodeSignal (Amazon OA, some others)
- Auto-graded for OAs
- Has time pressure: be FAST, optimize secondarily
- For onsite live coding: similar to CoderPad

### Google Docs (Google sometimes)
- No syntax highlighting
- Cannot run code
- Be EXTRA careful with indentation
- Practice on a real Google Doc occasionally

### Hackerrank
- Has IDE with autocomplete (some companies disable)
- Be cautious — autocomplete in interview is sometimes a trap

### Anthropic / OpenAI interviews
- Sometimes use VS Code via screen share
- Sometimes a Jupyter notebook
- Practice in both environments

---

## The "Don't Panic" protocol

You will, at some point, blank out in an interview. Here's the recovery:

1. **Pause 5 seconds. Breathe.** Silence is fine for a moment.
2. **Re-read the problem.** Out loud, slowly.
3. **State what you know.** "What I have so far: input X, want output Y. The constraint is Z."
4. **State what's stuck.** "I'm trying to figure out how to handle [thing]."
5. **Ask for time.** "Can I take 90 seconds to think?"
6. **Think in writing.** Sketch on side of editor.

Interviewers expect blanks. Recovery shows resilience.

---

## Final checklist (print this; review before any interview)

Before going in:
- [ ] Slept ≥7 hours
- [ ] Reviewed pattern templates morning of
- [ ] Set up: water, paper, pen, quiet environment, browser ready
- [ ] Tested camera + mic if remote
- [ ] Re-read this file's 6-step framework

During:
- [ ] Step 1: Restate problem, ask 2-4 questions
- [ ] Step 2: Walk through example
- [ ] Step 3: Describe approach BEFORE coding
- [ ] Step 4: Code with narration
- [ ] Step 5: Test with example + edge cases
- [ ] Step 6: State time + space complexity

After:
- [ ] Thank interviewer
- [ ] Ask 1-2 thoughtful questions about role/team
- [ ] Send thank-you email within 24 hours (to recruiter, not interviewer)
- [ ] Write debrief within 30 minutes

---

End of DSA folder. Now [`02-dsa/problem-tracker.md`](./problem-tracker.md) is your active file from here on.
