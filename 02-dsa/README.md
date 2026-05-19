# 02 — DSA (Data Structures & Algorithms)

**Your single biggest interview risk.** If you fail the coding round, your projects don't save you.

---

## Honest assessment of where you are

You said: 50-200 LC problems solved, can solve mediums.

**Required by Month 11:** LC-medium in 25 min, LC-hard in 40 min, while explaining out loud. Pattern recognition, not memorization.

**Gap:** 200-400 more problems with timed pressure and pattern mastery.

---

## Files

1. [`01-patterns-master-list.md`](./01-patterns-master-list.md) — The 18 DSA patterns + problem assignments per pattern
2. [`02-90-day-plan.md`](./02-90-day-plan.md) — Day-by-day plan for Months 1-3 (your DSA phase)
3. [`03-mock-interview-protocol.md`](./03-mock-interview-protocol.md) — How to do mocks effectively
4. [`04-coding-during-interview.md`](./04-coding-during-interview.md) — Communication framework for live interviews
5. [`problem-tracker.md`](./problem-tracker.md) — YOUR log; fill in as you go

---

## The mental model

DSA interviews are NOT about clever tricks. They are about:

1. **Pattern recognition** — "this is a sliding window problem" — within 60 seconds
2. **Clean implementation** — translate pattern to code without bugs
3. **Communication** — narrate your thinking; interviewer is grading process, not just outcome
4. **Edge cases + complexity** — explain time/space, handle empty/single/extreme inputs

**You don't need to memorize 1000 problems. You need to recognize ~18 patterns and apply them under pressure.**

---

## The 18 patterns (covered in detail in `01-patterns-master-list.md`)

1. Two Pointers
2. Sliding Window
3. Binary Search (and variants)
4. Modified Binary Search (on rotated arrays, on answer space)
5. Cyclic Sort
6. In-place Linked List Reversal
7. Tree BFS
8. Tree DFS
9. Graph BFS / DFS / Topological Sort
10. Backtracking
11. Greedy
12. Two Heaps
13. Subsets / Combinations
14. K-way Merge
15. Dynamic Programming (1D)
16. Dynamic Programming (2D / sequence)
17. Tries
18. Union Find (DSU)

Master all 18. Average ~15-25 problems per pattern. Total: ~300-450 problems.

---

## Tools you'll use

### Primary platform: LeetCode (Premium worth it)
- $35/month or $159/year — buy 6 months of Premium starting Month 1
- Premium unlocks: company-tagged questions (filter "Google", "Meta", etc.), more solutions
- Use the **"Top Interview Questions"** list for fundamentals
- Use **"Blind 75"** list (curated 75 most important LC problems)
- Use **"NeetCode 150"** as the next tier

### Secondary tools:
- **NeetCode.io** (free) — has problem walkthroughs in video form
- **AlgoExpert** (paid, optional) — structured video course
- **Codeforces** (free) — for harder problems (if you have time)
- **Pramp** (free) — mock interviews with peers
- **interviewing.io** (paid, $200/mock) — mocks with actual FAANG engineers (worth it 2-3x)

### Language: Python
- Most ML candidates use Python for interviews. Stick with it.
- Get comfortable with: `collections` (`Counter`, `deque`, `defaultdict`), `heapq`, `bisect`, `itertools`, `functools.lru_cache`
- One alternative: C++ if you've done competitive programming. But pick ONE and master it.

---

## How to study (most important section)

### The wrong way (most people do this):
1. Read problem
2. Try for 15 minutes
3. Give up
4. Look at solution
5. Type out solution from memory
6. Mark as "solved"
7. Move on

This builds zero pattern recognition. You'll fail interviews despite "solving" 500 problems.

### The right way:

**Phase A — Initial attempt (20-45 min):**
1. Read the problem twice
2. Identify the PATTERN out loud: "this looks like sliding window because of the substring + condition"
3. Brainstorm 2-3 approaches at high level
4. Pick the best approach, code it
5. Test with examples
6. Compute complexity

**Phase B — If stuck (after ~30 min):**
1. Look at the FIRST HINT only (not the full solution)
2. Try again with the hint
3. Spend another 15-20 min

**Phase C — Looking at solution:**
1. ONLY after genuinely stuck. Read the editorial/solution.
2. Don't just memorize. Ask: "What pattern was this? Why did I miss it? What's the trigger to recognize this next time?"
3. Close the solution.
4. **Re-implement from scratch** without looking.
5. Trace through with a different example by hand.

**Phase D — Spaced repetition:**
1. Tag the problem with the pattern + difficulty + "needs review"
2. Re-solve the same problem in 3-7 days, then 2 weeks, then 1 month
3. By the third pass, you should solve it in <50% of the original time

### Track in your log:
| Date | Problem | Pattern | First-time? | Time taken | Solved without help? | Re-review by |
|---|---|---|---|---|---|---|
| 2026-05-22 | LC 76 Min Window Substring | Sliding Window | Yes | 35 min | After 1 hint | 2026-05-29 |

---

## Tactical rules

### Rule 1: Time-box your attempts
- Easy: 15 minutes
- Medium: 30 minutes
- Hard: 45 minutes

After that, look at the solution. Don't lose hours grinding one problem.

### Rule 2: Always test your code
Run through your code on paper with the example input. Trace state changes. This catches off-by-one errors that interviewers will catch.

### Rule 3: State complexity out loud
After every solve: "This is O(n log n) time, O(n) space. The bottleneck is the sort." Forces you to think about it.

### Rule 4: Once a week, do a "timed mock"
Pick 2 fresh LC-mediums you haven't seen. Set 45-min timer. Solve while narrating (record yourself if alone). Watch back. Cringe. Improve.

### Rule 5: One pattern at a time during learning
Don't randomly mix problems early on. Spend 1-2 weeks per pattern. Build deep recognition, then mix.

### Rule 6: Don't get cute
Use the simplest data structures that work. If a hashmap + array works, don't add a segment tree. Interviewers want clarity.

---

## What "good code" looks like in a 45-minute interview

```python
def longest_substring_without_repeating(s: str) -> int:
    """
    Sliding window approach: maintain a window [left, right] with all unique chars.
    Time: O(n), Space: O(min(n, alphabet_size))
    """
    char_index = {}  # char -> most recent index
    left = 0
    max_len = 0
    
    for right, char in enumerate(s):
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        char_index[char] = right
        max_len = max(max_len, right - left + 1)
    
    return max_len
```

Notice:
- Type hints
- Docstring with approach + complexity
- Meaningful variable names
- One algorithm, clearly executed
- No unused imports or extra abstractions

**What you'd say while coding:** "I'll use sliding window. The window contains chars seen so far. When I see a duplicate that's inside my window, I move the left pointer to skip past it. I'll track the last seen index of each char to do this in O(1)."

---

## Daily rhythm during DSA phase (Months 1-4)

- **Weekdays (2-3 hrs available):**
  - 1 new problem (45 min)
  - 1 re-review problem (15 min)
  - 30 min reading concepts / watching NeetCode video on a pattern
  - 15 min logging in tracker

- **Weekends (8+ hrs available):**
  - 4-5 new problems
  - 2 re-review problems
  - 1 timed mock (45 min) — every Saturday
  - Pattern deep-dive (1-2 hours per session)

---

## Targets by month

| Month | Total problems | New problems | Re-reviews | Mocks | LC-medium time |
|---|---|---|---|---|---|
| 1 | 80 | 60 | 20 | 0 | <50 min |
| 2 | 160 | 80 | 40+ | 2 | <40 min |
| 3 | 220 | 60 | 60 | 4 | <30 min |
| 4 | 280 | 60 | 80 | 6 | <25 min |

By Month 4: you should be able to do 2 LC-mediums in 45 min consistently.

---

## If you're already past the 90-day plan and want maintenance

After Phase 1 (Month 4), DSA goes into maintenance:

- 1 problem/day (15-30 min)
- Focus on company-tagged problems for your active applications
- 1 timed mock/week minimum
- Re-review old problems weekly

Don't drop it entirely. Skills decay in 4-6 weeks if untouched.

---

Now go to [`02-90-day-plan.md`](./02-90-day-plan.md).
