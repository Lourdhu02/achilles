# The 18 DSA Patterns — Master List

Each pattern has: trigger signals, template, ~10-15 essential problems. Master these and you can solve 80%+ of FAANG ML Eng coding interview questions.

---

## Pattern 1: Two Pointers

### Trigger signals
- Sorted array + finding pair/triplet
- Comparing/merging two sequences
- Palindrome check
- "Container with most water" style problems

### Template
```python
left, right = 0, len(arr) - 1
while left < right:
    if condition_met(arr[left], arr[right]):
        # process
        left += 1
        right -= 1
    elif need_larger(arr[left] + arr[right]):
        left += 1
    else:
        right -= 1
```

### Essential problems (LC numbers)
- 1 Two Sum (easy — hashmap variant)
- 167 Two Sum II Sorted Array
- 15 3Sum
- 11 Container With Most Water
- 42 Trapping Rain Water
- 75 Sort Colors
- 125 Valid Palindrome
- 26 Remove Duplicates from Sorted Array
- 88 Merge Sorted Array
- 283 Move Zeroes

---

## Pattern 2: Sliding Window

### Trigger signals
- Substring or subarray problem
- "Longest/shortest with condition"
- Fixed-size or variable-size window

### Template (variable window)
```python
left = 0
result = 0
state = {}  # or counter, sum, etc.

for right in range(len(arr)):
    # expand window: add arr[right] to state
    update_state(state, arr[right])
    
    while window_invalid(state):
        # shrink window from left
        update_state(state, arr[left], remove=True)
        left += 1
    
    result = max(result, right - left + 1)
```

### Essential problems
- 3 Longest Substring Without Repeating Characters
- 76 Minimum Window Substring
- 209 Minimum Size Subarray Sum
- 567 Permutation in String
- 424 Longest Repeating Character Replacement
- 30 Substring with Concatenation of All Words (hard)
- 239 Sliding Window Maximum (uses deque)
- 1004 Max Consecutive Ones III
- 1438 Longest Subarray Absolute Diff
- 2461 Maximum Sum of Distinct Subarrays With Length K

---

## Pattern 3: Binary Search

### Trigger signals
- Sorted (or partially sorted) array
- "Find target" or "find boundary"
- Need O(log n)

### Template (canonical)
```python
left, right = 0, len(arr) - 1
while left <= right:
    mid = left + (right - left) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
return -1
```

### Essential problems
- 704 Binary Search
- 35 Search Insert Position
- 278 First Bad Version
- 374 Guess Number Higher or Lower

---

## Pattern 4: Modified Binary Search (on answer space, on rotated arrays)

### Trigger signals
- "Smallest X such that Y" → binary search on X
- Rotated sorted array
- Find peak / find boundary

### Template (binary search on answer)
```python
def can_achieve(value):
    # returns True if value is achievable
    ...

left, right = min_possible, max_possible
while left < right:
    mid = left + (right - left) // 2
    if can_achieve(mid):
        right = mid
    else:
        left = mid + 1
return left
```

### Essential problems
- 33 Search in Rotated Sorted Array
- 153 Find Minimum in Rotated Sorted Array
- 162 Find Peak Element
- 875 Koko Eating Bananas
- 1011 Capacity to Ship Packages
- 410 Split Array Largest Sum (hard)
- 4 Median of Two Sorted Arrays (hard)
- 1539 Kth Missing Positive Number
- 2563 Count the Number of Fair Pairs
- 1268 Search Suggestions System

---

## Pattern 5: Cyclic Sort

### Trigger signals
- Array of N integers in range [1, N] or [0, N-1]
- Find missing / duplicate / kth

### Template
```python
i = 0
while i < len(nums):
    correct_pos = nums[i] - 1
    if 0 <= correct_pos < len(nums) and nums[i] != nums[correct_pos]:
        nums[i], nums[correct_pos] = nums[correct_pos], nums[i]
    else:
        i += 1
```

### Essential problems
- 268 Missing Number
- 448 Find All Numbers Disappeared in an Array
- 41 First Missing Positive (hard)
- 442 Find All Duplicates in an Array
- 287 Find the Duplicate Number

---

## Pattern 6: In-place Linked List Reversal

### Trigger signals
- Reverse linked list (whole or part)
- Reorder list
- Detect cycle / find middle

### Template
```python
def reverse(head):
    prev, curr = None, head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev
```

### Essential problems
- 206 Reverse Linked List
- 92 Reverse Linked List II
- 25 Reverse Nodes in K-Group (hard)
- 234 Palindrome Linked List
- 143 Reorder List
- 141 Linked List Cycle
- 142 Linked List Cycle II
- 160 Intersection of Two Linked Lists
- 21 Merge Two Sorted Lists
- 23 Merge K Sorted Lists (hard)

---

## Pattern 7: Tree BFS (level-order traversal)

### Trigger signals
- "Level by level"
- Shortest path in tree
- Right-side / left-side view

### Template
```python
from collections import deque
queue = deque([root])
while queue:
    level_size = len(queue)
    for _ in range(level_size):
        node = queue.popleft()
        # process node
        if node.left: queue.append(node.left)
        if node.right: queue.append(node.right)
```

### Essential problems
- 102 Binary Tree Level Order Traversal
- 199 Binary Tree Right Side View
- 103 Zigzag Level Order Traversal
- 116 Populating Next Right Pointers
- 314 Vertical Order Traversal
- 297 Serialize and Deserialize Binary Tree (hard)

---

## Pattern 8: Tree DFS

### Trigger signals
- Path problems
- Tree structure / construction
- Validation problems
- Lowest Common Ancestor (LCA)

### Template (recursive)
```python
def dfs(node):
    if not node:
        return base_case
    left = dfs(node.left)
    right = dfs(node.right)
    return combine(node.val, left, right)
```

### Essential problems
- 104 Maximum Depth of Binary Tree
- 100 Same Tree
- 226 Invert Binary Tree
- 110 Balanced Binary Tree
- 543 Diameter of Binary Tree
- 124 Binary Tree Maximum Path Sum (hard)
- 236 Lowest Common Ancestor
- 98 Validate BST
- 230 Kth Smallest in BST
- 105 Construct Tree from Preorder and Inorder
- 297 Serialize and Deserialize (hard)
- 437 Path Sum III

---

## Pattern 9: Graph BFS / DFS / Topological Sort

### Trigger signals
- Adjacency list/matrix
- Shortest path (BFS)
- Connected components
- Cycle detection
- Course scheduling / dependency order

### Templates
```python
# BFS
visited = set([start])
queue = deque([start])
while queue:
    node = queue.popleft()
    # process
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)

# Topo sort (Kahn's algorithm)
in_degree = {n: 0 for n in nodes}
for u, v in edges:
    in_degree[v] += 1
queue = deque([n for n in nodes if in_degree[n] == 0])
order = []
while queue:
    n = queue.popleft()
    order.append(n)
    for nbr in graph[n]:
        in_degree[nbr] -= 1
        if in_degree[nbr] == 0:
            queue.append(nbr)
```

### Essential problems
- 200 Number of Islands
- 695 Max Area of Island
- 994 Rotting Oranges
- 207 Course Schedule (cycle detection)
- 210 Course Schedule II (topo sort)
- 269 Alien Dictionary (hard, topo)
- 743 Network Delay Time (Dijkstra)
- 787 Cheapest Flights Within K Stops (Bellman-Ford)
- 130 Surrounded Regions
- 417 Pacific Atlantic Water Flow
- 547 Number of Provinces
- 1971 Find if Path Exists in Graph

---

## Pattern 10: Backtracking

### Trigger signals
- Generate all combinations/permutations/subsets
- Constraint satisfaction (N-queens, Sudoku)
- Path finding with all paths

### Template
```python
def backtrack(state, choices):
    if is_solution(state):
        result.append(state.copy())
        return
    for choice in choices:
        if valid(choice, state):
            state.append(choice)  # make choice
            backtrack(state, remaining_choices(choices, choice))
            state.pop()  # undo choice
```

### Essential problems
- 46 Permutations
- 47 Permutations II (with duplicates)
- 78 Subsets
- 90 Subsets II
- 39 Combination Sum
- 40 Combination Sum II
- 17 Letter Combinations of a Phone Number
- 22 Generate Parentheses
- 51 N-Queens (hard)
- 79 Word Search
- 131 Palindrome Partitioning

---

## Pattern 11: Greedy

### Trigger signals
- "Find optimal" where local choices work
- Interval scheduling
- Jump game style

### Essential problems
- 55 Jump Game
- 45 Jump Game II
- 122 Best Time to Buy and Sell Stock II
- 134 Gas Station
- 435 Non-overlapping Intervals
- 56 Merge Intervals (also intervals pattern)
- 763 Partition Labels
- 1029 Two City Scheduling
- 252 Meeting Rooms
- 253 Meeting Rooms II

---

## Pattern 12: Two Heaps

### Trigger signals
- Running median
- Top-K problems with two groups

### Template
```python
import heapq
small = []  # max heap (negate values)
large = []  # min heap

def add(num):
    if not small or num <= -small[0]:
        heapq.heappush(small, -num)
    else:
        heapq.heappush(large, num)
    rebalance()

def median():
    if len(small) > len(large):
        return -small[0]
    return (-small[0] + large[0]) / 2
```

### Essential problems
- 295 Find Median from Data Stream
- 480 Sliding Window Median
- 502 IPO (hard)

---

## Pattern 13: Subsets / Combinations / Top-K

### Trigger signals
- "Top K", "Kth largest", "K most frequent"
- Generate combinations of length K

### Essential problems
- 215 Kth Largest Element in Array
- 347 Top K Frequent Elements
- 692 Top K Frequent Words
- 973 K Closest Points to Origin
- 658 Find K Closest Elements

---

## Pattern 14: K-way Merge

### Trigger signals
- Merge K sorted lists/arrays
- Kth smallest in matrix

### Essential problems
- 23 Merge K Sorted Lists (hard)
- 378 Kth Smallest Element in a Sorted Matrix
- 632 Smallest Range Covering Elements from K Lists (hard)
- 373 Find K Pairs with Smallest Sums

---

## Pattern 15: Dynamic Programming (1D)

### Trigger signals
- "Number of ways"
- "Min/max cost"
- Overlapping subproblems

### Essential problems
- 70 Climbing Stairs
- 198 House Robber
- 213 House Robber II
- 91 Decode Ways
- 322 Coin Change
- 300 Longest Increasing Subsequence
- 139 Word Break
- 152 Maximum Product Subarray
- 53 Maximum Subarray
- 121 Best Time to Buy/Sell Stock
- 740 Delete and Earn

---

## Pattern 16: Dynamic Programming (2D / sequences)

### Trigger signals
- Two strings or two sequences
- Grid problems
- "Edit distance" / "LCS" style

### Essential problems
- 1143 Longest Common Subsequence
- 72 Edit Distance (hard)
- 10 Regular Expression Matching (hard)
- 44 Wildcard Matching (hard)
- 62 Unique Paths
- 64 Minimum Path Sum
- 221 Maximal Square
- 5 Longest Palindromic Substring
- 516 Longest Palindromic Subsequence
- 354 Russian Doll Envelopes (hard)
- 718 Maximum Length of Repeated Subarray
- 312 Burst Balloons (hard, interval DP)

---

## Pattern 17: Tries

### Trigger signals
- Prefix matching
- Word search problems
- Autocomplete

### Template
```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False
    
    def insert(self, word):
        node = self
        for ch in word:
            if ch not in node.children:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.is_end = True
```

### Essential problems
- 208 Implement Trie
- 211 Design Add and Search Words (with wildcards)
- 212 Word Search II (hard; trie + backtracking)
- 1268 Search Suggestions System
- 648 Replace Words

---

## Pattern 18: Union Find (Disjoint Set Union)

### Trigger signals
- Connected components dynamically
- "Find if two nodes connected"
- Counting components

### Template
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

### Essential problems
- 547 Number of Provinces
- 200 Number of Islands (can be done with UF)
- 684 Redundant Connection
- 1319 Number of Operations to Make Network Connected
- 990 Satisfiability of Equality Equations
- 721 Accounts Merge

---

## How to use this list

### Per-pattern study (during Months 1-3):

1. Read the pattern description + template
2. Solve 3-5 problems in order of difficulty
3. After each problem, articulate WHY this pattern applied
4. Move to the next pattern only when you can recognize problems in <60 seconds

### Cross-pattern mixing (Month 3-4 onward):

1. Use LeetCode's random function within a difficulty
2. Or use Blind 75 / NeetCode 150 mixed lists
3. For each problem, identify pattern FIRST, then code

---

## Time investment

| Phase | Activity | Total problems |
|---|---|---|
| Pattern learning (Month 1-3) | 1 pattern/week, 10-15 problems each | 200-250 |
| Mixed practice (Month 3-4) | Random, timed | 50-80 |
| Maintenance (Month 5+) | 1/day, company-tagged | 60-100 |
| **Total by Month 11** | | **~350-400 problems** |

This is enough to cover all patterns deeply and pass FAANG coding interviews.

---

Next: [coding-interviews.md](coding-interviews.md)
