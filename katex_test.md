# KaTeX render test

## 1. Clean top-level block, single line

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

## 2. Blank line before/after has trailing spaces (simulating list-indent leftover)

Text before.
    
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
    
Text after.

## 3. aligned with single backslash line break

$$
\begin{aligned}
\delta_9 &= r_9 - V(s_9) \\
&= 4 - 2 = 2
\end{aligned}
$$

## 4. aligned with doubled backslash line break

$$
\begin{aligned}
\delta_9 &= r_9 - V(s_9) \\\\
&= 4 - 2 = 2
\end{aligned}
$$

## 5. Nested inside a list item, 4-space indented

1. Some list item text
   $$
   \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
   $$

## 6. Nested inside list item, separated by blank line (loose list paragraph)

1. Some list item text

   $$
   \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
   $$

## 7. Nested two levels deep (matches real notes structure)

1. Top level item
   * Sub bullet
     ### A heading inside the sub bullet

     $$
     \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
     $$

## 8. List item, single-line $$...$$ on its own line

1. Item text
   $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

## 9. List item, single-line aligned with \\ row break

1. Item text
   $$\begin{aligned} \delta_9 &= r_9 - V(s_9) \\ &= 4 - 2 = 2 \end{aligned}$$

## 10. Deeply nested, single-line $$...$$

1. Top level item
   * Sub bullet
     ### A heading inside the sub bullet
     $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

## 11. Loose list (blank line between items), single-line $$...$$ in first item

1. Item one text
   $$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

2. Item two text, forces loose list

## 12. Loose list, multi-line $$ block in first item

1. Item one text
   $$
   \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
   $$

2. Item two text, forces loose list
