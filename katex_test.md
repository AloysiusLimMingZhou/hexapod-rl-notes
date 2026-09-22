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
