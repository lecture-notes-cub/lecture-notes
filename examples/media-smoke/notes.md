# Conditional Probability and Independence

## Definition of Conditional Probability

The conditional distribution of X given Y is defined as the ratio of the joint distribution to the marginal distribution of Y:

$$P(X|Y) = \frac{P(X, Y)}{P(Y)}$$

This definition applies to both continuous and discrete random variables. [00:03:04–00:03:35]

## Intuition

The conditional distribution shows how the distribution of X changes when we observe Y. This is a fundamental concept in understanding relationships between random variables. [00:03:35–00:04:12]

## Independence

Two random variables X and Y are independent if their joint distribution can be decomposed as the product of their marginal distributions:

$$P(X, Y) = P(X) \cdot P(Y)$$

[00:04:13–00:05:01]

## Conditional Distribution Under Independence

If X and Y are independent, then:

$$P(X|Y) = \frac{P(X, Y)}{P(Y)} = \frac{P(X) \cdot P(Y)}{P(Y)} = P(X)$$

This means that observing Y provides no information about X when X and Y are independent. [00:05:01–00:05:31]

## Dependent Case

When X and Y are dependent, the observation of Y changes the distribution of X. This is the opposite of the independent case where Y contains no information about X.

## Key Insight

The concept of conditional probability is central to understanding how information from one random variable affects the distribution of another.
