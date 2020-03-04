---
layout: post
published: true
title: Variational Inference with Normalizing Flows
subtitle: 'Paper by : Danilo Jimenez Rezende and Shakir Mohamed'
date: '2020-03-04'
---
### Motivation
The broad idea of Variational Inference (VI) is to approximate a hard posterior $p$ (does not have an analytical form and we cannot easily sample from it) with a distribution $q$ from a family $Q$ that is easier to sample from. The choice of this $Q$ is one of the core problems in VI. Most applications employ simple families to allow for efficient inference, focusing on mean-field assumptions. This states that our approximate distribution 

$$q(z) = \prod_{i=d}^Dq(z_d)$$

i.e. factorizes completely completely over each dimension. It turns out that this restriction significantly impacts the quality of inferences made using VI, because it cannot model any correlations between dimensions. Hence, no solution of VI will ever be able to resemble the true posterior $p$ if there is any correlation present in it. In contrast, other inferential methods such as MCMC guarantee samples from the true $p$ in the asymptotic limit. We illustrate this shortcoming of mean-field VI with a simple example below.