# Inference Architecture

## Overview

Proposals begin on the CPU in the `Bodies` class, are dispatched to the GPU for parallelized forward physics simulation and likelihood computation, then likelihoods are returned to the CPU for new proposal sampling. This loop continues until convergence.

## Diagram

```mermaid
flowchart TD
    Start([Start]) --> Bodies

    subgraph CPU
        Bodies[Bodies Class\nInitial Proposals]
        Sample[Sample New Proposals\nMetropolis-Hastings]
        Accept{Converged?}
    end

    subgraph GPU
        Physics[Forward Physics Pass\nAll Proposals in Parallel]
        Likelihood[Compute Likelihood\nPoint Cloud Matching]
    end

    Bodies -->|Proposals| Physics
    Physics --> Likelihood
    Likelihood -->|Likelihoods| Accept
    Accept -->|No| Sample
    Sample -->|New Proposals| Physics
    Accept -->|Yes| End([Return Best Scene])
```
