# Relay Operator Strategy
-------------------------------
In order to lower Relay operators to the implementations defined in TOPI library, a compute and schedule function need to be registered to each Relay operator. However, compute and schedule functions are usually specialized for each target, and further, even for the same target, we may have multiple algorithms and implementations available. 
To deal with the complexity, we introduce operator strategy to allow developers to define a flexible lowering strategy for each operator and target.