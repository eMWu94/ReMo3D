Formation models:
The models were prepared as follows. The formation was divided into major intervals of thicknesses that were multiples of the logging step (which for that simulation was set to 0.25 m) with the boundaries located precisely halfway between the depths of the measurement points. The locations of the interval boundaries were perturbed by adding to them a value randomly chosen from the uniform distribution in the interval [-0.125 m, 0.125 m]. Each major interval was further divided into smaller subintervals, with a thickness of approximately 0.125 m. The locations of the subinterval boundaries were perturbed by adding to them a value randomly chosen from the uniform distribution in the interval [-0.0625 m, 0.0625 m]. Resistivity values in the range of 1 to 10 Ωm were assigned to each of the major intervals, and the value of resistivity in each subinterval was randomly chosen from the normal distribution with a mean equal to the resistivity value as-signed to the major interval to which that subdivision belongs and a standard deviation of 0.5 Ωm. The second variation of the model is identical to the first, except that the intervals at the top and bottom of the model were replaced by thick uniform layers to prevent the occurrence of boundary effects. 


Borehole models:
The models containe a borehole with a constant diameter of 0.2 m filled with drilling mud with a resistivity of 0.2 Ωm, 0.35 Ωm, and 0.5 Ωm.


Logs:
For each formation model, two sets of normal and lateral resistivity logs were generated - one where true measurement depths were aligned with depths to which measurements were assigned, and one where true measurement depths were misaligned with depths to which measurements were assigned, which resulted in four distinct sets of synthetic logs:
Logs 1 - logs unaffected by boundary effects and measurement depth misalignment
Logs 2 - logs affected only by boundary effects
Logs 3 - logs affected only by measurement depth misalignment
Logs 4 - logs affected by both boundary effects and measurement depth misalignment

Borehole model with drilling mud with a resistivity of 0.35 Ωm was used to generate all synthetic logs described above.

The Logs_depth_shifts file contains true measurement depths, misaligned measurement depths and values of depth shifts.