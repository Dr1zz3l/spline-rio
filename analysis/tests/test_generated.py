import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))        # analysis/
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))  # analysis/lib/

from codegen.generated_jacobians import radar_residual_with_jacobians, accel_residual_with_jacobians, Rot3
import numpy as np

R = Rot3([0,0,0,1])
res, Jv, Jd, Jo = radar_residual_with_jacobians(
    np.array([1.,0,0]), R, np.zeros(3), np.zeros(3),
    np.array([1.,0,0]), np.array([.07,0,0]), R, 0.5, 1e-10)
print(f'Radar: res={res}, Jv={Jv}')
assert abs(res[0] - (-0.5)) < 1e-6
assert abs(Jv[0] - (-1.0)) < 1e-6
print('Radar OK')

res2, Ja, Jd2, Jba = accel_residual_with_jacobians(
    np.array([0,0,9.81]), R, np.zeros(3), np.array([0,0,-9.81]),
    np.array([0,0,19.62]), np.zeros(3), 1e-10)
print(f'Accel: res={res2.flatten()}, Jba diag={np.diag(Jba)}')
assert np.allclose(res2.flatten(), [0,0,0], atol=1e-6)
assert np.allclose(Jba, -np.eye(3), atol=1e-6)
print('Accel OK')
print('All tests passed - no SymForce dependency!')
