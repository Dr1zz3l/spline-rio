import symforce
symforce.set_epsilon_to_symbol()
import symforce.symbolic as sf
from symforce.codegen import Codegen, PythonConfig

def test_residual(R_bs: sf.Rot3, u_sensor: sf.V3, epsilon: sf.Scalar) -> sf.V3:
    return R_bs * u_sensor

codegen = Codegen.function(test_residual, config=PythonConfig())
codegen_jac = codegen.with_jacobians(which_args=["R_bs"])
data = codegen_jac.generate_function(output_dir="/tmp/sf_test")

import glob
for pyfile in glob.glob("/tmp/sf_test/symforce/**/*.py", recursive=True):
    if '__init__' not in pyfile:
        print(f"--- {pyfile} ---")
        with open(pyfile, 'r') as f:
            print(f.read())
