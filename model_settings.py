
from src.base import LBMExternalForce, LBMBase
from src.models import BGKSim, KBCSim
from src.utils import *
from src.boundary_conditions import *
import numpy as np
class Cylinder(BGKSim):
    def __init__(self, **kwargs):
        self.diam=kwargs['diam']
        self.prescribed_vel=0.006
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # Define the cylinder surface
        coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        xx, yy = coord[:, 0], coord[:, 1]
        cx, cy = 2.*self.diam, 2.*self.diam
        cylinder = (xx - cx)**2 + (yy-cy)**2 <= (self.diam/2.)**2
        cylinder = coord[cylinder]
        implicit_distance = np.reshape((xx - cx)**2 + (yy-cy)**2 - (self.diam/2.)**2, (self.nx, self.ny))
        self.BCs.append(InterpolatedBounceBackBouzidi(tuple(cylinder.T), implicit_distance, self.gridInfo, self.precisionPolicy))

        # Outflow BC
        outlet = self.boundingBoxIndices['right']
        rho_outlet = np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        # self.BCs.append(ZouHe(tuple(outlet.T), self.gridInfo, self.precisionPolicy, 'pressure', rho_outlet))

        # Inlet BC
        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        vel_inlet[:, 0] = poiseuille_profile(yy_inlet,
                                             yy_inlet.min(),
                                             yy_inlet.max()-yy_inlet.min(), 3.0 / 2.0 * self.prescribed_vel)
        self.BCs.append(Regularized(tuple(inlet.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_inlet))

        # No-slip BC for top and bottom
        wall = np.concatenate([self.boundingBoxIndices['top'], self.boundingBoxIndices['bottom']])
        vel_wall = np.zeros(wall.shape, dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(Regularized(tuple(wall.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_wall))

        # 1:-1 to remove boundary voxels (not needed for visualization when using bounce-back)

# Helper function to specify a parabolic poiseuille profile
poiseuille_profile  = lambda x,x0,d,umax: np.maximum(0.,4.*umax/(d**2)*((x-x0)*d-(x-x0)**2))


class BGKSimForce(LBMExternalForce):
    """
    BGK simulation class.

    This class implements the Bhatnagar-Gross-Krook (BGK) approximation for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @partial(jit, static_argnums=(0,))
    def collision(self, f, feq, rho, u):
        """
        BGK collision step for lattice.

        The collision step is where the main physics of the LBM is applied. In the BGK approximation,
        the distribution function is relaxed towards the equilibrium distribution function.
        """
        fneq = f - feq
        fout = f - self.omega * fneq
        return self.precisionPolicy.cast_to_output(fout)


class CylinderForce(BGKSimForce):
    def __init__(self, **kwargs):
        self.diam=kwargs['diam']
        self.prescribed_vel=0.006
        super().__init__(**kwargs)

    def set_boundary_conditions(self):
        # Define the cylinder surface
        coord = np.array([(i, j) for i in range(self.nx) for j in range(self.ny)])
        xx, yy = coord[:, 0], coord[:, 1]
        cx, cy = 2. * self.diam, 2. * self.diam
        cylinder = (xx - cx) ** 2 + (yy - cy) ** 2 <= (self.diam / 2.) ** 2
        cylinder = coord[cylinder]
        implicit_distance = np.reshape((xx - cx) ** 2 + (yy - cy) ** 2 - (self.diam / 2.) ** 2, (self.nx, self.ny))
        self.BCs.append(
            InterpolatedBounceBackBouzidi(tuple(cylinder.T), implicit_distance, self.gridInfo, self.precisionPolicy))

        # Outflow BC
        outlet = self.boundingBoxIndices['right']
        rho_outlet = np.ones((outlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(ExtrapolationOutflow(tuple(outlet.T), self.gridInfo, self.precisionPolicy))
        # self.BCs.append(ZouHe(tuple(outlet.T), self.gridInfo, self.precisionPolicy, 'pressure', rho_outlet))

        # Inlet BC
        inlet = self.boundingBoxIndices['left']
        rho_inlet = np.ones((inlet.shape[0], 1), dtype=self.precisionPolicy.compute_dtype)
        vel_inlet = np.zeros(inlet.shape, dtype=self.precisionPolicy.compute_dtype)
        yy_inlet = yy.reshape(self.nx, self.ny)[tuple(inlet.T)]
        vel_inlet[:, 0] = poiseuille_profile(yy_inlet,
                                             yy_inlet.min(),
                                             yy_inlet.max() - yy_inlet.min(), 3.0 / 2.0 * self.prescribed_vel)
        self.BCs.append(Regularized(tuple(inlet.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_inlet))

        # No-slip BC for top and bottom
        wall = np.concatenate([self.boundingBoxIndices['top'], self.boundingBoxIndices['bottom']])
        vel_wall = np.zeros(wall.shape, dtype=self.precisionPolicy.compute_dtype)
        self.BCs.append(Regularized(tuple(wall.T), self.gridInfo, self.precisionPolicy, 'velocity', vel_wall))

    @partial(jit, static_argnums=(0,))
    def get_force(self, f_postcollision, feq, rho, u):
        return state.apply_fn({'params': state.params}, u)