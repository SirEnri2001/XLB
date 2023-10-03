"""
This example implements a 3D Lid-Driven Cavity Flow simulation using the lattice Boltzmann method (LBM). 
The Lid-Driven Cavity Flow is a standard test case for numerical schemes applied to fluid dynamics, which involves fluid in a square cavity with a moving lid (top boundary).

In this example you'll be introduced to the following concepts:

1. Lattice: The simulation employs a D3Q27 lattice. It's a 3D lattice model with 27 discrete velocity directions.

2. Boundary Conditions: The code implements two types of boundary conditions:

    BounceBack: This condition is applied to the stationary walls, except the top wall. It models a no-slip boundary where the velocity of fluid at the wall is zero.
    EquilibriumBC: This condition is used for the moving lid (top boundary). It defines a boundary with a set velocity, simulating the "driving" of the cavity by the lid.

4. Visualization: The simulation outputs data in VTK format for visualization. The data can be visualized using software like Paraview.

"""
# Use 8 CPU devices
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
from src.models import BGKSim, KBCSim
from src.lattice import LatticeD3Q19
from src.utils import *
from jax.config import config
from src.boundary_conditions import *
import json, codecs

precision = 'f64/f64'
config.update('jax_enable_x64', True)

class Cavity(BGKSim):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_boundary_conditions(self):

        # apply inlet equilibrium boundary condition to the top wall
        moving_wall = self.boundingBoxIndices['top']
        vel_wall = np.zeros(moving_wall.shape, dtype=self.precisionPolicy.compute_dtype)
        vel_wall[:, 0] = prescribed_vel
        self.BCs.append(BounceBackHalfway(tuple(moving_wall.T), self.gridInfo, self.precisionPolicy, vel_wall))

        # concatenate the indices of the left, right, and bottom walls
        walls = np.concatenate(
            (self.boundingBoxIndices['left'], self.boundingBoxIndices['right'],
             self.boundingBoxIndices['front'], self.boundingBoxIndices['back'],
             self.boundingBoxIndices['bottom']))
        # apply bounce back boundary condition to the walls
        self.BCs.append(BounceBackHalfway(tuple(walls.T), self.gridInfo, self.precisionPolicy))
        return

    def output_data(self, **kwargs):
        # 1: -1 to remove boundary voxels (not needed for visualization when using full-way bounce-back)
        rho = np.array(kwargs['rho'])
        u = np.array(kwargs['u'])
        timestep = kwargs['timestep']
        u_prev = kwargs['u_prev']

        u_old = np.linalg.norm(u_prev, axis=2)
        u_new = np.linalg.norm(u, axis=2)

        err = np.sum(np.abs(u_old - u_new))
        print('error= {:07.6f}'.format(err))
        fields = {"rho": rho[..., 0], "u_x": u[..., 0], "u_y": u[..., 1], "u_z": u[..., 2]}
        # save_fields_vtk(timestep, fields)

        # output profiles of velocity at mid-plane for benchmarking
        output_filename = "./profiles_" + f"{timestep:07d}.json"
        ux_mid = 0.5*(u[nx//2, ny//2, :, 0] + u[nx//2+1, ny//2+1, :, 0])
        uz_mid = 0.5*(u[:, ny//2, nz//2, 2] + u[:, ny//2+1, nz//2+1, 2])
        ldc_ref_result = {'ux(x=y=0)': list(ux_mid/prescribed_vel),
                          'uz(z=y=0)': list(uz_mid/prescribed_vel)}
        json.dump(ldc_ref_result, codecs.open(output_filename, 'w', encoding='utf-8'),
                separators=(',', ':'),
                sort_keys=True,
                indent=4)

        # Calculate the velocity magnitude
        # u_mag = np.linalg.norm(u, axis=2)
        # live_volume_randering(timestep, u_mag)

if __name__ == '__main__':
    lattice = LatticeD3Q19(precision)

    nx = 256
    ny = 256
    nz = 256

    Re = 1000.0
    prescribed_vel = 0.06
    clength = nx - 2

    visc = prescribed_vel * clength / Re
    omega = 1.0 / (3. * visc + 0.5)
    print('omega = ', omega)

    os.system("rm -rf ./*.vtk && rm -rf ./*.png")

    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'precision': precision,
        'io_rate': 10000,
        'print_info_rate': 10000,
        'downsampling_factor': 1
    }
    sim = Cavity(**kwargs)
    sim.run(1000000)