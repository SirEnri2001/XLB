from src.lattice import LatticeD2Q9
from src.utils import *
from termcolor import colored
import model_settings
def generate_sim_dataset(diam, ts_start, ts_end, output_offset=0, output_stride=1, init_f=None):
    precision = 'f64/f64'
    # diam_list = [10, 20, 30, 40, 60, 80]
    scale_factor = 2
    prescribed_vel = 0.003 * scale_factor
    lattice = LatticeD2Q9(precision)

    nx = int(22 * diam)
    ny = int(4.1 * diam)

    Re = 100.0
    visc = prescribed_vel * diam / Re
    omega = 1.0 / (3. * visc + 0.5)
    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'diam': diam,
        'return_fpost': True  # Need to retain fpost-collision for computation of lift and drag
    }
    # characteristic time
    tc = prescribed_vel / diam
    if ts_end < int(100 // tc):
        print(colored("WARNING: timestep_end is too small, Karman flow may not appear. Recommend value is {}".format(
            int(100 // tc)), "red"))
    sim = model_settings.Cylinder(**kwargs)
    if init_f is not None:
        loaded_data = np.load('./data/init_frame.npz')
        init_f = resample_field(loaded_data['f'], shape=(nx, ny, loaded_data['f'].shape[2]))
        generated_data = sim.run(ts_end, ts_start, output_offset, output_stride, init_f=init_f)
    else:
        generated_data = sim.run(ts_end, ts_start, output_offset, output_stride)
    print("sim completed, data postprocessing ...")
    generated_data['timestep'] = np.array(generated_data['timestep'])
    generated_data['f_poststreaming'] = np.stack(generated_data['f_poststreaming'], axis=0)
    generated_data['rho'] = np.stack(generated_data['rho'], axis=0)
    generated_data['u'] = np.stack(generated_data['u'], axis=0)
    return generated_data

def batch_generator_generate_sim_dataset(diam, ts_start, ts_end, output_offset=0, output_stride=1, init_f=None):
    precision = 'f64/f64'
    # diam_list = [10, 20, 30, 40, 60, 80]
    scale_factor = 2
    prescribed_vel = 0.003 * scale_factor
    lattice = LatticeD2Q9(precision)

    nx = int(22 * diam)
    ny = int(4.1 * diam)

    Re = 100.0
    visc = prescribed_vel * 40 / Re
    omega = 1.0 / (3. * visc + 0.5)
    kwargs = {
        'lattice': lattice,
        'omega': omega,
        'nx': nx,
        'ny': ny,
        'nz': 0,
        'precision': precision,
        'diam': diam,
        'return_fpost': True  # Need to retain fpost-collision for computation of lift and drag
    }
    # characteristic time
    tc = prescribed_vel / diam
    if ts_end < int(100 // tc):
        print(colored("WARNING: timestep_end is too small, Karman flow may not appear. Recommend value is {}".format(
            int(100 // tc)), "red"))
    sim = model_settings.Cylinder(**kwargs)
    if init_f is not None:
        loaded_data = np.load('./data/init_frame.npz')
        init_f = resample_field(loaded_data['f'], shape=(nx, ny, loaded_data['f'].shape[2]))
        return sim.run_batch_generator(ts_end, ts_start, output_offset, output_stride, init_f=init_f)
    else:
        return sim.run_batch_generator(ts_end, ts_start, output_offset, output_stride)

